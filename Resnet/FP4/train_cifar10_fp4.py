import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from resnet_cifar_fp4 import ResNet18_CIFAR, QuantConvBn
import os

def get_cifar10_loaders(batch_size=128, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.25),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss(reduction='sum')
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = crit(logits, y)
        loss_sum += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        tot += y.numel()
    return loss_sum / tot, 100.0 * correct / tot

def cosine_lr(optimizer, base_lr, warmup, total_steps):
    def lr_lambda(step):
        if step < warmup:
            return step / float(max(1, warmup))
        progress = (step - warmup) / float(max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    # Phase 0: FP32
    model = ResNet18_CIFAR(num_classes=10, quantize=False, first_last_fp32=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4, nesterov=True)
    epochs_fp32 = 120
    steps_per_epoch = len(train_loader)
    sched = cosine_lr(optimizer, base_lr=0.2, warmup=5 * steps_per_epoch, total_steps=epochs_fp32 * steps_per_epoch)
    best_fp32 = 0.0
    for epoch in range(epochs_fp32):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            sched.step()
        _, val_acc = evaluate(model, test_loader, device)
        if val_acc > best_fp32:
            best_fp32 = val_acc
            sd = {k: v for k, v in model.state_dict().items() if ('.aq.' not in k and '.wq.' not in k)}
            torch.save(sd, 'resnet18_cifar_fp32.pth')
        print(f'[FP32] Epoch {epoch+1}/{epochs_fp32}  ValAcc: {val_acc:.2f}%')
    print(f'Best FP32 Acc: {best_fp32:.2f}%')    

    # Phase 1: Weight-only QAT
    qat = ResNet18_CIFAR(num_classes=10, quantize=True, first_last_fp32=True).to(device)
    qat.load_state_dict(torch.load('resnet18_cifar_fp32.pth', map_location=device), strict=False)
    qat.enable_quant(False)  # disable act quant
    for m in qat.modules():
        if hasattr(m, 'wq'):
            m.wq.enable(True)
    optimizer = optim.SGD(qat.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4, nesterov=True)
    epochs_wq = 20
    sched = cosine_lr(optimizer, base_lr=0.02, warmup=2 * steps_per_epoch, total_steps=epochs_wq * steps_per_epoch)
    best_wq = 0.0
    for epoch in range(epochs_wq):
        qat.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = qat(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat.parameters(), max_norm=5.0)
            optimizer.step()
            sched.step()
        _, val_acc = evaluate(qat, test_loader, device)
        if val_acc > best_wq:
            best_wq = val_acc
            torch.save(qat.state_dict(), 'resnet18_cifar_qat_wq.pth')

    # Phase 2: Full QAT (W + A), freeze BN
    
    for m in qat.modules():
        if isinstance(m, QuantConvBn) and hasattr(m, "wq"):
            co = m.conv.out_channels
        # Re-register the buffer at the correct size on the right device
            m.wq.register_buffer("ema_maxabs_w", torch.zeros(co, device=next(m.parameters()).device))
        # We'll load real EMA values from the checkpoint anyway
            m.wq.calibrated.fill_(False)
    

# Now load the Phase-1 checkpoint
    ckpt = torch.load("resnet18_cifar_qat_wq.pth", map_location=device)
    missing, unexpected = qat.load_state_dict(ckpt, strict=False)
    print("[Phase2] load_state_dict:", "missing:", len(missing), "unexpected:", len(unexpected))
    qat.load_state_dict(torch.load('resnet18_cifar_qat_wq.pth', map_location=device), strict=False)
    qat.enable_quant(True)
    for m in qat.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
    

    optimizer = optim.SGD(qat.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    epochs_full = 80
    sched = cosine_lr(optimizer, base_lr=0.01, warmup=2 * steps_per_epoch, total_steps=epochs_full * steps_per_epoch)
    best_full = 0.0
    for epoch in range(epochs_full):
        qat.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = qat(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat.parameters(), max_norm=5.0)
            optimizer.step()
            sched.step()
        _, val_acc = evaluate(qat, test_loader, device)
        if val_acc > best_full:
            best_full = val_acc
            torch.save(qat.state_dict(), 'resnet18_cifar_fp4_qat.pth')
        print(f'[FP4-QAT] Epoch {epoch+1}/{epochs_full}  ValAcc: {val_acc:.2f}%')

    print(f'Best FP4-QAT Acc: {best_full:.2f}%')
    return best_fp32, best_full

if __name__ == '__main__':
    train()
