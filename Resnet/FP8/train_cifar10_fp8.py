import os, math, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from resnet_cifar_fp8 import ResNet18_CIFAR_FP8, QuantConvBn_FP8

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
    crit = nn.CrossEntropyLoss(reduction='sum')
    tot, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += crit(logits, y).item()
        pred = logits.argmax(1)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--epochs-fp32', type=int, default=120)
    ap.add_argument('--epochs-wq', type=int, default=20)
    ap.add_argument('--epochs-full', type=int, default=80)
    ap.add_argument('--lr-fp32', type=float, default=0.2)
    ap.add_argument('--lr-wq', type=float, default=0.02)
    ap.add_argument('--lr-full', type=float, default=0.01)
    ap.add_argument('--first-last-fp32', action='store_true', default=True)
    ap.add_argument('--w-fmt', type=str, default='e5m2', choices=['e5m2','e4m3'])
    ap.add_argument('--a-fmt', type=str, default='e4m3', choices=['e5m2','e4m3'])
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch)
    steps_per_epoch = len(train_loader)
    criterion = nn.CrossEntropyLoss()

    ckpt_fp32 = 'resnet18_cifar_fp32.pth'           # reuse your strong float baseline if present
    tag = f'FP8_W{args.w_fmt}_A{args.a_fmt}'
    ckpt_wq = f'resnet18_cifar_{tag}_wq.pth'
    ckpt_full = f'resnet18_cifar_{tag}_qat.pth'

    # ---------------- Phase 0: FP32 (skip if exists) ----------------
    if os.path.exists(ckpt_fp32):
        print(f'[FP32] Found existing {ckpt_fp32}, skipping training.')
    else:
        from resnet_cifar_fp4 import ResNet18_CIFAR  # same clean CIFAR-ResNet (no quantizers)
        model = ResNet18_CIFAR(num_classes=10, quantize=False, first_last_fp32=True).to(device)
        opt = optim.SGD(model.parameters(), lr=args.lr_fp32, momentum=0.9, weight_decay=5e-4, nesterov=True)
        sched = cosine_lr(opt, base_lr=args.lr_fp32, warmup=5 * steps_per_epoch, total_steps=args.epochs_fp32 * steps_per_epoch)
        best = 0.0
        for epoch in range(args.epochs_fp32):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); sched.step()
            _, val_acc = evaluate(model, test_loader, device)
            if val_acc > best:
                best = val_acc
                sd = {k: v for k, v in model.state_dict().items() if ('.aq.' not in k and '.wq.' not in k)}
                torch.save(sd, ckpt_fp32)
            print(f'[FP32] Epoch {epoch+1}/{args.epochs_fp32}  ValAcc: {val_acc:.2f}%')
        print(f'[FP32] Best Acc: {best:.2f}%')

    # Build FP8-QAT model from float
    qat = ResNet18_CIFAR_FP8(num_classes=10, quantize=True, first_last_fp32=True,
                             w_fmt=args.w_fmt, a_fmt=args.a_fmt).to(device)
    # load float baseline
    sd = torch.load(ckpt_fp32, map_location=device)
    qat.load_state_dict(sd, strict=False)

    # ---------------- Phase 1: Weight-only FP8 ----------------
    qat.enable_quant(False)  # disable activation quant
    # ensure weight quant is ON
    for m in qat.modules():
        if isinstance(m, QuantConvBn_FP8) and hasattr(m, 'wq'):
            m.wq.enable(True)

    opt = optim.SGD(qat.parameters(), lr=args.lr_wq, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = cosine_lr(opt, base_lr=args.lr_wq, warmup=2 * steps_per_epoch, total_steps=args.epochs_wq * steps_per_epoch)
    best_wq = 0.0
    for epoch in range(args.epochs_wq):
        qat.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = qat(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat.parameters(), 5.0)
            opt.step(); sched.step()
        _, val_acc = evaluate(qat, test_loader, device)
        if val_acc > best_wq:
            best_wq = val_acc
            torch.save(qat.state_dict(), ckpt_wq)
        print(f'[FP8-WQ] Epoch {epoch+1}/{args.epochs_wq}  ValAcc: {val_acc:.2f}%')
    print(f'[FP8-WQ] Best Acc: {best_wq:.2f}%')

    # ---------------- Phase 2: Full QAT (W + A) ----------------
    qat.load_state_dict(torch.load(ckpt_wq, map_location=device), strict=False)
    qat.enable_quant(True)
    for m in qat.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)

    opt = optim.SGD(qat.parameters(), lr=args.lr_full, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = cosine_lr(opt, base_lr=args.lr_full, warmup=2 * steps_per_epoch, total_steps=args.epochs_full * steps_per_epoch)
    best_full = 0.0
    for epoch in range(args.epochs_full):
        qat.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = qat(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat.parameters(), 5.0)
            opt.step(); sched.step()
        _, val_acc = evaluate(qat, test_loader, device)
        if val_acc > best_full:
            best_full = val_acc
            torch.save(qat.state_dict(), ckpt_full)
        print(f'[FP8-QAT] Epoch {epoch+1}/{args.epochs_full}  ValAcc: {val_acc:.2f}%')

    print(f'[FP8-QAT] Best Acc: {best_full:.2f}%')

if __name__ == '__main__':
    main()

