import os, math, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Reuse your CIFAR-optimized ResNet18 (Conv->BN->ReLU order; CIFAR stem)
from resnet import ResNet18_CIFAR  # model is fine for float/BF16 too

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
def evaluate(model, loader, device, use_bf16=True):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction='sum')
    tot, correct, loss_sum = 0, 0, 0.0
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(use_bf16 and device.type=='cuda'))
    with amp_ctx:
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
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--lr', type=float, default=0.2)
    ap.add_argument('--channels-last', action='store_true', help='use NHWC memory format')
    ap.add_argument('--save', type=str, default='resnet18_cifar_bf16_amp.pth')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # Optional speed-ups on Ampere+:
        torch.backends.cudnn.benchmark = True
        # (BF16 uses Tensor Cores; TF32 setting helps only when running FP32 matmuls)
        torch.set_float32_matmul_precision('high')

    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch)
    steps_per_epoch = len(train_loader)

    model = ResNet18_CIFAR(num_classes=10, quantize=False, first_last_fp32=True).to(device)  # pure float model
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = cosine_lr(optimizer, base_lr=args.lr, warmup=5 * steps_per_epoch, total_steps=args.epochs * steps_per_epoch)

    # AMP BF16 context (no GradScaler needed for BF16)
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type=='cuda'))

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if args.channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                logits = model(xb)
                loss = criterion(logits, yb)
            # Backward stays in FP32; AMP handles casts internally for BF16
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

        val_loss, val_acc = evaluate(model, test_loader, device, use_bf16=True)
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), args.save)
        print(f'[BF16-AMP] Epoch {epoch+1}/{args.epochs}  ValAcc: {val_acc:.2f}%')

    print(f'Best BF16-AMP Acc: {best:.2f}%')

if __name__ == '__main__':
    main()
