
import os, math, argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from resnet_cifar_posit import ResNet18_CIFAR_POSIT
from posit_quant import make_weight_quant_fn, make_grad_quant_fn, make_momentum_quant_fn

# Optional integration with jeffreyyu0602/quantized-training
HAVE_QT = False
try:
    from quantized_training import add_qspec_args, quantize
    HAVE_QT = True
except Exception:
    HAVE_QT = False

try:
    from qtorch_plus.optim import OptimLP
    HAVE_OPTIMLP = True
except Exception:
    HAVE_OPTIMLP = False

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
    # Our posit knobs
    ap.add_argument('--bits', type=int, default=8, choices=[4,8,16], help='posit nsize')
    ap.add_argument('--es', type=int, default=0, help='posit exponent size')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--epochs-fp32', type=int, default=120)
    ap.add_argument('--epochs-wq', type=int, default=20)
    ap.add_argument('--epochs-full', type=int, default=80)
    ap.add_argument('--lr-fp32', type=float, default=0.2)
    ap.add_argument('--lr-wq', type=float, default=0.02)
    ap.add_argument('--lr-full', type=float, default=0.01)
    ap.add_argument('--first-last-fp32', action='store_true', default=True)
    ap.add_argument('--use-lp-optimizer', action='store_true')
    ap.add_argument('--use-qt', action='store_true', help='also apply quantized-training.quantize(model, args) if available')

    # If the repo is installed, expose its CLI flags too (so users can override via args)
    if HAVE_QT:
        add_qspec_args(ap)

    args = ap.parse_args()
    # Clamp es to valid range (defensive)
    if args.es <= 0:
        print("[WARN] es <= 0 is not supported by qtorch-plus; using es=1")
        args.es = 1
    elif args.es > 3:
        print("[WARN] es > 3 is not supported by qtorch-plus; clamping to 3")
        args.es = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch)
    steps_per_epoch = len(train_loader)
    criterion = nn.CrossEntropyLoss()

    ckpt_fp32 = 'resnet18_cifar_fp32.pth'
    tag = f'P{args.bits}_es{args.es}'
    ckpt_wq = f'resnet18_cifar_{tag}_wq.pth'
    ckpt_full = f'resnet18_cifar_{tag}_qat.pth'

    # ---------------- Phase 0: FP32 ----------------
    if os.path.exists(ckpt_fp32):
        print(f'[FP32] Found existing {ckpt_fp32}, skipping training.')
    else:
        model = ResNet18_CIFAR_POSIT(num_classes=10, quantize=False, first_last_fp32=True).to(device)
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

    # Build a quantized model skeleton
    qat = ResNet18_CIFAR_POSIT(
        num_classes=10, quantize=True, first_last_fp32=True,
        w_bits=args.bits, w_es=args.es, a_bits=args.bits, a_es=args.es
    ).to(device)
    qat.load_state_dict(torch.load(ckpt_fp32, map_location=device), strict=False)

    # ---------------- Optional: apply quantized-training quantize() ----------------
    # Note: The repo supports integer/FP4/FP6/FP8/posit and exposes add_qspec_args/quantize().
    if args.use_qt and HAVE_QT:
        print('[INFO] Applying quantized-training.quantize on the model with provided args')
        quantize(qat, args)

    # ---------------- Phase 1: Weight-only Posit ----------------
    qat.enable_quant(False)  # disable activation quant
    for m in qat.modules():
        if hasattr(m, 'wq'):
            m.wq.enable(True)

    opt = optim.SGD(qat.parameters(), lr=args.lr_wq, momentum=0.9, weight_decay=5e-4, nesterov=True)
    if args.use_lp_optimizer and HAVE_OPTIMLP:
        opt = OptimLP(
            opt,
            weight_quant   = make_weight_quant_fn(args.bits, args.es, 'nearest'),
            grad_quant     = make_grad_quant_fn(args.bits, args.es, 'nearest'),
            momentum_quant = make_momentum_quant_fn(16, 1, 'nearest'),
        )

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
        print(f'[P{args.bits}-WQ] Epoch {epoch+1}/{args.epochs_wq}  ValAcc: {val_acc:.2f}%')
    print(f'[P{args.bits}-WQ] Best Acc: {best_wq:.2f}%')

    # ---------------- Phase 2: Full QAT (W + A) ----------------
    qat.load_state_dict(torch.load(ckpt_wq, map_location=device), strict=False)
    qat.enable_quant(True)
    for m in qat.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)

    opt = optim.SGD(qat.parameters(), lr=args.lr_full, momentum=0.9, weight_decay=5e-4, nesterov=True)
    if args.use_lp_optimizer and HAVE_OPTIMLP:
        opt = OptimLP(
            opt,
            weight_quant   = make_weight_quant_fn(args.bits, args.es, 'nearest'),
            grad_quant     = make_grad_quant_fn(args.bits, args.es, 'nearest'),
            momentum_quant = make_momentum_quant_fn(16, 1, 'nearest'),
        )

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
        print(f'[P{args.bits}-QAT] Epoch {epoch+1}/{args.epochs_full}  ValAcc: {val_acc:.2f}%')

    print(f'[P{args.bits}-QAT] Best Acc: {best_full:.2f}%')

if __name__ == '__main__':
    main()
