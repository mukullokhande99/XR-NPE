# Posit-8 (great starting point)
python train_cifar10_posit.py --bits 8 --es 0 --first-last-fp32

# Posit-16 (should be very close to FP32)
python train_cifar10_posit.py --bits 16 --es 1 --first-last-fp32

# Posit-4 (aggressive; expect a drop without KD—keep first/last FP32)
python train_cifar10_posit.py --bits 4 --es 0 --first-last-fp32
