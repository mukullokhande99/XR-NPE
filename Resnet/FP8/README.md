# FP8 with common defaults: weights E5M2, activations E4M3
python train_cifar10_fp8.py --w-fmt e5m2 --a-fmt e4m3 --first-last-fp32

# Try both E4M3 (weights) / E4M3 (acts)
python train_cifar10_fp8.py --w-fmt e4m3 --a-fmt e4m3 --first-last-fp32
