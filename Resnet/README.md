# ResNet Quantization Experiments on CIFAR-10

This repository contains experiments for training ResNet on CIFAR-10 with different quantization formats: **BF16**, **FP8**, **FP4**, and **Posit**.

## Environment Setup

Create a new Conda environment with **Python 3.9**, **PyTorch 2.x**, and **CUDA 11.8**:

```bash
conda create -n resnet-quant python=3.9
conda activate resnet-quant
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
## Training Scripts

### BF16
```bash
cd BF16
python bf16_train.py
```
### FP4
```bash
cd FP4
python train_cifar10_fp4.py
```
### FP8
Two variations are provided:  
(a) FP8 with common defaults:  
Weights: E5M2  
Activations: E4M3  
```bash
cd FP4
python train_cifar10_fp8.py --w-fmt e5m2 --a-fmt e4m3 --first-last-fp32
```

(b) FP8 with E4M3 for both weights & activations:
```bash
cd FP4
python train_cifar10_fp8.py --w-fmt e4m3 --a-fmt e4m3 --first-last-fp32
```

### Posit
Setup:
```bash
git clone https://github.com/minhhn2910/QPyTorch.git
cd QPyTorch && pip install -e .
pip install ninja
```

Three variations are provided:  

(a) Posit-4 :
```bash
cd Posit
python train_cifar10_posit.py --bits 4 --es 0 --first-last-fp32
```

(b) Posit-8:  
```bash
cd Posit
python train_cifar10_fp8.py --w-fmt e4m3 --a-fmt e4m3 --first-last-fp32
```

(c) Posit-16 :
```bash
cd Posit
python train_cifar10_posit.py --bits 16 --es 1 --first-last-fp32
```
