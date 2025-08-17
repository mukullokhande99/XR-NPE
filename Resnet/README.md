# ResNet

## Overview  
ResNet (Residual Network) is a widely used deep convolutional neural network architecture introduced in 2015 (ResNet-18, ResNet-34, ResNet-50, ResNet-101, etc.). Its key innovation is the **residual/skip connection**, which allows gradients to flow directly through identity mappings, making it possible to train very deep models without suffering from vanishing or exploding gradients.  

ResNet has become a standard benchmark for image classification tasks such as **CIFAR-10** and **ImageNet**, and its modular design makes it highly suitable for exploring **quantization-aware training (QAT)**.  

This repository focuses on **training ResNet on CIFAR-10** using different numerical formats — **BF16**, **FP8**, **FP4**, and **Posit** — to investigate trade-offs between accuracy, efficiency, and hardware suitability.  

---

![ResNe Architecture ](D1.png)

---

## ResNet Architecture  

- **Building Block**: Each ResNet block consists of two or three convolutional layers plus a **skip (identity) connection** that bypasses the block.  
- **Skip Connections**: These shortcuts allow direct gradient flow, enabling the training of very deep networks (e.g., ResNet-152).  
- **Variants**:  
  - **ResNet-18 / 34** → BasicBlock with two 3×3 convolutions.  
  - **ResNet-50 / 101 / 152** → BottleneckBlock with 1×1 → 3×3 → 1×1 convolutions.  
- **Normalization & Activation**: Each convolution is followed by **BatchNorm + ReLU** for stability.  
- **Global Average Pooling (GAP)**: Reduces features before the final fully connected layer for classification.  

For CIFAR-10, a smaller variant (ResNet-20, ResNet-32, or ResNet-44) is often used, where the first layer is adapted to handle small 32×32 images efficiently.  

**Key Advantages**:  
- Deep yet efficient training via residual learning.  
- Excellent benchmark for quantization experiments.  
- Scalable to different datasets and hardware platforms.  

---
## ResNet Quantization Experiments 


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
