# EfficientNet QAT Quantization Experiments   

## Environment Setup

Create a new Conda environment with **Python 3.9**, **PyTorch 2.x**, and **CUDA 11.8**:

```bash
conda create -n resnet-quant python=3.9
conda activate resnet-quant
pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```
## Structure
EfficientNet/  
│  
├── FP4/  
│   └── FP4_cifar.pynb   
│   ├── FP4_imagenet.pynb   
├── FP8.pynb   
│  
└── Posit.pynb   

# FP4 Quantization training plot
<img width="1107" height="362" alt="Screenshot from 2025-08-15 21-02-46" src="https://github.com/user-attachments/assets/bfcfe6ec-1593-40b4-8fbc-67cf9ef754af" />
