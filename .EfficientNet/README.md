# EfficientNet Quantization Experiments   

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
│   └── FP4_cifar.py  
│   ├── FP4_imagenet.py  
├── FP8.py  
│  
└── Posit.py  
