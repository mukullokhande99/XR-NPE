# XR-NPE
## Multi-Precision Quantization for Vision and Sensor Fusion Models

This repository provides Python implementations of **multi-precision quantization** for various computer vision and sensor fusion workloads.  
It has three workloads:  
- [`Gaze-LLE`](./Gaze-LLE) – Eye gaze extraction
- [`ResNet`](./Resnet) – Image classification
- [`UL-VIO`](./UL-VIO) – Visual-Inertial Odometry 

The code supports the following quantization formats:
- **FP4**  
- **FP8**  
- **Posit4** (Posit(4,1))  
- **Posit8** (Posit(8,0))  
- **BF16**
- **Mixed-Precision**

This  facilitates  researchers and practitioners to explore the **trade-offs across accuracy, latency, and resource usage**.

---

## 📂 Repository Structure
multi-precision-quantization/
│
├── Gaze-LLE/                       # Eye gaze estimation quantization
│   ├── images             
│   ├── bf16.py       
│   ├── fp4.ipynb                
│   ├── fp8.py                
│   ├── posit4.py                     
│   ├── posit8.py                 
│   └── README.md                    
│
├── Resnet/                          # ResNet image classification quantization
│   ├── BF16/ 
|   |     │   ├── bf16_train.py            
|   |     │   ├── resnet.py   
│   ├── FP4/ 
|   |     │   ├── fp4_quant.py            
|   |     │   ├── resnet18_cifar_fp32.pth     
|   |     │   ├── resnet18_cifar_qat_wq.pth             
|   |     │   ├── resnet_cifar_fp4.py
|   |     │   ├── train_cifar10_fp4.py
│   ├── FP8/ 
|   |     │   ├── fp8_quant.py            
|   |     │   ├── resnet_cifar_fp8.py   
|   |     │   ├── train_cifar10_fp8.py               
│   ├── Posit/ 
|   |     │   ├── posit_quant.py            
|   |     │   ├── resnet18_cifar_P8_es1_qat.pth   
|   |     │   ├── resnet18_cifar_P8_es1_wq.pth                
|   |     │   ├── resnet18_cifar_fp32.pth   
|   |     │   ├── resnet_cifar_posit.py   
|   |     │   ├── train_cifar10_posit.py   
│   └── README.md
│
├── UL-VIO/                          # Visual-Inertial Odometry quantization
│   ├── BF16_INT8/ BF16INT8.ipynb
│   ├── FP4/ FP4.ipynb
│   ├── FP8/ FP8.ipynb
│   ├── Mixed_Precision/ Posit8+FP4.ipynb
│   ├── Posit/ Posit4_8_16.ipynb
│   └── README.md
│
└── README.md                        # Generic repository README


