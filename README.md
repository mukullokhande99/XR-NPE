# XR-NPE
## Multi-Precision Quantization for Vision and Sensor Fusion Models

This repository provides Python implementations of **multi-precision quantization** for various computer vision and sensor fusion workloads.  
It has three workloads:  
- [`Gaze-LLE`](./Gaze-LLE) – Eye gaze extraction using low-latency estimation.  
- [`ResNet`](./Resnet) – Image classification on ImageNet using ResNet architecture.  
- [`UL-VIO`](./UL-VIO) – Visual-Inertial Odometry for sensor fusion tasks.

The code supports the following quantization formats:
- **FP4**  
- **FP8**  
- **Posit4** (Posit(4,1))  
- **Posit8** (Posit(8,0))  
- **BF16**

This  facilitates  researchers and practitioners to explore the **trade-offs across accuracy, latency, and resource usage**.

---

## 📂 Repository Structure

