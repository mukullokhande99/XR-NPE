# Multi-Precision Quantization for Vision Models

This repository includes Python implementations for **multi-precision quantization** on [Gazelle](https://github.com/fkryan/gazelle) architecture.  
It includes implementations for:  
- **FP4**  
- **FP8**  
- **Posit4** (Posit(4,1))  
- **Posit8** (Posit(8,0))
- **BF16**

## Requirements
Install the dependencies:
```bash
pip install torch torchvision timm matplotlib opencv-python scipy transformers
```

## Usage
### 1. Clone the Gazelle Repository
The notebooks assumes the [Gazelle](https://github.com/fkryan/gazelle) repository to be present.  
You can either run the given code cells that do this automatically or manually execute it:
```bash
git clone https://github.com/fkryan/gazelle.git
```
<table>
  <tr>
    <td align="center">
      <img src="images/fp32.png" alt="FP32 Model" width="400px" /><br>
      <b>FP32 Model</b>
    </td>
    <td align="center">
      <img src="images/fp4.png" alt="FP4 Model" width="400px" /><br>
      <b>FP4 Model</b>
    </td>
  </tr>
</table>

## Perform Model Analysis
The notebooks include utility functions to:
- **Analyze memory usage**  
  `analyze_model_memory_usage` – returns total parameters, quantized parameters, and model size.

- **Benchmark inference**  
  `benchmark_inference` – measures inference time over multiple times.

- **Visualize activations**  
  `visualize_heatmap_on_image` – overlays heatmaps on images.
