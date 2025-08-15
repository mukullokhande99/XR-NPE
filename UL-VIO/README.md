# UL-VIO: Noise-Robust Visual-Inertial Odometry Across Numerical Precisions

UL-VIO is a Visual-Inertial Odometry (VIO) framework designed for robustness under extreme quantization and sensor noise.  
It fuses RGB imagery and IMU data with a Noise-Robust Test-Time Adaptation (NR-TTA) mechanism, enabling deployment on low-power edge devices without significant loss in accuracy.

---

## Why Quantization Matters for VIO

Traditional VIO models rely on high-precision floating-point operations (FP32/FP16) to maintain accuracy.  
In resource-constrained or embedded environments (e.g., AR glasses, drones, mobile robotics),  
high-precision arithmetic can increase latency, energy consumption, and memory footprint.

UL-VIO addresses these challenges by:

- Supporting multiple numerical formats: FP32, FP16, BF16, INT8, FP8, FP4, and Posit formats (Posit4/8/16).
- Maintaining competitive translational and rotational accuracy across different precisions.
- Demonstrating NR-TTA’s ability to adapt to quantization-induced distribution shifts in feature space.

---

## Quantization Experiments

We systematically evaluated UL-VIO under various quantization settings on the KITTI Odometry dataset.

| Precision      | Translational Error (%) | Rotational Error (%) 
|----------------|------------------------|------------------------|
| FP32           |   2.18%                 |   0.52%               |
| Posit(4,0)     |   2.75%                 |   0.62%               | 
| Posit(8,1)     |   2.50%                 |   0.58%               | 
| Posit(16,2)    |   2.20%                 |   0.55%               | 
| FP4(E2M1)      |   2.90%                 |   0.65%               | 
| Posit(8,1)+FP4 |   2.36%                 |   0.56%               | 


Key finding: NR-TTA significantly reduces performance loss from quantization combined with noise,  
making sub-INT8 precision viable for real-world VIO.

---

## Structure

### Quantization Evaluation Notebooks
- `FP4.ipynb` — FP4 evaluation  
- `FP8.ipynb` — FP8 evaluation  
- `BF16INT8.ipynb` — BF16 and INT8 evaluation  
- `Posit4_8_16.ipynb` — Posit formats evaluation  
- `Posit8+FP4.ipynb` — Mixed-precision experiment  

### Assets
- `assets/vio_precision_error.png` — Summary plot of errors across precisions

---

## Key Takeaways
- UL-VIO maintains competitive accuracy even under FP4 quantization when combined with NR-TTA.  
- Quantization-aware evaluation is essential for realistic edge deployment scenarios.  
- Posit formats can outperform IEEE FP in certain noise conditions, suggesting further opportunities for efficiency in VIO systems.

---

## Citation
```bibtex
  @article{park2024ulvio,
        author    = {Park, Jinho 
                    and Chun, Se Young 
                    and Seok, Mingoo},
        title     = {UL-VIO: Ultra-lightweight Visual-Inertial Odometry with Noise Robust Test-time Adaptation},
        journal   = {ECCV},
        year      = {2024},
  }
  @article{yang2022efficient,
          title={Efficient Deep Visual and Inertial Odometry with Adaptive Visual Modality Selection},
          author={Yang, Mingyu and Chen, Yu and Kim, Hun-Seok},
          journal={arXiv preprint arXiv:2205.06187},
          year={2022}
}
