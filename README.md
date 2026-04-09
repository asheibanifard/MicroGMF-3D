# MicroGMF-3D

[![Live Demo](https://img.shields.io/badge/demo-live%20viewer-7fffb0?style=flat-square&logo=github)](https://asheibanifard.github.io/MicroGMF-3D/)

**3D Gaussian Mixture Field for microscopy neuron volume reconstruction.**  
Represents fluorescence microscopy volumes (TIFF stacks) as a set of anisotropic 3D Gaussians trained end-to-end with MIP splatting supervision.

---

## Demo

### [▶ Live Interactive Demo](https://asheibanifard.github.io/MicroGMF-3D/)

Runs entirely in the browser — no install, no server. WebGL MIP splatting renderer with real-time orbit, depth sorting, and splat-scale control.

> Drag to orbit · Scroll to zoom · Right-drag to pan

### Local Real-Time Viewer

An HTTP-based viewer renders the Gaussian field from any camera angle in real time using [gsplat](https://github.com/nerfstudio-project/gsplat)'s CUDA rasterizer with MIP splatting (antialiased mode).

```bash
python viewer.py --ckpt checkpoints/gmf_refined_best.pt
# Open http://localhost:8081
```

- Drag to orbit · Scroll to zoom · Sliders for fine control  
- FPS counter overlaid on the render  
- No WebSocket required — works through any proxy (Jupyter, VS Code, SSH)

### Notebook Widget

An `ipywidgets`-based orbit viewer lives in [`renderer/main.ipynb`](renderer/main.ipynb):

```python
# Run the last cell — azimuth / elevation / radius sliders update the render live
```

---

## Method

| Component | Details |
|---|---|
| Representation | N anisotropic 3D Gaussians with `means`, `log_scales`, `quaternions`, `log_amplitudes` |
| Training loss | Volume MSE + MIP projection loss |
| MIP splatting | Per-axis `[-1,1]` normalisation + Jacobian projection + isotropic 2D filter |
| Rasterizer | Custom CUDA kernel + [gsplat](https://github.com/nerfstudio-project/gsplat) antialiased mode |
| Aspect correction | SVD-based covariance rescaling from normalised → physical voxel space |

---

## Installation

```bash
git clone --recurse-submodules https://github.com/asheibanifard/MicroGMF-3D.git
cd MicroGMF-3D
pip install torch gsplat flask pillow tifffile
# Optional: build custom CUDA kernels
bash build_cuda.sh
```

---

## Training

```bash
# Edit config.yml with your TIFF and SWC paths, then:
python run.py --config config.yml
```

---

## Inference / Export

```bash
# Render a single view
python render.py --ckpt checkpoints/gmf_refined_best.pt

# Export to PLY
python export_ply.py --ckpt checkpoints/gmf_refined_best.pt

# Export to .splat (browser viewer)
python export_splat.py --ckpt checkpoints/gmf_refined_best.pt
```

---

## Repository Structure

```
├── viewer.py              # Real-time HTTP viewer (Flask + gsplat)
├── renderer/
│   ├── renderer.py        # Pure PyTorch MIP splatting pipeline
│   ├── render_gsplat.py   # gsplat CUDA adapter with aspect correction
│   ├── main.ipynb         # Interactive notebook viewer
│   └── gsplat/            # gsplat submodule
├── cuda/                  # Custom CUDA MIP kernels
├── model.py               # Gaussian field model
├── train.py               # Training loop
├── run.py                 # Entry point
├── loss.py                # Volume + MIP losses
├── sampling.py            # Point sampling utilities
├── camera.py              # Camera utilities
└── config.yml             # Training configuration
```

---

## Citation

If you use this code, please cite:

```bibtex
@misc{microgmf2026,
  title   = {MicroGMF-3D: 3D Gaussian Mixture Field for Microscopy Volume Reconstruction},
  author  = {Sheibanifard, Armin},
  year    = {2026},
  url     = {https://github.com/asheibanifard/MicroGMF-3D}
}
```
