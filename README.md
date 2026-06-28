# U-Net Image & Video Inpainting

> A U-Net that reconstructs missing regions of face images — trained on **CelebA**
> and extended to **video** by inpainting frame-by-frame, with an interactive
> Gradio demo.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

---

## Overview

Image inpainting reconstructs corrupted or missing parts of an image from
surrounding context. This project trains a **U-Net** (convolutional encoder–decoder
with skip connections, so high-resolution detail survives the bottleneck) to fill
randomly masked rectangular regions of CelebA faces, then applies the same model
to video by processing each frame and reassembling with FFmpeg.

**Key features**
- U-Net with skip connections; **L1 reconstruction loss**.
- On-the-fly random rectangular masking for self-supervised training.
- Quantitative evaluation with **PSNR / SSIM** (skimage).
- Frame-by-frame **video inpainting** (FFmpeg) for temporally consistent results.
- Interactive **Gradio** UI for trying the model.

---

## Results

Trained 10 epochs on a CelebA subset:

| Metric | Test value |
|---|---|
| **PSNR** | **27.52 dB** |
| **SSIM** | **0.923** |

Example reconstructions and a sample inpainted video ship with the repo:

```
Report/Output_Image.png   ·   Report/Output_Video.mp4   ·   Report/UI.png
```

---

## Architecture

```
Input (masked 3×H×W)
  enc1 → enc2 → enc3 → bottleneck
                         │  (skip connections)
  dec3 ← dec2 ← dec1 ← ──┘
  → out_conv → reconstructed 3×H×W      Loss: L1(reconstructed, original)
```

Masked region = mid-gray (0 on the [-1,1] tensor); the network learns to
regenerate it from context.

---

## Repository Contents

```
.
├── Code/CV_Final.ipynb                 # full pipeline: data → train → eval → video → UI
├── Input/
│   ├── Face1.png · Face2.png           # sample stills
│   ├── Face1.mp4 · Face2.mp4           # sample input videos
│   └── inpainting_unet_final.pth       # trained model weights (~15 MB)
├── Report/                             # HTML run logs, output image/video, UI screenshot
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
pip install -r requirements.txt          # plus FFmpeg available on PATH
```

- **Inference (no training):** load `Input/inpainting_unet_final.pth` in the
  notebook and run the single-image / video / Gradio sections.
- **Training from scratch:** the notebook downloads **CelebA** from Kaggle
  (`jessicali9530/celeba-dataset`). You must supply your **own** Kaggle API token
  (`kaggle.json`) — it is intentionally **not** included in this repo. See
  [Kaggle API docs](https://github.com/Kaggle/kaggle-api).

---

## Tech Stack

Python · PyTorch · torchvision · scikit-image (PSNR/SSIM) · FFmpeg · Gradio

## License

MIT — see [LICENSE](LICENSE).

## Contact

**Mahdiar Harandi** — harandimahdiar@gmail.com
[GitHub](https://github.com/MahdiarHarandi) · [LinkedIn](https://www.linkedin.com/in/mahdiarharandi)
