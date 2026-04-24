# MetaDepth

Efficient image to 3D geometry foundation models from Meta Reality Labs for monocular depth, point maps, and surface normals. Featuring HyDen (ICLR 2026).

This reposity contains the official implementation of the following paper(s):

## **HyDen: Hybrid Dual Path Encoder for High-resolution Monocular Geometry Estimation**

Zaiwei Zhang, Marc Mapeke, Wei Ye, Rakesh Ranjan, JQ Huang

*ICLR 2026* | [OpenReview](https://openreview.net/forum?id=2eL6yXLCh8)

HyDen introduces a hybrid CNN+ViT dual-path encoder for sharp, accurate high-res geometry with up to 10× faster 4K inference. We train via self-distillation: a frozen teacher pseudo-labels unlabeled high-res images, then we only train the CNN encoder, lightweight fusion layers, and decoder. Sharp predictions, much lower latency at very high resolutions.


## News

- **[Coming ~early May 2026]** Releasing HyDen-MoGeV2 model code and checkpoints for metric 3D point maps and surface normal prediction (~2 week ETA).
- **[April 24, 2026]** Released [HyDen-DA2](https://huggingface.co/facebook/hyden-da2-relative-depth) model code and checkpoint for relative depth estimation.



## Codebase Structure

MetaDepth is a modular codebase for model architecture definition and model loading/formatting.

| Module | Description |
|--------|-------------|
| `cnn/` | Lightweight CNN encoder (MobileNetV2-style, 7.37M params). Extracts multi-scale features at strides 2, 4, 8, 16, and 32. |
| `da2/` | Depth Anything 2 — DINOv2 ViT encoder + DPT decoder. |
| `mogev2/` | MoGe V2 — DINOv2 encoder + ConvStack decoder that can jointly predict 3D maps, validity masks, and metric scale factors. |
| `example_code/` | Example inference scripts demonstrating model usage. See [Usage](#usage). |

## Getting Started

### Model Checkpoints For Pre-trained Models

| Model | Description | Download |
|-------|-------------|----------|
| HyDen-DA2-Large | Relative depth estimation | [HuggingFace](https://huggingface.co/facebook/hyden-da2-relative-depth) |
| HyDen-MoGeV2-Large-Surface-Normal | Surface normal prediction | - |
| HyDen-MoGeV2-Large-Metric-Point | Metric 3D point estimation | - |

### Installation

We recommend creating a virtual environment (see [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), [uv](https://docs.astral.sh/uv/getting-started/installation/)) then cloning this repository and installing the required packages:

```bash
git clone https://github.com/facebookresearch/metadepth.git
cd metadepth
pip install torch torchvision
```

### Usage

See [`example_code/hyden_load_and_inference.py`](example_code/hyden_load_and_inference.py) for a complete example covering all three model types.

```python
import torch
from PIL import Image
from torchvision import transforms

from da2 import HyDenDepthAnything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

raw_image = Image.open("your/image/path.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = HyDenDepthAnything(encoder="vitl")
model.load_state_dict(torch.load("checkpoints/hyden_da2_vitl.pth", map_location="cpu", weights_only=True))
model = model.to(DEVICE).eval()

with torch.no_grad():
    depth = model(transform(raw_image).unsqueeze(0).to(DEVICE))  # (1, H, W)
```

## Citation

If you find our code or paper useful, please consider citing:

```bibtex
@inproceedings{
zhang2026hyden,
title={Hyden: A Hybrid Dual-Path Encoder for Monocular Geometry of High-resolution Images},
author={Zaiwei Zhang and Marc Mapeke and Wei Ye and Rakesh Ranjan and JQ Huang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=2eL6yXLCh8}
}
```

## License

MetaDepth is licensed under the FAIR Noncommercial Research License, as found in
the [LICENSE](LICENSE) file.

## Acknowledgements

This project makes use of the excellent [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2), [MoGe](https://github.com/microsoft/moge), and [DINOv2](https://github.com/facebookresearch/dinov2) libraries. We are very grateful for their work.
