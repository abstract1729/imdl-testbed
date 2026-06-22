# TruFor Inference

This document contains the steps followed to run inference for **TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization** using the official pretrained checkpoints.

Inference only was performed. Training was not used.

---

## Directory Structure

Place the checkpoints under the following structure:

```text
pretrained_models/
├── noiseprint++/
│   └── noiseprint++.th
├── segformers/
│   └── mit_b2.pth
└── trufor.pth.tar
```

Required files:

- `pretrained_models/noiseprint++/noiseprint++.th`
- `pretrained_models/segformers/mit_b2.pth`
- `pretrained_models/trufor.pth.tar`

---

## Environment Setup

Create environment:

```bash
conda create -n trufor python=3.10
conda activate trufor
```

Install dependencies:

```bash
pip install -r requirements_inference.txt
```

---

## macOS (Apple Silicon) Changes

`test.py` was modified to support Apple Silicon (MPS).

Replace:

```python
device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
```

with:

```python
if torch.backends.mps.is_available():
    device = "mps"
elif gpu >= 0 and torch.cuda.is_available():
    device = f"cuda:{gpu}"
else:
    device = "cpu"

if device.startswith("cuda"):
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
```

Checkpoint loading was updated:

Replace:

```python
checkpoint = torch.load(
    model_state_file,
    map_location=torch.device(device)
)
```

with:

```python
checkpoint = torch.load(
    model_state_file,
    map_location=device,
    weights_only=False
)
```

This resolves PyTorch 2.6+ checkpoint loading issues.


---

## Running Inference

### Single Image

```bash
python test.py \
-in test.jpg \
-out outputs \
-exp trufor_ph3 \
TEST.MODEL_FILE pretrained_models/trufor.pth.tar
```

---

### Folder Inference

Input structure:

```text
test_images/
├── real/
│   ├── img1.jpg
│   └── img2.jpg
└── fake/
    ├── fake1.jpg
    └── fake2.jpg
```

Run:

```bash
python test.py \
-in test_images \
-out outputs \
-exp trufor_ph3 \
TEST.MODEL_FILE pretrained_models/trufor.pth.tar
```

Folder traversal is recursive.

---

## High Resolution Images

Images with dimensions around `4000 × 3000` may cause memory issues during attention computation on Apple Silicon devices.

Recommended approaches:

1. Patch inference (preferred)
2. Temporary resizing for local testing

Example resize:

```python
MAX_SIZE = 1024

img = img.resize(
    (new_w, new_h),
    resample=Image.Resampling.BICUBIC
)
```

Patch-based inference is preferred to preserve forensic traces.

---

## Output Files

Generated outputs:

```text
outputs/
├── image1.jpg.npz
├── image2.jpg.npz
└── ...
```

Stored arrays:

- `map` → localization mask
- `conf` → confidence map
- `score` → forgery score
- `np++` → Noiseprint output
- `imgsize`

Inspect outputs:

```python
import numpy as np

data = np.load("outputs/image1.jpg.npz")

print(data.files)
```

Expected:

```python
['map', 'conf', 'score', 'np++', 'imgsize']
```