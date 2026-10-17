# preprocessing/clipdet_preprocess.py

from PIL import Image
import torch
import yaml
from types import SimpleNamespace
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode

# Add CLIP-Det to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLIP_DET_PATH = PROJECT_ROOT / "models" / "clip-det"

if str(CLIP_DET_PATH) not in sys.path:
    sys.path.insert(0, str(CLIP_DET_PATH))

from utils.processing import make_normalize

class ClipDetTransform:
    def __init__(self, norm_type="resnet", patch_size=None):
        transform = []
        # Only apply resize if explicitly required
        if patch_size == "Clip224":
            transform.append(Resize(224))
            transform.append(CenterCrop(224))

        # For Corvi2023 → patch_size = None → NO resize/crop forced
        transform.append(make_normalize(norm_type))
        self.transform = Compose(transform)

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        x = self.transform(image)
        return x