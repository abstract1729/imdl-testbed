# src/model/pgc-net/preprocessing.py

import os
import sys
import logging
from pathlib import Path
from typing import Union

import torch
from PIL import Image


# -------------------------------------------------
# 1. Add original repo to PYTHONPATH
# -------------------------------------------------

# Add PGC-Net to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PGC_REPO_PATH = PROJECT_ROOT / "models" / "pgc-net"

if str(PGC_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(PGC_REPO_PATH))


# -------------------------------------------------
# 2. Suppress unnecessary logs (optional)
# -------------------------------------------------
logging.getLogger().setLevel(logging.ERROR)


# -------------------------------------------------
# 3. Import original preprocessing
# -------------------------------------------------

from data.transforms import create_eval_transforms, MEAN, STD  # noqa: E402


# -------------------------------------------------
# 4. Preprocessor Wrapper
# -------------------------------------------------

class PGCTransform:
    """
    Wrapper over official PGC preprocessing pipeline.

    Uses:
    - PadCenterCrop
    - ToTensor
    - AppendResidual (official residual extractor)
    - Normalize (RGB + residual)

    Output:
        Tensor of shape [6, H, W]
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size

        self.transform = create_eval_transforms(
            image_size=image_size,
            mean=MEAN["dino"],
            std=STD["dino"],
        )

    def __call__(self, image: Union[Image.Image, str]) -> torch.Tensor:
        """
        Args:
            image: PIL Image OR path to image

        Returns:
            torch.Tensor: [6, H, W]
        """

        # Load if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Apply official preprocessing
        tensor = self.transform(image)

        return tensor