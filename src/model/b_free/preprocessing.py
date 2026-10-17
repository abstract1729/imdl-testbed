# src/model/bfree/preprocessing.py

import os
import sys
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision.transforms import Compose


# -------------------------------------------------
# 1. Add B-Free repo to PYTHONPATH
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BFREE_REPO_PATH = PROJECT_ROOT / "models" / "b-free" / "code"

if str(BFREE_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(BFREE_REPO_PATH))


# -------------------------------------------------
# 2. Import official preprocessing utilities
# -------------------------------------------------

from utils.normalization import get_list_norm  # noqa: E402


# -------------------------------------------------
# 3. Preprocessing Wrapper
# -------------------------------------------------

class BFreeTransform:
    """
    Wrapper over official B-Free preprocessing pipeline.

    Uses:
    - get_list_norm(norm_type) from official repo
    - Compose transforms

    Output:
        torch.Tensor of shape [3, H, W]

    Notes:
    - No manual resizing is enforced here (as per official code)
    - Cropping is handled internally by Wrapper5crops (model side)
    """

    def __init__(self, norm_type: str):
        """
        Args:
            norm_type: normalization type from config.yaml
                       ('resnet', 'clip', 'xception', 'none')
        """

        self.norm_type = norm_type

        # Build transform pipeline using official function
        self.transform = Compose(get_list_norm(norm_type))

    def __call__(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Args:
            image: path to image OR PIL Image

        Returns:
            torch.Tensor: [3, H, W]
        """

        # Load image if path is given
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Apply official preprocessing
        tensor = self.transform(image)

        return tensor