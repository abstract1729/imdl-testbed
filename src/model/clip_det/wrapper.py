# wrappers/clipdet_wrapper.py

import torch
import yaml
import os
from pathlib import Path
import sys

# Add CLIP-Det to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLIP_DET_PATH = PROJECT_ROOT / "models" / "clip-det"

if str(CLIP_DET_PATH) not in sys.path:
    sys.path.insert(0, str(CLIP_DET_PATH))

from networks import create_architecture, load_weights

class ClipDetInference:
    """
    ClipDet inference wrapper (no modification of model logic)
    """

    def __init__(self, weights_dir: Path, model_name: str, device: str | None = None):

        # ---------------- Device ----------------
        if device is not None:
            self.device = torch.device(device)
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        # ---------------- Config ----------------
        weights_dir = Path(weights_dir)
        config_path = weights_dir / model_name / "config.yaml"

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        arch = cfg["arch"]
        weights_file = cfg["weights_file"]

        model_path = weights_dir / model_name / weights_file

        # ---------------- Model ----------------
        self.model = create_architecture(arch)
        self.model = load_weights(self.model, str(model_path))

        self.model = self.model.to(self.device)
        self.model.eval()

    # ---------------- Validation ----------------
    @staticmethod
    def _validate_input(x: torch.Tensor):
        if not torch.is_tensor(x):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        if x.ndim != 4:
            raise ValueError("Expected shape [B, 3, H, W]")

        if x.shape[1] != 3:
            raise ValueError("Expected RGB input")

    # ---------------- Forward ----------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:

        self._validate_input(x)

        x = x.to(self.device)

        out = self.model(x)

        return {
            "logits": out,
            "image_size": tuple(x.shape[-2:])
        }