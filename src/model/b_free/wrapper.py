# src/model/bfree/wrapper.py

import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


# -------------------------------------------------
# 1. Add B-Free repo to PYTHONPATH
# -------------------------------------------------

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[3]
BFREE_REPO_PATH = PROJECT_ROOT / "models" / "b-free" / "code"

if str(BFREE_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(BFREE_REPO_PATH))


# -------------------------------------------------
# 2. Import official functions
# -------------------------------------------------

from networks import get_network, load_weights  # noqa: E402
from main_bfree_single import get_config        # noqa: E402


# -------------------------------------------------
# 3. Wrapper
# -------------------------------------------------

class BFreeInference:
    """
    Wrapper for B-Free model (detection only).

    Responsibilities:
    - Load model using official functions
    - Load checkpoint
    - Handle device (CPU / MPS only)
    - Return raw logits (NO postprocessing)
    """

    def __init__(self, model_name: str = "BFREE_dino2reg4",
                 weights_dir: str = None,
                 device: str = None):
        """
        Args:
            model_name: model folder name inside weights/
            weights_dir: path to weights directory (optional)
            device: 'cpu' | 'mps' | None (auto)
        """

        # ----------------------------
        # Device selection (NO CUDA)
        # ----------------------------
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            # Prevent accidental CUDA usage
            if "cuda" in device:
                raise ValueError("CUDA is not supported in this wrapper. Use 'cpu' or 'mps'.")
            self.device = torch.device(device)

        # ----------------------------
        # Load config (official)
        # ----------------------------
        if weights_dir is None:
            weights_dir = os.path.join(BFREE_REPO_PATH, "weights")

        _, model_path, arch, _ = get_config(model_name, weights_dir)

        # ----------------------------
        # Build model (official)
        # ----------------------------
        self.model = get_network(arch)

        # ----------------------------
        # Load weights (official)
        # ----------------------------
        self.model = load_weights(self.model, model_path)

        # ----------------------------
        # Final setup
        # ----------------------------
        self.model.to(self.device)
        self.model.eval()

    # -------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict:
        """
        Args:
            x: [3, H, W] OR [B, 3, H, W]

        Returns:
            dict:
                {
                    "logits": tensor [B, 1] OR [B, 2]
                }
        """

        # Ensure batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        # Forward pass
        logits = self.model(x)

        return {"logits": logits}