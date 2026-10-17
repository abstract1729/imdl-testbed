import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


# -------------------------------------------------
# 1. Add repo to PYTHONPATH
# -------------------------------------------------

CURRENT_DIR = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_DIR.parents[3]
PGC_REPO_PATH = PROJECT_ROOT / "models" / "pgc-net"

if str(PGC_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(PGC_REPO_PATH))


# -------------------------------------------------
# 2. Import model
# -------------------------------------------------

from models.pgc import PGCNetwork  # noqa: E402


# -------------------------------------------------
# 3. Wrapper
# -------------------------------------------------

class PGCInference:
    """
    Wrapper for PGC model (detection only).

    Responsibilities:
    - Load model
    - Load checkpoint
    - Handle device (CPU / MPS)
    - Return raw logits (NO postprocessing)
    """

    def __init__(self,checkpoint_path: str,dino_variant: str = "dinov2-large",
                 pretrained_root = None,device = None):
        """
        Args:
            checkpoint_path: path to .pth checkpoint
            dino_variant: DINO backbone
            pretrained_root: path to local DINO weights
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
            self.device = torch.device(device)

        # ----------------------------
        # Build model
        # ----------------------------
        self.model = PGCNetwork(dino_variant=dino_variant,num_classes=1,pretrained_root=pretrained_root)

        # ----------------------------
        # Load checkpoint
        # ----------------------------
        self._load_checkpoint(checkpoint_path)

        # ----------------------------
        # Final setup
        # ----------------------------
        self.model.to(self.device)
        self.model.eval()

    # -------------------------------------------------
    def _load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        # CASE 1: Full training checkpoint
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Optional: handle DataParallel prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k.replace("module.", "")
            new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=True)

    # -------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict:
        """
        Args:
            x: [6, H, W] OR [B, 6, H, W]

        Returns:
            dict:
                {
                    "logits": tensor [B, 1]
                }
        """

        # Ensure batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        logits = self.model(x)   # [B, 1]

        return { "logits": logits}