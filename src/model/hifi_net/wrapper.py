from pathlib import Path
import sys

import torch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HIFI_NET_ROOT = PROJECT_ROOT / "models" / "hifi-net"

if str(HIFI_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(HIFI_NET_ROOT))

from utils.utils import (restore_weight_helper,one_hot_label_new,level_1_convert)
from utils.custom_loss import (IsolatingLossFunction,load_center_radius_api)
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_api import NLCDetection


class HiFiNetInference:
    """
    Reusable HiFi-Net inference wrapper.

    Responsibilities
    ----------------
    - Load HRNet backbone.
    - Load NLCDetection head.
    - Accept preprocessed tensors.
    - Run official forward propagation.
    - Return raw model outputs.

    Input
    -----
    torch.Tensor

        shape = [B,3,H,W]

    The input tensor must already be preprocessed by
    HiFiNetTransform.
    """

    def __init__(self,hrnet_checkpoint,segnet_checkpoint,center_radius_path,device=None):

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if device is not None:
            self.device = torch.device(device)
            if self.device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but unavailable.")

            elif self.device.type == "mps":
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MPS requested but unavailable.")

            elif self.device.type != "cpu":
                raise ValueError(f"Unsupported device {self.device}")

        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")

            elif torch.cuda.is_available():
                self.device = torch.device("cuda")

            else:
                self.device = torch.device("cpu")

        print("=" * 60)
        print(f"Using device : {self.device}")
        print("=" * 60)

        # ----------------------------------------------------
        # Build Models
        # ----------------------------------------------------
        cfg = get_cfg_defaults()
        self.FENet = get_seg_model(cfg).to(self.device)
        self.SegNet = NLCDetection().to(self.device)

        # ----------------------------------------------------
        # Load checkpoints
        # ----------------------------------------------------
        self.FENet = restore_weight_helper(self.FENet,str(hrnet_checkpoint.parent),hrnet_checkpoint.stem)
        self.SegNet = restore_weight_helper(self.SegNet,str(segnet_checkpoint.parent),segnet_checkpoint.stem)

        self.FENet.eval()
        self.SegNet.eval()

        # ----------------------------------------------------
        # Localization helper
        # ----------------------------------------------------

        center, radius = load_center_radius_api(center_radius_path)
        self.loss_map = IsolatingLossFunction(center,radius).to(self.device)

    # --------------------------------------------------------
    # Input Validation
    # --------------------------------------------------------

    @staticmethod
    def _validate_input(rgb):
        if not torch.is_tensor(rgb):
            raise TypeError(f"Expected torch.Tensor, got {type(rgb)}")
        if rgb.ndim != 4:
            raise ValueError("Expected shape [B,3,H,W]")
        if rgb.shape[0] != 1:
            raise ValueError("Official HiFi-Net inference expects batch size 1.")
        if rgb.shape[1] != 3:
            raise ValueError("Expected RGB image.")
        if rgb.dtype != torch.float32:
            raise TypeError("Expected torch.float32 input.")
        
    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    @torch.no_grad()
    def predict(self, rgb):
        self._validate_input(rgb)
        rgb = rgb.to(self.device)

        features = self.FENet(rgb)

        (mask_features,binary_logits,cls0,cls1,cls2,cls3) = self.SegNet(features,rgb)

        return {
            "mask_features": mask_features,
            "binary_logits": binary_logits,

            "cls0": cls0,
            "cls1": cls1,
            "cls2": cls2,
            "cls3": cls3,

            "image_size": tuple(rgb.shape[-2:]),
        }
