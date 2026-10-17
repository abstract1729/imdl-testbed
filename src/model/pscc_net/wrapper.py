from pathlib import Path
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Add PSCC-Net repository to PYTHONPATH
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PSCC_ROOT = PROJECT_ROOT / "models" / "pscc-net"

if str(PSCC_ROOT) not in sys.path:
    sys.path.insert(0, str(PSCC_ROOT))

# print(PSCC_ROOT)

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.config import get_pscc_args


class PSCCNetInference:
    """
    Reusable PSCC-Net inference wrapper.

    Responsibilities
    ----------------
    - Build the complete PSCC-Net architecture.
    - Load pretrained checkpoints.
    - Handle device placement.
    - Run inference on preprocessed tensors.

    Image loading and preprocessing are intentionally handled
    outside this class by the dataset pipeline.
    """

    def __init__(self,checkpoint_root,device=None):
        self.checkpoint_root = Path(checkpoint_root)

        # ----------------------------------------------------------
        # Device selection
        # ----------------------------------------------------------

        if device is not None:
            self.device = torch.device(device)
            if self.device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but unavailable.")

            elif self.device.type == "mps":
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MPS requested but unavailable.")

            elif self.device.type != "cpu":
                raise ValueError(f"Unsupported device: {self.device.type}")

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

        # ----------------------------------------------------------
        # Configuration
        # ----------------------------------------------------------

        self.args = get_pscc_args()

        # ----------------------------------------------------------
        # Build HRNet Backbone
        # ----------------------------------------------------------

        hrnet_cfg = get_hrnet_cfg()

        self.FENet = get_seg_model(hrnet_cfg)

        # ----------------------------------------------------------
        # Build Localization Head
        # ----------------------------------------------------------

        self.SegNet = NLCDetection(self.args)

        # ----------------------------------------------------------
        # Build Detection Head
        # ----------------------------------------------------------

        self.ClsNet = DetectionHead(self.args)

        # ----------------------------------------------------------
        # Device placement
        # ----------------------------------------------------------

        self.FENet = self.FENet.to(self.device)
        self.SegNet = self.SegNet.to(self.device)
        self.ClsNet = self.ClsNet.to(self.device)

        if torch.cuda.is_available():

            self.FENet = nn.DataParallel(self.FENet)
            self.SegNet = nn.DataParallel(self.SegNet)
            self.ClsNet = nn.DataParallel(self.ClsNet)

        # ----------------------------------------------------------
        # Load checkpoints
        # ----------------------------------------------------------

        self._load_network_weight(self.FENet,self.checkpoint_root / "HRNet_checkpoint","HRNet.pth")
        self._load_network_weight(self.SegNet,self.checkpoint_root / "NLCDetection_checkpoint","NLCDetection.pth")
        self._load_network_weight(self.ClsNet,self.checkpoint_root / "DetectionHead_checkpoint","DetectionHead.pth")
        self.FENet.eval()
        self.SegNet.eval()
        self.ClsNet.eval()

        print(
            f"Model loaded on : "
            f"{next(self.FENet.parameters()).device}"
        )

    # --------------------------------------------------------------
    # Checkpoint Loader
    # --------------------------------------------------------------

    def _load_network_weight(self,net,checkpoint_dir,checkpoint_name):

        checkpoint_path = Path(checkpoint_dir) / checkpoint_name

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            else:
                state_dict = checkpoint

        else:

            state_dict = checkpoint

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):

                key = key[7:]

            cleaned_state_dict[key] = value

        net.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

        print(f"Loaded {checkpoint_name}")

    # --------------------------------------------------------------
    # Input Validation
    # --------------------------------------------------------------

    @staticmethod
    def _validate_input(image):
        if not torch.is_tensor(image):
            raise TypeError("PSCCNetInference expects a torch.Tensor.")

        if image.ndim != 4:

            raise ValueError(
                "Expected input tensor of shape "
                "[B, 3, H, W]."
            )

        if image.shape[1] != 3:

            raise ValueError(
                "Expected RGB input."
            )

        if image.dtype != torch.float32:

            raise TypeError("Expected torch.float32 input.")


    # --------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image):

        self._validate_input(image)
        image = image.to(self.device)
        feat = self.FENet(image)
        mask1, mask2, mask3, mask4 = self.SegNet(feat)
        logits = self.ClsNet(feat)

        return {

            "features": feat,

            "mask1": mask1,
            "mask2": mask2,
            "mask3": mask3,
            "mask4": mask4,

            "logits": logits,

            "imgsize": tuple(image.shape[2:]),
        }
    