from pathlib import Path
import sys

import torch


# ============================================================
# Repository Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UNIVFD_ROOT = PROJECT_ROOT / "models" / "univfd"

if str(UNIVFD_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIVFD_ROOT))

from models import get_model


class UnivFDInference:
    """
    Reusable Universal Fake Detector inference wrapper.

    Responsibilities
    ----------------
    - Build official UnivFD model.
    - Load pretrained FC weights.
    - Accept preprocessed tensors.
    - Run official forward propagation.
    - Return raw model outputs.

    Input
    -----
    torch.Tensor

        shape = [B, 3, 224, 224]

    The input tensor must already be preprocessed by
    UnivFDTransform.
    """

    def __init__(self,checkpoint_path: Path,arch: str = "CLIP:ViT-L/14",device: str | None = None):

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
                raise ValueError(f"Unsupported device: {self.device}")

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
        # Validate checkpoint
        # ----------------------------------------------------

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.arch = arch

        # ----------------------------------------------------
        # Build Model
        # ----------------------------------------------------
        self.model = get_model(self.arch)

        # ----------------------------------------------------
        # Load FC checkpoint
        # ----------------------------------------------------

        state_dict = torch.load(str(checkpoint_path),map_location="cpu")
        self.model.fc.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

    # --------------------------------------------------------
    # Input Validation
    # --------------------------------------------------------

    @staticmethod
    def _validate_input(image: torch.Tensor):
        if not torch.is_tensor(image):
            raise TypeError(f"Expected torch.Tensor, got {type(image)}")

        if image.ndim != 4:
            raise ValueError("Expected input shape [B, 3, H, W].")

        if image.shape[1] != 3:
            raise ValueError("Expected RGB image.")

        if image.dtype != torch.float32:
            raise TypeError("Expected torch.float32 input.")

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    @torch.no_grad()
    def predict(self,image: torch.Tensor) -> dict:
        self._validate_input(image)
        image = image.to(self.device)
        logits = self.model(image)

        return {
            "logits": logits,
            "image_size": tuple(image.shape[-2:]),
        }