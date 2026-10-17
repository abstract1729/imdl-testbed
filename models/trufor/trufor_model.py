from pathlib import Path
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F



# Add TruFor_train_test to PYTHONPATH
THIS_DIR = Path(__file__).resolve().parent
TRUFOR_ROOT = THIS_DIR / "TruFor_train_test"

if str(TRUFOR_ROOT) not in sys.path:
    sys.path.insert(0, str(TRUFOR_ROOT))


from lib.config import config, update_config
from lib.utils import get_model


class _Args:
    """
    Dummy argparse namespace used by update_config().
    """

    def __init__(self, experiment, checkpoint):

        self.experiment = experiment

        self.opts = [
            "TEST.MODEL_FILE",
            str(checkpoint),
        ]


class TruForInference:
    """
    Reusable TruFor inference wrapper.

    Responsibilities
        
    - Load and configure the TruFor model.
    - Accept preprocessed image tensors.
    - Run model inference.
    - Apply official TruFor output post-processing.

    Input
    -----
    torch.Tensor with shape:

        [B, 3, H, W]

    For official TruFor inference:

        B = 1

    Image loading and preprocessing must be handled outside
    this class by the Dataset / transform pipeline.
    """

    def __init__(self,checkpoint_path,experiment="trufor_ph3",device=None,save_np=False):
        self.save_np = save_np

        # Device
        if device is not None:
            self.device = torch.device(device)
            if self.device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA device requested, but CUDA is not available.")

            elif self.device.type == "mps":
                if not torch.backends.mps.is_available():
                    raise RuntimeError("MPS device requested, but MPS is not available.")

            elif self.device.type != "cpu":
                raise ValueError(
                    f"Unsupported device type: {self.device.type}. "
                    "Expected one of: cpu, cuda, mps.")

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

        # Configure TruFor
        args = _Args(
            experiment=experiment,
            checkpoint=checkpoint_path)
        
        # TruFor configuration expects TruFor repository cwd
        old_cwd = Path.cwd()
        try:
            os.chdir(TRUFOR_ROOT)

            update_config(
                config,
                args)
        finally:
            os.chdir(old_cwd)
         
        # Load Checkpoint
        checkpoint = torch.load(checkpoint_path,map_location=self.device,weights_only=False)

        print(
            f"Loaded checkpoint "
            f"(epoch={checkpoint['epoch']})")

        # Build Model
        self.model = get_model(config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(
            f"Model loaded on : "
            f"{next(self.model.parameters()).device}")

    # Validate Input Tensor
    @staticmethod
    def _validate_input(rgb):

        if not torch.is_tensor(rgb):
            raise TypeError(
                "TruForInference expects a torch.Tensor. "
                f"Received {type(rgb).__name__}.")

        if rgb.ndim != 4:
            raise ValueError(
                "Expected input tensor with shape "
                "[B, 3, H, W]. "
                f"Received shape {tuple(rgb.shape)}.")

        if rgb.shape[0] != 1:
            raise ValueError(
                "Official TruFor inference uses batch_size=1. "
                f"Received batch size {rgb.shape[0]}.")

        if rgb.shape[1] != 3:
            raise ValueError(
                "Expected 3-channel RGB input. "
                f"Received {rgb.shape[1]} channels.")

        if rgb.dtype != torch.float32:
            raise TypeError(
                "Expected torch.float32 input. "
                f"Received {rgb.dtype}.")


    # Inference
    @torch.no_grad()
    def predict(self, rgb):
        # Validate Preprocessed Tensor
        self._validate_input(rgb)

        # Move Input to Model Device
        rgb = rgb.to(self.device)

        # Model Inference
        pred, conf, det, npp = self.model(rgb,save_np=self.save_np)

        # Output Dictionary
        output = {}
         
        # Localization Map
        pred = torch.squeeze(pred,0)
        pred = F.softmax(pred,dim=0)[1]
        pred = pred.cpu().numpy()
        output["map"] = pred

        # Original Input Tensor Size
        output["imgsize"] = tuple(rgb.shape[2:])

        # Detection Score
        if det is not None:
            output["score"] = (torch.sigmoid(det).item())
     
        # Confidence Map
        if conf is not None:
            conf = torch.squeeze(conf,0,)
            conf = torch.sigmoid(conf)[0]
            conf = conf.cpu().numpy()
            output["conf"] = conf

        # Noiseprint++
        if self.save_np and npp is not None:
            npp = torch.squeeze(npp,0)[0]
            npp = npp.cpu().numpy()
            output["np++"] = npp

        return output