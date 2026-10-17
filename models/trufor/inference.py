from pathlib import Path
import sys

import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F



# ---------------------------------------------------------------------
# Add TruFor_train_test to PYTHONPATH
# ---------------------------------------------------------------------

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

    Example
    -------
    model = TruForInference(
        checkpoint_path="TruFor_train_test/pretrained_models/trufor.pth.tar"
    )

    output = model.predict("image.jpg")

    print(output.keys())
    """

    def __init__(
        self,
        checkpoint_path,
        experiment="trufor_ph3",
        device=None,
        save_np=False,
    ):

        self.save_np = save_np

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print("=" * 60)
        print(f"Using device : {self.device}")
        print("=" * 60)

        # -------------------------------------------------------------
        # Configure model exactly as official implementation
        # -------------------------------------------------------------

        args = _Args(experiment, checkpoint_path)

        # ---------------------------------------------------------
        # TruFor expects cwd = TruFor_train_test
        # ---------------------------------------------------------

        old_cwd = Path.cwd()

        try:
            os.chdir(TRUFOR_ROOT)
            update_config(config, args)
        finally:
            os.chdir(old_cwd)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        print(f"Loaded checkpoint (epoch={checkpoint['epoch']})")

        self.model = get_model(config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on : {next(self.model.parameters()).device}")

    # -----------------------------------------------------------------

    def _preprocess(self, image):

        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image), cv2.IMREAD_COLOR)

            if image is None:
                raise FileNotFoundError(image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be image path or RGB numpy array.")

        image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).unsqueeze(0)

        return image.to(self.device)

    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image):

        rgb = self._preprocess(image)

        pred, conf, det, npp = self.model(
            rgb,
            save_np=self.save_np,
        )

        output = {}

        # -------------------------------------------------------------
        # Localization Map
        # -------------------------------------------------------------

        pred = torch.squeeze(pred, 0)
        pred = F.softmax(pred, dim=0)[1]
        pred = pred.cpu().numpy()

        output["map"] = pred

        # -------------------------------------------------------------
        # Original Image Size
        # -------------------------------------------------------------

        output["imgsize"] = tuple(rgb.shape[2:])

        # -------------------------------------------------------------
        # Detection Score
        # -------------------------------------------------------------

        if det is not None:
            output["score"] = torch.sigmoid(det).item()

        # -------------------------------------------------------------
        # Confidence Map
        # -------------------------------------------------------------

        if conf is not None:

            conf = torch.squeeze(conf, 0)
            conf = torch.sigmoid(conf)[0]
            conf = conf.cpu().numpy()

            output["conf"] = conf

        # -------------------------------------------------------------
        # Noiseprint++
        # -------------------------------------------------------------

        if self.save_np and npp is not None:

            npp = torch.squeeze(npp, 0)[0]
            npp = npp.cpu().numpy()

            output["np++"] = npp

        return output
    


if __name__ == "__main__":

    model = TruForInference(
        checkpoint_path="TruFor_train_test/pretrained_models/trufor.pth.tar",
        save_np=True,
    )

    # output = model.predict("./TruFor_train_test/test_images/real/original.png")
    output = model.predict("./TruFor_train_test/test_images/fake/result.png")

    print("=" * 60)
    print("Output Keys")
    print("=" * 60)

    for k, v in output.items():

        if isinstance(v, np.ndarray):
            print(f"{k:10s} -> shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{k:10s} -> {v}")