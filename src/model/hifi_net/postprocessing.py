import numpy as np
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HIFI_NET_ROOT = PROJECT_ROOT / "models" / "hifi-net"

if str(HIFI_NET_ROOT) not in sys.path:
    sys.path.insert(0, str(HIFI_NET_ROOT))

from utils.utils import (one_hot_label_new,level_1_convert)

# ============================================================
# Detection Post-processing
# ============================================================

def extract_detection_output(outputs):
    """
    Official HiFi-Net detection post-processing.

    Parameters
    ----------
    outputs : dict
        Output dictionary returned by HiFiNetInference.predict()

    Returns
    -------
    prediction : int

        0 -> Real
        1 -> Forged

    probability : float

        Probability of being forged.
    """

    prediction, probability = one_hot_label_new(outputs["cls3"])
    prediction = level_1_convert(prediction)[0]
    probability = float(probability[0])

    return {"prediction": prediction,"score": probability}


# ============================================================
# Localization Post-processing
# ============================================================

def extract_localization_output(outputs,model,threshold=2.3):
    """
    Official HiFi-Net localization
    post-processing.
    """

    _, score = model.loss_map.inference(outputs["mask_features"])
    score = score.cpu().numpy()
    score = np.where( score >= threshold, 1.0, 0.0)

    return {
        "mask": score[0],
        "threshold": threshold
    }