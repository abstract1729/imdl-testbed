# postprocessing/clipdet_postprocess.py

import torch


def extract_detection_output(outputs: dict, threshold: float = 0.5) -> dict:
    """
    ClipDet detection postprocessing (STRICTLY from main.py)

    Input
    -----
    outputs : dict
        Must contain:
            "logits" : torch.Tensor of shape [B, C]

    Returns
    -------
    dict
        {
            "prediction": int,
            "score": float,
        }
    """

    logits = outputs["logits"]

    # ------------------------------
    # Match main.py logic exactly
    # ------------------------------
    if logits.shape[1] == 1:
        logit = logits[:, 0]
    elif logits.shape[1] == 2:
        logit = logits[:, 1] - logits[:, 0]
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    # ------------------------------
    # Convert to probability
    # ------------------------------
    prob = torch.sigmoid(logit)

    # ------------------------------
    # Scalar extraction
    # ------------------------------
    prob = prob.squeeze().item()

    # ------------------------------
    # Thresholding
    # ------------------------------
    pred = int(prob >= threshold)

    return {
        "prediction": pred,
        "score": prob,
    }