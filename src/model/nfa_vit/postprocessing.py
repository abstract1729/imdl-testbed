import torch
import numpy as np

def extract_detection_output(output, threshold=0.5):
    """
    Extract and postprocess detection outputs from NFA-ViT model.

    Args:
        output (dict): raw model output
        threshold (float): threshold for binary mask

    Returns:
        dict:
            {
                "pred_mask": np.ndarray (H, W) float32 in [0,1],
                "binary_mask": np.ndarray (H, W) uint8 {0,1},
                "pred_score": float
            }
    """

    # ---- Extract tensors ----
    pred_label = output["pred_label"]    # scalar or (B,)
    pred_label = pred_label.detach().cpu()

    # ---- Detection score ----
    # pred_label is scalar tensor
    if pred_label.ndim == 0:
        pred_score = float(pred_label.item())
    else:
        pred_score = float(pred_label.squeeze().item())

    return {
        "score": pred_score
    }