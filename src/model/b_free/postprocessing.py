import torch


def extract_detection_output(raw_output):
    """
    Convert raw B-Free output into detection result.

    Applies sigmoid to obtain probability scores for ROC/AUC evaluation.

    Args:
        raw_output (dict): Output from BFreeInference.predict()

    Returns:
        dict:
            {
                "score": float (probability of fake),
                "label": int (0 = real, 1 = fake)
            }
    """

    if "logits" not in raw_output:
        raise KeyError("Expected key 'logits' in raw_output")

    logits = raw_output["logits"]

    # Ensure tensor
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)

    # Flatten to [B, C]
    logits = logits.view(logits.shape[0], -1)

    # Case 1: Single logit [B, 1]
    if logits.shape[1] == 1:
        score_tensor = logits[:, 0]

    # Case 2: Two logits [B, 2]
    elif logits.shape[1] == 2:
        score_tensor = logits[:, 1] - logits[:, 0]

    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    # Apply sigmoid → probability
    prob = torch.sigmoid(score_tensor)

    # Take first element (batch size = 1)
    score = prob[0].item()

    # Threshold at 0.5
    label = 1 if score >= 0.5 else 0

    return {
        "score": score,
        "label": label
    }