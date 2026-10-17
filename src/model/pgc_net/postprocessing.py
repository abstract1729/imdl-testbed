import torch


def extract_detection_output(raw_output):
    """
    Convert raw PGC-Net output into detection result.

    Args:
        raw_output (dict): Output from PGCInference.predict()

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

    # Shape handling: [B, 1] or [1]
    logits = logits.view(-1)

    # Apply sigmoid → probability
    probs = torch.sigmoid(logits)

    # Take first element (batch size = 1 in your pipeline)
    score = probs[0].item()

    # Threshold (default = 0.5)
    label = 1 if score >= 0.5 else 0

    return {
        "score": score,
        "label": label
    }