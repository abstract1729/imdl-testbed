import torch


# ============================================================
# Detection Post-processing
# ============================================================

def extract_detection_output(outputs: dict,threshold: float = 0.5) -> dict:
    """
    Official Universal Fake Detector detection post-processing.

    Parameters
    ----------
    outputs : dict

        Output dictionary returned by
        UnivFDInference.predict().

    threshold : float, default=0.5

        Decision threshold used to convert the probability
        into a binary prediction.

    Returns
    -------
    dict

        {
            "prediction": int,
            "score": float
        }

        prediction

            0 -> Real
            1 -> Fake

        score

            Probability of being fake.
    """

    logits = outputs["logits"]
    probability = torch.sigmoid(logits)
    probability = probability.squeeze().item()
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "score": probability,
    }