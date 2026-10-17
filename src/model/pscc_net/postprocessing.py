import torch
import torch.nn.functional as F


def extract_detection_output(output):
    """
    Official PSCC-Net detection post-processing.

    Parameters
    ----------
    output : dict
        Raw outputs returned by PSCCNetInference.predict().

    Returns
    -------
    dict
        {
            "score": float,
            "probabilities": torch.Tensor,
        }
    """

    logits = output["logits"]

    probabilities = torch.softmax(logits, dim=1)

    return {
        "score": probabilities[0, 1].item(),
        "probabilities": probabilities,
    }


def extract_localization_output(output):
    """
    Official PSCC-Net localization post-processing.

    Parameters
    ----------
    output : dict
        Raw outputs returned by PSCCNetInference.predict().

    Returns
    -------
    dict
        {
            "mask": np.ndarray,
        }
    """

    mask = output["mask1"]

    h, w = output["imgsize"]

    mask = F.interpolate(
        mask,
        size=(h, w),
        mode="bilinear",
        align_corners=True,
    )

    mask = torch.sigmoid(mask)

    return {
        "mask": (
            mask.squeeze()
                .cpu()
                .numpy()
        )
    }