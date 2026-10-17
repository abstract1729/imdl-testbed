import torch
import torch.nn.functional as F


def extract_detection_output(output):
    """
    Official TruFor detection post-processing.

    Parameters
    ----------
    output : dict
        Raw outputs returned by TruForInference.predict().

    Returns
    -------
    dict
        {
            "score": float,
        }
    """

    det = output["det"]

    if det is None:
        raise ValueError(
            "TruFor did not return a detection logit."
        )

    return {
        "score": torch.sigmoid(det).item(),
    }


def extract_localization_output(output):
    """
    Official TruFor localization post-processing.

    Parameters
    ----------
    output : dict
        Raw outputs returned by TruForInference.predict().

    Returns
    -------
    dict
        {
            "mask": np.ndarray,
        }
    """

    pred = output["pred"]

    if pred is None:
        raise ValueError(
            "TruFor did not return localization logits."
        )

    pred = torch.squeeze(pred, 0)
    pred = F.softmax(pred, dim=0)[1]

    return {
        "mask": pred.cpu().numpy(),
    }


def extract_confidence_map(output):
    """
    Extract TruFor confidence map.

    Parameters
    ----------
    output : dict

    Returns
    -------
    np.ndarray | None
    """

    conf = output["conf"]

    if conf is None:
        return None

    conf = torch.squeeze(conf, 0)
    conf = torch.sigmoid(conf)[0]

    return conf.cpu().numpy()


def extract_noiseprint(output):
    """
    Extract TruFor Noiseprint++ residual.

    Parameters
    ----------
    output : dict

    Returns
    -------
    np.ndarray | None
    """

    npp = output["npp"]

    if npp is None:
        return None

    npp = torch.squeeze(npp, 0)[0]

    return npp.cpu().numpy()