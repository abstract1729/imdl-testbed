import numpy as np
import torch
from PIL import Image


class HiFiNetTransform:
    """
    HiFi-Net image preprocessing.

    Pipeline

        PIL RGB Image
            -> Resize (256x256, Bicubic)
            -> NumPy HWC
            -> CHW
            -> torch.float32
            -> divide by 255.0
    """

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((256, 256),resample=Image.BICUBIC)
        image = np.asarray(image)
        image = torch.tensor(image.transpose(2, 0, 1),dtype=torch.float32)
        image = image / 255.0

        return image