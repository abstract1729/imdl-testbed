import numpy as np
import torch
from PIL import Image


# TruFor Image Preprocessing
class TruForTransform:
    """
    TruFor image preprocessing.

    Pipeline:
        PIL RGB Image
            -> NumPy HWC
            -> CHW
            -> torch.float32
            -> scale by 255.0
    """
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = np.array(image)
        image = torch.tensor(image.transpose(2, 0, 1),dtype=torch.float32)
        image = image / 255.0

        return image