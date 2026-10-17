import numpy as np
import torch
from PIL import Image


class PSCCNetTransform:
    """
    PSCC-Net image preprocessing.

    Pipeline:
        PIL Image
            -> RGBA to RGB (if required)
            -> NumPy HWC
            -> torch.float32
            -> Scale to [0, 1]
            -> CHW
    """

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # Convert RGBA to RGB (same behaviour as official TestData)
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        elif image.mode != "RGB":
            image = image.convert("RGB")

        image = np.array(image, dtype=np.float32)
        image /= 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image