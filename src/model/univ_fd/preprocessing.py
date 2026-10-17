import torch
import torchvision.transforms as transforms
from PIL import Image


class UnivFDTransform:
    """
    Universal Fake Detector preprocessing.

    Official preprocessing pipeline
    --------------------------------
        PIL RGB Image
            -> CenterCrop(224)
            -> ToTensor()
            -> Normalize(CLIP mean/std)

    Notes
    -----
    - This follows the preprocessing used by RealFakeDataset in the
      official repository.
    - The CLIP preprocessing returned by clip.load() is NOT used by
      the official implementation.
    - Gaussian blur and JPEG compression used in the paper are only
      for robustness evaluation and are intentionally omitted here.
    """

    CLIP_MEAN = (0.48145466,0.4578275,0.40821073)
    CLIP_STD = (0.26862954,0.26130258,0.27577711)

    def __init__(self):
        self.transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=self.CLIP_MEAN,std=self.CLIP_STD),])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected PIL.Image.Image, got {type(image)}"
            )

        image = image.convert("RGB")

        return self.transform(image)