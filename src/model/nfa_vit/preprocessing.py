import cv2
import numpy as np
import torch


class NFAViTTransform:
    def __init__(self, img_size=512):
        self.img_size = img_size

        # ImageNet normalization (confirmed from iml_transforms.py)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, image):
        """
        Args:
            image: input image (H, W, 3) in BGR or RGB

        Returns:
            tensor: (1, 3, H, W)
        """

        # --- Step 1: Ensure RGB ---
        if isinstance(image, str):
            image = cv2.imread(image)
        
        # Convert BGR → RGB if needed
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Step 2: Resize ---
        image = cv2.resize(image, (self.img_size, self.img_size))

        # --- Step 3: Normalize ---
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        # --- Step 4: HWC → CHW ---
        image = image.transpose(2, 0, 1)

        # --- Step 5: To Tensor ---
        image = torch.from_numpy(image).float()

        # --- Step 6: Add batch dimension ---
        image = image.unsqueeze(0)

        return image