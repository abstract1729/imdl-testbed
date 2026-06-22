from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

from pathlib import Path
import sys

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from paths import (MANIPULATED_OUTPUT_DIR,PRED_MASK_DIR)


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

EXPERIMENT_ROOT = Path("./")

OUTPUT_DIR = EXPERIMENT_ROOT / "outputs" / "manipulated"
SAVE_DIR = EXPERIMENT_ROOT / "processed" / "masks"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Threshold used in CSI-IMD paper
# -------------------------------------------------------------------

THRESHOLD = 0.5


# -------------------------------------------------------------------
# Process all npz files
# -------------------------------------------------------------------

npz_files = sorted(OUTPUT_DIR.glob("*.npz"))

print(f"Found {len(npz_files)} files")

for npz_path in tqdm(npz_files):

    # file name without extension
    image_id = npz_path.stem

    # load npz
    data = np.load(npz_path)

    # localization heatmap
    heatmap = data["map"]

    # convert to binary mask
    binary_mask = (heatmap >= THRESHOLD).astype(np.uint8)

    # convert {0,1} → {0,255}
    binary_mask *= 255

    # output filename
    save_path = SAVE_DIR / image_id

    # save png
    cv2.imwrite(str(save_path), binary_mask)


print("Done.")