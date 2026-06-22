import cv2
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
import numpy as np

# Add experiment root (.../trufor) to Python path
EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paths import (
    PRED_MASK_DIR,
    GT_MASK_DIR,
    PER_IMAGE_METRICS_CSV
)

from metrics.localization_metrics import (compute_iou,compute_accuracy,compute_precision,compute_recall,compute_f1)


def main():

    # ------------------------------------------------
    # Validation
    # ------------------------------------------------

    pred_files = sorted(PRED_MASK_DIR.glob("*.png"))
    gt_files = sorted(GT_MASK_DIR.glob("*.png"))

    print(f"Found {len(pred_files)} predicted masks")
    print(f"Found {len(gt_files)} ground-truth masks")

    assert len(pred_files) == len(gt_files), \
        "Mismatch between number of prediction and GT masks"

    results = []
    num_resized  = 0
    for pred_path in tqdm(pred_files):

        filename = pred_path.name

        gt_path = GT_MASK_DIR / filename

        if not gt_path.exists():
            raise FileNotFoundError(
                f"GT mask missing for {filename}"
            )

        # ------------------------------------------------
        # Load prediction
        # ------------------------------------------------

        pred_mask = cv2.imread(
            str(pred_path),
            cv2.IMREAD_GRAYSCALE
        )

        if pred_mask is None:
            raise ValueError(f"Could not load prediction mask {pred_path}")

        pred_mask = pred_mask > 0

        # ------------------------------------------------
        # Load GT
        # ------------------------------------------------

        gt_mask = cv2.imread(str(gt_path),cv2.IMREAD_GRAYSCALE)

        if gt_mask is None:
            raise ValueError(f"Could not load GT mask {gt_path}")

        gt_mask = gt_mask > 0

        # ------------------------------------------------
        # Shape validation
        # ------------------------------------------------

        # ------------------------------------------------
        # Resize prediction if shape mismatch exists
        # ------------------------------------------------

        if pred_mask.shape != gt_mask.shape:
            num_resized += 1
            gt_h, gt_w = gt_mask.shape

            pred_mask = pred_mask.astype(np.uint8)
            pred_mask = cv2.resize(pred_mask,(gt_w, gt_h),interpolation=cv2.INTER_NEAREST)
            pred_mask = pred_mask > 0

        # ------------------------------------------------
        # Metrics
        # ------------------------------------------------

        results.append(
            {
                "image_name": filename,

                "iou":
                    compute_iou(
                        pred_mask,
                        gt_mask
                    ),

                "accuracy":
                    compute_accuracy(
                        pred_mask,
                        gt_mask
                    ),

                "precision":
                    compute_precision(
                        pred_mask,
                        gt_mask
                    ),

                "recall":
                    compute_recall(
                        pred_mask,
                        gt_mask
                    ),

                "f1":
                    compute_f1(
                        pred_mask,
                        gt_mask
                    )
            }
        )

    df = pd.DataFrame(results)

    df.to_csv(
        PER_IMAGE_METRICS_CSV,
        index=False
    )

    print()
    print(df.head())

    print()
    print(
        f"Saved per-image metrics to:\n"
        f"{PER_IMAGE_METRICS_CSV}"
    )


if __name__ == "__main__":
    main()