import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys

# Add experiment root (.../trufor) to Python path
EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paths import (
    MANIPULATED_OUTPUT_DIR,
    GT_MASK_DIR
)

from metrics.localization_metrics import (compute_iou,compute_accuracy,compute_precision,compute_recall,compute_f1)

def main():

    # ------------------------------------------------
    # Validation
    # ------------------------------------------------

    gt_files = sorted(GT_MASK_DIR.glob("*.png"))
    npz_files = sorted(MANIPULATED_OUTPUT_DIR.glob("*.npz"))

    print(f"Found {len(npz_files)} outputs")
    print(f"Found {len(gt_files)} ground-truth masks")

    assert len(npz_files) == len(gt_files), \
        "Mismatch between number of outputs and GT masks"

    results = []

    for npz_path in tqdm(npz_files):

        filename = npz_path.stem

        gt_path = GT_MASK_DIR / filename

        # ----------------------------
        # load GT
        # ----------------------------

        gt_mask = cv2.imread(
            str(gt_path),
            cv2.IMREAD_GRAYSCALE
        )

        gt_mask = gt_mask > 0

        gt_h, gt_w = gt_mask.shape

        # ----------------------------
        # load map
        # ----------------------------

        data = np.load(npz_path)

        heatmap = data["map"]

        # ----------------------------
        # resize probability map
        # ----------------------------

        if heatmap.shape != gt_mask.shape:

            heatmap = cv2.resize(
                heatmap,
                (gt_w, gt_h),
                interpolation=cv2.INTER_LINEAR
            )

        # ----------------------------
        # threshold after resizing
        # ----------------------------

        pred_mask = heatmap >= 0.5

        # ----------------------------
        # metrics
        # ----------------------------

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

    save_path = "./metrics/per_image_metrics_exp1.csv"
    df.to_csv(save_path, index=False)

    print()
    print(df.head())

    print()
    print(
        f"Saved per-image metrics to:\n"
        f"{save_path}"
    )


if __name__ == "__main__":
    main()

    