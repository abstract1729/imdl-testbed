import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------
# Add experiment root to sys.path
# ---------------------------------------------------------

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from paths import (
    MANIPULATED_OUTPUT_DIR,
    GT_MASK_DIR
)

from metrics.metrics_official import (
    computeLocalizationMetrics
)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    npz_files = sorted(MANIPULATED_OUTPUT_DIR.glob("*.npz"))
    gt_files = sorted(GT_MASK_DIR.glob("*.png"))
    print(f"Found {len(npz_files)} outputs")
    print(f"Found {len(gt_files)} GT masks")

    assert len(npz_files) == len(gt_files)

    results = []

    skipped = 0
    skipped_mismatch = 0

    for npz_path in tqdm(npz_files):

        filename = npz_path.stem

        gt_path = GT_MASK_DIR / filename

        if not gt_path.exists():

            print(f"Missing GT for {filename}")
            skipped += 1
            continue

        # -------------------------------------------------
        # Load prediction map
        # -------------------------------------------------

        data = np.load(npz_path)

        pred_map = data["map"]

        # -------------------------------------------------
        # Load GT mask
        # -------------------------------------------------

        gt = cv2.imread(str(gt_path),cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"Could not load {gt_path}")
            skipped += 1
            continue

        gt = gt > 0

        # -------------------------------------------------
        # Resize prediction if necessary
        # -------------------------------------------------

        if pred_map.shape != gt.shape:
            skipped_mismatch+=1
            skipped += 1
            continue

            # gt_h, gt_w = gt.shape

            # pred_map = cv2.resize(
            #     pred_map,
            #     (gt_w, gt_h),
            #     interpolation=cv2.INTER_LINEAR
            # )

        # -------------------------------------------------
        # Official TruFor metrics
        # -------------------------------------------------

        F1_best, F1_th = computeLocalizationMetrics(
            pred_map,
            gt
        )

        results.append(
            {
                "image_name": filename,
                "f1_best": F1_best,
                "f1_th": F1_th
            }
        )

    df = pd.DataFrame(results)

    output_csv = (
        Path(__file__).parent /
        "per_image_metrics_official.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print()

    print("========== SUMMARY ==========")

    print(f"Images resized : {skipped_mismatch}")
    print(f"Images skipped : {skipped}")
    print(f"Mean F1_best : {np.nanmean(df['f1_best']):.4f}")
    print(f"Mean F1_th   : {np.nanmean(df['f1_th']):.4f}")
    print()
    print(f"Per-image metrics saved to:\n{output_csv}")

if __name__ == "__main__":
    main()