import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

# Add experiment root
EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paths import (
    MANIPULATED_OUTPUT_DIR,
    GT_MASK_DIR
)


def main():

    gt_files = sorted(GT_MASK_DIR.glob("*.png"))
    npz_files = sorted(MANIPULATED_OUTPUT_DIR.glob("*.npz"))

    print(f"Found {len(npz_files)} outputs")
    print(f"Found {len(gt_files)} GT masks")

    assert len(npz_files) == len(gt_files)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    resized_count = 0

    for npz_path in tqdm(npz_files):

        filename = npz_path.stem

        gt_path = GT_MASK_DIR / filename

        # GT
        gt_mask = cv2.imread(
            str(gt_path),
            cv2.IMREAD_GRAYSCALE
        )

        gt_mask = gt_mask > 0

        gt_h, gt_w = gt_mask.shape

        # Prediction
        data = np.load(npz_path)

        heatmap = data["map"]

        if heatmap.shape != gt_mask.shape:

            resized_count += 1

            heatmap = cv2.resize(
                heatmap,
                (gt_w, gt_h),
                interpolation=cv2.INTER_LINEAR
            )

        pred_mask = heatmap >= 0.5

        # Confusion matrix
        TP += np.logical_and(pred_mask, gt_mask).sum()

        TN += np.logical_and(
            np.logical_not(pred_mask),
            np.logical_not(gt_mask)
        ).sum()

        FP += np.logical_and(
            pred_mask,
            np.logical_not(gt_mask)
        ).sum()

        FN += np.logical_and(
            np.logical_not(pred_mask),
            gt_mask
        ).sum()

    print()

    print(f"Images resized: {resized_count}")

    # ---------------------------------
    # Metrics
    # ---------------------------------

    precision = TP / (TP + FP)

    recall = TP / (TP + FN)

    f1 = (
        2 * precision * recall
        / (precision + recall)
    )

    iou = TP / (TP + FP + FN)

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print()
    print("===== FINAL RESULTS =====")
    print(f"IoU       : {iou:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")


if __name__ == "__main__":
    main()