import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# Add experiment root to sys.path
# --------------------------------------------------

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from paths import (REAL_OUTPUT_DIR,MANIPULATED_OUTPUT_DIR)

from metrics.metrics_official import computeDetectionMetrics


def main():

    scores = []
    labels = []

    rows = []

    # ---------------------------------------------
    # Real images
    # ---------------------------------------------

    real_files = sorted(REAL_OUTPUT_DIR.glob("*.npz"))

    print(f"Found {len(real_files)} real outputs")

    for npz_path in real_files:

        score = float(np.load(npz_path)["score"])

        scores.append(score)
        labels.append(0)

        rows.append(
            {
                "image_name": npz_path.stem,
                "label": 0,
                "score": score
            }
        )

    # ---------------------------------------------
    # Manipulated images
    # ---------------------------------------------

    manipulated_files = sorted(MANIPULATED_OUTPUT_DIR.glob("*.npz"))
    print(f"Found {len(manipulated_files)} manipulated outputs")

    for npz_path in manipulated_files:

        score = float(np.load(npz_path)["score"])
        scores.append(score)
        labels.append(1)

        rows.append(
            {
                "image_name": npz_path.stem,
                "label": 1,
                "score": score
            }
        )

    # ---------------------------------------------
    # Official metrics
    # ---------------------------------------------

    auc, bacc = computeDetectionMetrics(scores,labels)

    print()
    print("===== FINAL RESULTS =====")
    print(f"AUC                : {auc:.4f}")
    print(f"Balanced Accuracy  : {bacc:.4f}")

    # ---------------------------------------------
    # Save per-image scores
    # ---------------------------------------------

    df = pd.DataFrame(rows)
    csv_path = (Path(__file__).parent /"detection_scores.csv")
    df.to_csv(csv_path,index=False)

    summary = {
        "auc": float(auc),
        "balanced_accuracy": float(bacc)
    }

    import json

    json_path = (
        Path(__file__).parent /
        "detection_summary.json"
    )

    with open(json_path, "w") as f:
        json.dump(summary,f,indent=4)

    print()
    print(f"Scores saved to:\n{csv_path}")
    print(f"Summary saved to:\n{json_path}")


if __name__ == "__main__":
    main()