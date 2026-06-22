import json
import pandas as pd
import sys
from pathlib import Path

# Add experiment root (.../trufor) to Python path
EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXPERIMENT_ROOT))

from paths import (
    PER_IMAGE_METRICS_CSV,
    SUMMARY_METRICS_JSON
)


def main():
    SUMMARY_METRICS_JSON = "./metrics/summary_metrics_exp1.json"
    PER_IMAGE_METRICS_CSV = "./metrics/per_image_metrics_exp1.csv"
    # ------------------------------------------------
    # Load per-image metrics
    # ------------------------------------------------

    df = pd.read_csv(PER_IMAGE_METRICS_CSV)

    # ------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------

    summary = {

        "IoU":
            df["iou"].mean(),

        "Accuracy":
            df["accuracy"].mean(),

        "Precision":
            df["precision"].mean(),

        "Recall":
            df["recall"].mean(),

        "F1":
            df["f1"].mean()

    }

    # ------------------------------------------------
    # Print results
    # ------------------------------------------------

    print()
    print("===== FINAL RESULTS =====")

    for metric_name, value in summary.items():

        print(
            f"{metric_name:<10}: {value:.4f}"
        )

    # ------------------------------------------------
    # Save summary
    # ------------------------------------------------

    with open(
            SUMMARY_METRICS_JSON,
            "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print()
    print(
        f"Summary saved to:\n"
        f"{SUMMARY_METRICS_JSON}"
    )


if __name__ == "__main__":
    main()