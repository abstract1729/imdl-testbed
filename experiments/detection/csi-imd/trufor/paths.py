from pathlib import Path


# -----------------------------------------------------
# Directory hierarchy
# -----------------------------------------------------

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent.parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "csi-imd"


# -----------------------------------------------------
# Input folders
# -----------------------------------------------------

REAL_OUTPUT_DIR = (EXPERIMENT_ROOT / "outputs" / "real")
MANIPULATED_OUTPUT_DIR = (EXPERIMENT_ROOT / "outputs" / "manipulated")
GT_MASK_DIR = (DATASET_ROOT / "semantic_relavent_pixel_truth_mask")

# -----------------------------------------------------
# Processed outputs
# -----------------------------------------------------

PROCESSED_DIR = (EXPERIMENT_ROOT / "processed")
PRED_MASK_DIR = (PROCESSED_DIR / "masks")
VISUALIZATION_DIR = (PROCESSED_DIR / "visualization")

# -----------------------------------------------------
# Metric files
# -----------------------------------------------------

METRIC_DIR = (EXPERIMENT_ROOT / "metrics")
PER_IMAGE_METRICS_CSV = (METRIC_DIR / "per_image_metrics.csv")
SUMMARY_METRICS_JSON = (METRIC_DIR / "summary_metrics.json")