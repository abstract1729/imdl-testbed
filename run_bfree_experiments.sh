#!/bin/bash

# -----------------------------
# List of datasets
# -----------------------------
DATASETS=(
    "auto-splice"
    "br-gen"
    "deal-300k"
    "csi-imd"
    "img-edit"
    "sid-set"
)

# -----------------------------
# Base script path
# -----------------------------
BASE_PATH="experiments-pipeline/detection/"

# -----------------------------
# Run experiments sequentially
# -----------------------------
for DATASET in "${DATASETS[@]}"; do
    echo "======================================"
    echo "Running B-Free on dataset: $DATASET"
    echo "======================================"

    SCRIPT_PATH="$BASE_PATH/$DATASET/b-free/experiment1.py"

    # Run script
    python "$SCRIPT_PATH"

    # Check exit status
    if [ $? -ne 0 ]; then
        echo "❌ Failed on dataset: $DATASET"
        echo "➡️ Continuing to next dataset..."
        continue
    fi

    echo "✅ Completed dataset: $DATASET"
    echo ""
done

echo "🎯 All experiments attempted."