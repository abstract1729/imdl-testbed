"""
Img-Edit Dataset Parser

Creates

metadata/
    img-edit/
        dataset_index.json

Dataset Structure
-----------------

datasets/
    img-edit/
        single-turn/
            results_remove/
                <triplet_id>/
                    original.png
                    result.png
                    mask.png
"""

from pathlib import Path
import json
from tqdm import tqdm


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = (PROJECT_ROOT/ "datasets"/ "img-edit"/ "single-turn"/ "results_remove")
OUTPUT_DIR = PROJECT_ROOT / "metadata" / "img-edit"
OUTPUT_FILE = OUTPUT_DIR / "dataset_index.json"


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def relative_path(path: Path):

    return str(path.relative_to(PROJECT_ROOT))


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------

def build_dataset_index():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(DATASET_ROOT)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_index = {}
    duplicate_ids = []
    triplet_dirs = sorted( d for d in DATASET_ROOT.iterdir() if d.is_dir())

    print(f"Found {len(triplet_dirs)} triplets.")

    missing_files = []

    for triplet_dir in tqdm( triplet_dirs, desc="Indexing triplets", unit="triplets"):
        directory_name = triplet_dir.name
        original_path = triplet_dir / "original.png"
        result_path = triplet_dir / "result.png"
        mask_path = triplet_dir / "mask.png"

        if ( not original_path.exists() or not result_path.exists() or not mask_path.exists() ):
            missing_files.append(directory_name)
            continue

        # ----------------------------------------------------------
        # Real Image
        # ----------------------------------------------------------

        real_id = f"real_{directory_name}"

        if real_id in dataset_index:
            duplicate_ids.append(real_id)
        else:

            dataset_index[real_id] = {
                "label": 0,
                "file_name": original_path.name,
                "image_path": relative_path(original_path),
                "mask_path": "",
            }

        # ----------------------------------------------------------
        # Fake Image
        # ----------------------------------------------------------

        fake_id = f"fake_{directory_name}"

        if fake_id in dataset_index:
            duplicate_ids.append(fake_id)

        else:
            dataset_index[fake_id] = {
                "label": 1,
                "file_name": result_path.name,
                "image_path": relative_path(result_path),
                "mask_path": "",
            }

    # ------------------------------------------------------------------

    with open(OUTPUT_FILE, "w") as f:
        json.dump( dataset_index, f, indent=4)

    # ------------------------------------------------------------------

    print("\nDataset Index Saved")
    print(f"Location      : {OUTPUT_FILE}")
    print(f"Total Samples : {len(dataset_index)}")
    print(f"Triplets      : {len(triplet_dirs)}")
    print(f"Real Images   : {len(dataset_index)//2}")
    print(f"Fake Images   : {len(dataset_index)//2}")

    if duplicate_ids:
        print(f"\nWarning: {len(duplicate_ids)} duplicate IDs found.")

    if missing_files:
        print(f"\nWarning: {len(missing_files)} incomplete triplets skipped.")
        print("First 10:")
        for item in missing_files[:10]:
            print(" ", item)


# ------------------------------------------------------------------

if __name__ == "__main__":
    build_dataset_index()