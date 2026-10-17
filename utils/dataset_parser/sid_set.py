"""
SID-Set Dataset Parser

Creates

metadata/
    sid-set/
        dataset_index.json

Dataset Structure
-----------------

datasets/
    sid-set/
        real/
        edited/
        masks/
"""

from pathlib import Path
import json
from tqdm import tqdm


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "sid-set"
REAL_DIR = DATASET_ROOT / "real"
FAKE_DIR = DATASET_ROOT / "edited"
MASK_DIR = DATASET_ROOT / "masks"
OUTPUT_DIR = PROJECT_ROOT / "metadata" / "sid-set"
OUTPUT_FILE = OUTPUT_DIR / "dataset_index.json"
VALID_EXTENSIONS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def relative_path(path: Path):
    return str(path.relative_to(PROJECT_ROOT))


def get_image_files(directory: Path):
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------

def build_dataset_index():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(DATASET_ROOT)

    if not REAL_DIR.exists():
        raise FileNotFoundError(REAL_DIR)

    if not FAKE_DIR.exists():
        raise FileNotFoundError(FAKE_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_index = {}
    duplicate_ids = []

    # -------------------------------------------------------------
    # Real Images
    # -------------------------------------------------------------

    real_images = get_image_files(REAL_DIR)
    print(f"Found {len(real_images)} real images.")

    for image_path in tqdm(real_images,desc="Indexing real images",unit="images"):
        image_id = f"real_{image_path.stem}"
        if image_id in dataset_index:
            duplicate_ids.append(image_id)
            continue

        dataset_index[image_id] = {
            "label": 0,
            "file_name": image_path.name,
            "image_path": relative_path(image_path),
            "mask_path": "",
        }

    # -------------------------------------------------------------
    # Edited Images
    # -------------------------------------------------------------

    fake_images = get_image_files(FAKE_DIR)

    print(f"Found {len(fake_images)} edited images.")

    for image_path in tqdm(fake_images,desc="Indexing edited images",unit="images"):
        image_id = f"fake_{image_path.stem}"
        if image_id in dataset_index:
            duplicate_ids.append(image_id)
            continue

        # Optional: store mask path now for future localization
        mask_path = MASK_DIR / image_path.name
        dataset_index[image_id] = {
            "label": 1,
            "file_name": image_path.name,
            "image_path": relative_path(image_path),
            "mask_path": (
                relative_path(mask_path)
                if mask_path.exists()
                else ""
            ),
        }

    # -------------------------------------------------------------

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset_index,f,indent=4)

    # -------------------------------------------------------------

    print("\nDataset Index Saved")
    print(f"Location      : {OUTPUT_FILE}")
    print(f"Total Samples : {len(dataset_index)}")
    print(f"Real Images   : {len(real_images)}")
    print(f"Fake Images   : {len(fake_images)}")

    if duplicate_ids:
        print(f"\nWarning: {len(duplicate_ids)} duplicate IDs found.")


# ------------------------------------------------------------------

if __name__ == "__main__":
    build_dataset_index()