"""
CSI-IMD Dataset Parser

Builds a cached metadata index for the CSI-IMD dataset.

Output
------
metadata/
    csi-imd/
        dataset_index.json

Each entry in dataset_index.json has the form

{
    "real_1": {
        "label": 0,
        "image_path": "datasets/csi-imd/pristine_images_used_for_manipulation/1.jpg",
        "mask_path": ""
    },

    "fake_1_1": {
        "label": 1,
        "image_path": "datasets/csi-imd/manipulated_images/1_1.png",
        "mask_path": ""
    }
}
"""

from pathlib import Path
import json
from tqdm import tqdm


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "csi-imd"
REAL_DIR = DATASET_ROOT / "pristine_images_used_for_manipulation"
FAKE_DIR = DATASET_ROOT / "manipulated_images"
OUTPUT_DIR = PROJECT_ROOT / "metadata" / "csi-imd"
OUTPUT_FILE = OUTPUT_DIR / "dataset_index.json"
VALID_EXTENSIONS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def relative_path(path: Path) -> str:
    """
    Convert absolute path into project-relative path.
    """
    return str(path.relative_to(PROJECT_ROOT))


def get_image_files(directory: Path):
    """
    Return all valid image files.
    """
    return sorted( p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)


# ---------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------

def build_dataset_index():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found:\n{DATASET_ROOT}")

    if not REAL_DIR.exists():
        raise FileNotFoundError(f"Real image directory not found:\n{REAL_DIR}")

    if not FAKE_DIR.exists():
        raise FileNotFoundError(f"Fake image directory not found:\n{FAKE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_index = {}
    duplicate_ids = []

    # -------------------------------------------------------------
    # Real Images
    # -------------------------------------------------------------

    real_images = get_image_files(REAL_DIR)
    print(f"Found {len(real_images)} pristine images.")

    for image_path in tqdm(real_images,desc="Indexing pristine images",unit="images"):
        image_id = f"real_{image_path.stem}"
        if image_id in dataset_index:
            duplicate_ids.append(image_id)
            continue
        dataset_index[image_id] = {"label": 0,"file_name": image_path.name,"image_path": relative_path(image_path),"mask_path": ""}

    # -------------------------------------------------------------
    # Fake Images
    # -------------------------------------------------------------

    fake_images = get_image_files(FAKE_DIR)
    print(f"Found {len(fake_images)} manipulated images.")

    for image_path in tqdm(fake_images,desc="Indexing manipulated images",unit="images"):
        image_id = f"fake_{image_path.stem}"
        if image_id in dataset_index:
            duplicate_ids.append(image_id)
            continue
        dataset_index[image_id] = dataset_index[image_id] = {"label": 1,"file_name": image_path.name,"image_path": relative_path(image_path),"mask_path": ""}

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    if duplicate_ids:
        print(f"\nWarning: {len(duplicate_ids)} duplicate IDs found.")
        print("Duplicates:")

        for dup in duplicate_ids:
            print("  ", dup)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset_index, f, indent=4)

    print("\nDataset Index Saved")
    print(f"Location : {OUTPUT_FILE}")
    print(f"Total Samples : {len(dataset_index)}")
    print(f"Real Images : {len(real_images)}")
    print(f"Fake Images : {len(fake_images)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    build_dataset_index()