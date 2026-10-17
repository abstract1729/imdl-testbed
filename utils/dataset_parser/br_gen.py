"""
dataset_parser.py

BR-Gen Dataset Parser

This script reconstructs the official train/val/test
real-image splits and later will generate the
master index and expanded indices.
"""

from pathlib import Path
import json
from tqdm import tqdm


# ============================================================
# Globals
# ============================================================

GENERATORS = ["BrushNet","LaMa","MAT","PowerPaint","SDXL"]
EDIT_TYPES = ["Background","Stuff"]

# ============================================================
# Utility Functions
# ============================================================

def load_json(json_path):
    """Load a JSON file."""

    with open(json_path, "r") as f:
        return json.load(f)


def save_json(data, json_path):
    """Save JSON with pretty formatting."""

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved -> {json_path}")


def get_image_stem(image_path: str) -> str:
    """
    Examples
    --------
    000000123456.jpg -> 000000123456

    n02084071_1227.jpeg -> n02084071_1227

    c-conference_center-00000002.jpg
    -> c-conference_center-00000002
    """
    return Path(image_path).stem


def get_real_key(dataset, image_id):
    return f"real_{dataset.lower()}_{image_id}"


def get_fake_key(generator, edit_type, dataset, image_id):
    return (
        f"fake_"
        f"{generator.lower()}_"
        f"{edit_type.lower()}_"
        f"{dataset.lower()}_"
        f"{image_id}"
    )

def normalize_brgen_path(path: str) -> str:
    """
    Convert:
    ./BR-Gen/...  → datasets/br-gen/...
    """
    if path.startswith("./BR-Gen/"):
        return "datasets/br-gen/" + path[len("./BR-Gen/"):]
    return path

# ============================================================
# Creating Dataset basic split
# ============================================================

def extract_train_real(train_json):

    train_real = []

    for item in train_json:

        if len(item) != 2:
            continue

        image_path, second = item

        if second == "Negative":
            image_path = normalize_brgen_path(image_path)
            train_real.append([image_path, "Negative"])


    print(f"Train real images : {len(train_real)}")
    return train_real


def extract_val_real(val_json):

    val_real = []

    for item in val_json:

        if len(item) != 2:
            continue

        image_path, second = item

        image_path = normalize_brgen_path(image_path)
        val_real.append([image_path, second])

    print(f"Validation real images : {len(val_real)}")
    return val_real


def create_test_real(train_real, val_real, dataset_root):

    dataset_root = Path(dataset_root)

    # --------------------------------------------------
    # Build used set (already normalized)
    # --------------------------------------------------
    used = set()

    for path, _ in train_real:
        used.add(path)

    for path, _ in val_real:
        used.add(path)

    # --------------------------------------------------
    # Traverse dataset
    # --------------------------------------------------
    test_real = []
    real_root = dataset_root / "RealImage_Img"

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    for dataset_dir in sorted(real_root.iterdir()):

        if not dataset_dir.is_dir():
            continue

        for image_path in sorted(dataset_dir.iterdir()):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # --------------------------------------------------
            # Construct standardized path
            # --------------------------------------------------
            relative = (
                "datasets/br-gen/"
                + image_path.relative_to(dataset_root).as_posix()
            )

            if relative not in used:
                test_real.append([relative, "Negative"])

    print(f"Test real images : {len(test_real)}")
    return test_real


def validate_real_splits(train_real,val_real,test_real):
    """
    Validate that the reconstructed
    split is correct.
    """

    train_set = {x[0] for x in train_real}
    val_set = {x[0] for x in val_real}
    test_set = {x[0] for x in test_real}

    print("\n==============================")
    print("Split Statistics")
    print("==============================")

    print(f"Train : {len(train_set)}")
    print(f"Val   : {len(val_set)}")
    print(f"Test  : {len(test_set)}")

    print()
    print("Train ∩ Val  :",len(train_set & val_set))
    print("Train ∩ Test :",len(train_set & test_set))
    print("Val ∩ Test   :",len(val_set & test_set))
    total = train_set | val_set | test_set
    print()
    print(f"Total unique : {len(total)}")

    assert len(train_set) == 12000
    assert len(val_set) == 1500
    assert len(test_set) == 1500

    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
    assert len(total) == 15000

    print("\nValidation successful.")


# ============================================================
# Master Index Creation and Validation
# ============================================================

def create_master_index(train_real_json,val_real_json,test_real_json,dataset_root):
    """
    Creates the master index.

    One entry corresponds to one pristine image.

    Parameters
    ----------
    train_real_json
    val_real_json
    test_real_json

    dataset_root

    Returns
    -------
    list
    """

    dataset_root = Path(dataset_root)

    split_files = {
        "train": train_real_json,
        "val": val_real_json,
        "test": test_real_json,
    }

    master_index = []

    for split, json_file in split_files.items():
        real_entries = load_json(json_file)
        print(f"\nProcessing {split} split...")

        for real_path, _ in tqdm(real_entries):
            real_path_obj = Path(real_path)
            dataset = real_path_obj.parts[-2]
            filename = real_path_obj.name
            image_id = Path(filename).stem
            record = {
                "image_id": image_id,
                "dataset": dataset,
                "split": split,
                "real_image": real_path,
                "variants": [],
            }

            # --------------------------------------------------
            # Add all variants
            # --------------------------------------------------

            for generator in GENERATORS:
                for edit_type in EDIT_TYPES:
                    suffix = ( "_background" if edit_type == "Background" else "_stuff")
                    forged_image = (dataset_root/ "Forged"/ generator/ edit_type/ dataset/ f"{image_id}{suffix}.png")
                    mask = (dataset_root/ "Mask"/ edit_type/ dataset/ f"{image_id}{suffix}.png")

                    if not forged_image.exists():
                        print(
                            f"[WARNING] Missing forged image:\n"
                            f"{forged_image}"
                        )
                        continue

                    if not mask.exists():
                        print(
                            f"[WARNING] Missing mask:\n"
                            f"{mask}"
                        )

                        continue

                    record["variants"].append(
                        {
                            "generator": generator,
                            "edit_type": edit_type,
                            "forged_image":
                                "datasets/br-gen/"
                                + forged_image.relative_to(dataset_root).as_posix(),
                            "mask":
                                "datasets/br-gen/"
                                + mask.relative_to(dataset_root).as_posix(),
                        }
                    )

            master_index.append(record)

    return master_index


def validate_master_index(master_index):

    print("\n==========================================")
    print("MASTER INDEX STATISTICS")
    print("==========================================")

    print(f"Total Records : {len(master_index)}")

    split_counter = {}
    dataset_counter = {}
    variant_counter = 0

    for item in master_index:
        split_counter[item["split"]] = \
            split_counter.get(item["split"], 0) + 1

        dataset_counter[item["dataset"]] = \
            dataset_counter.get(item["dataset"], 0) + 1

        variant_counter += len(item["variants"])

    print("\nSplit Distribution")

    for k, v in split_counter.items():
        print(f"{k:<10}: {v}")

    print("\nDataset Distribution")

    for k, v in dataset_counter.items():
        print(f"{k:<10}: {v}")

    print(f"\nTotal Variants : {variant_counter}")

    avg = variant_counter / len(master_index)

    print(f"Average Variants/Image : {avg:.2f}")

    assert len(master_index) == 15000

    assert split_counter["train"] == 12000
    assert split_counter["val"] == 1500
    assert split_counter["test"] == 1500

    for record in master_index:
        assert len(record["variants"]) == 10, \
            f"{record['image_id']} has {len(record['variants'])} variants instead of 10."

    print("\nValidation successful.")


# ============================================================
# Final Index Creation and Validation
# ============================================================

def create_split_indices(master_index_json):
    """
    Create train_index.json,
    val_index.json,
    test_index.json
    from the master index.
    """

    master_index = load_json(master_index_json)

    train_index = {}
    val_index = {}
    test_index = {}

    split_lookup = {
        "train": train_index,
        "val": val_index,
        "test": test_index,
    }

    for record in master_index:

        split = record["split"]

        dataset = record["dataset"]

        image_id = record["image_id"]

        current_index = split_lookup[split]

        # --------------------------------------------------
        # Add real image
        # --------------------------------------------------

        real_key = get_real_key(
            dataset,
            image_id,
        )

        current_index[real_key] = {

            "label": 0,

            "file_name":
                Path(record["real_image"]).name,

            "image_path":
                record["real_image"],

            "mask_path":
                ""
        }

        # --------------------------------------------------
        # Add forged variants
        # --------------------------------------------------

        for variant in record["variants"]:

            fake_key = get_fake_key(
                variant["generator"],
                variant["edit_type"],
                dataset,
                image_id,
            )

            current_index[fake_key] = {

                "label": 1,

                "file_name":
                    Path(
                        variant["forged_image"]
                    ).name,

                "image_path":
                    variant["forged_image"],

                "mask_path":
                    variant["mask"]
            }

    return (
        train_index,
        val_index,
        test_index,
    )

def validate_split_indices(
    train_index,
    val_index,
    test_index,
):

    print("\n==============================")
    print("INDEX STATISTICS")
    print("==============================")

    expected = {
        "train": 132000,
        "val": 16500,
        "test": 16500,
    }

    indices = {
        "train": train_index,
        "val": val_index,
        "test": test_index,
    }

    for split, index in indices.items():

        real = sum(
            1
            for v in index.values()
            if v["label"] == 0
        )

        fake = sum(
            1
            for v in index.values()
            if v["label"] == 1
        )

        print(f"\n{split.upper()}")

        print(f"Real : {real}")

        print(f"Fake : {fake}")

        print(f"Total: {len(index)}")

        assert len(index) == expected[split]

    print("\nValidation successful.")

    
if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATASET_ROOT = Path(PROJECT_ROOT / "datasets"/ "br-gen")
    JSON_DIR = DATASET_ROOT / "dataset_jsons"
    OUTPUT_DIR = PROJECT_ROOT / "metadata" / "br-gen"

    ##### 1. Creation of Basic JSON Splits #####
    train_json = load_json(JSON_DIR / "br-gen_train.json")
    val_json = load_json(JSON_DIR / "real_val.json")

    train_real = extract_train_real(train_json)
    val_real = extract_val_real(val_json)
    test_real = create_test_real(train_real,val_real,DATASET_ROOT)

    validate_real_splits(train_real,val_real,test_real)

    save_json(train_real,OUTPUT_DIR / "train_real.json")
    save_json(val_real,OUTPUT_DIR / "val_real.json")
    save_json(test_real,OUTPUT_DIR / "test_real.json")

    ##### 2. Creation of Master Index #####
    master_index = create_master_index(train_real_json=OUTPUT_DIR / "train_real.json",val_real_json=OUTPUT_DIR / "val_real.json",
                                       test_real_json=OUTPUT_DIR / "test_real.json",dataset_root=DATASET_ROOT)
    validate_master_index(master_index)
    save_json(master_index,OUTPUT_DIR / "master_index.json")

    ##### 3. Creation of Dataset Split Indexes #####
    train_index, val_index, test_index = create_split_indices(OUTPUT_DIR / "master_index.json")
    validate_split_indices(train_index,val_index,test_index,)
    save_json(train_index,OUTPUT_DIR / "train_index.json",)
    save_json(val_index,OUTPUT_DIR / "val_index.json",)
    save_json(test_index,OUTPUT_DIR / "test_index.json",)
