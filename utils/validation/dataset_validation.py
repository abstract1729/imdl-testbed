from pathlib import Path
import json


REQUIRED_FIELDS = {
    "label",
    "file_name",
    "image_path",
    "mask_path",
}


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def load_dataset_index(metadata_path):
    metadata_path = Path(metadata_path)
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_subset(subset_file):
    subset = []

    with open(subset_file, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue
            subset.append(line)
    return subset


# -------------------------------------------------------------
# Dataset Index Validation
# -------------------------------------------------------------

def validate_dataset_index(metadata_path):

    dataset_index = load_dataset_index(metadata_path)

    results = {
        "passed": True,
        "num_real": 0,
        "num_fake": 0,
        "missing_images": [],
        "invalid_labels": [],
        "missing_fields": [],
        "filename_mismatch": [],
    }

    for image_id, sample in dataset_index.items():

        # Required fields

        missing = REQUIRED_FIELDS - sample.keys()
        if missing:
            results["passed"] = False
            results["missing_fields"].append((image_id, sorted(missing)))
            continue

        # Label

        if sample["label"] not in (0, 1):
            results["passed"] = False
            results["invalid_labels"].append(image_id)

        elif sample["label"] == 0:
            results["num_real"] += 1

        else:
            results["num_fake"] += 1

        # Image Exists

        image_path = Path(sample["image_path"])
        if not image_path.exists():
            results["passed"] = False
            results["missing_images"].append(image_id)

        # File Name

        if image_path.name != sample["file_name"]:
            results["passed"] = False
            results["filename_mismatch"].append(image_id)

    return results


# -------------------------------------------------------------
# Subset Validation
# -------------------------------------------------------------

def validate_subset(metadata_path, subset_file):
    dataset_index = load_dataset_index(metadata_path)
    subset = load_subset(subset_file)
    subset_name = Path(subset_file).name
    expected_label = 0 if subset_name.startswith("real") else 1

    results = {
        "passed": True,
        "count": len(subset),
        "duplicates": [],
        "missing_ids": [],
        "label_errors": [],
    }

    seen = set()

    for image_id in subset:
        if image_id in seen:
            results["duplicates"].append(image_id)
            results["passed"] = False

        seen.add(image_id)

        if image_id not in dataset_index:
            results["missing_ids"].append(image_id)
            results["passed"] = False
            continue

        if dataset_index[image_id]["label"] != expected_label:

            results["label_errors"].append(image_id)

            results["passed"] = False

    return results


# -------------------------------------------------------------
# Validation Error Display
# -------------------------------------------------------------

def _raise_validation_error(report):

    dataset = report["dataset_index"]

    if not dataset["passed"]:

        raise RuntimeError(
            "Dataset index validation failed.\n"
            f"Missing Images   : {len(dataset['missing_images'])}\n"
            f"Invalid Labels   : {len(dataset['invalid_labels'])}\n"
            f"Missing Fields   : {len(dataset['missing_fields'])}\n"
            f"Filename Errors  : {len(dataset['filename_mismatch'])}"
        )

    for subset_name, subset in report["subsets"].items():

        if not subset["passed"]:

            raise RuntimeError(
                f"Subset validation failed ({subset_name}).\n"
                f"Duplicates  : {len(subset['duplicates'])}\n"
                f"Missing IDs : {len(subset['missing_ids'])}\n"
                f"Label Errors: {len(subset['label_errors'])}"
            )


# -------------------------------------------------------------
# Main Validation
# -------------------------------------------------------------

def validate(metadata_path, subset_dir=None, raise_on_error = True):

    metadata_results = validate_dataset_index(metadata_path)
    subset_results = {}

    if subset_dir is not None:
        subset_dir = Path(subset_dir)
        for subset_file in sorted(subset_dir.glob("*.txt")):
            subset_results[subset_file.name] = validate_subset(metadata_path,subset_file)

    report = {
        "dataset_index": metadata_results,
        "subsets": subset_results,
    }

    if raise_on_error:
        _raise_validation_error(report)

    return report


# -------------------------------------------------------------
# Pretty Printing
# -------------------------------------------------------------

def print_report(report):
    print("\n========== Dataset Index ==========")
    dataset = report["dataset_index"]
    print("Passed :", dataset["passed"])
    print("Real   :", dataset["num_real"])
    print("Fake   :", dataset["num_fake"])
    print("Missing Images :", len(dataset["missing_images"]))
    print("Invalid Labels :", len(dataset["invalid_labels"]))
    print("Missing Fields :", len(dataset["missing_fields"]))
    print("Filename Errors:", len(dataset["filename_mismatch"]))
    print("\n========== Subsets ==========")

    for subset_name, result in report["subsets"].items():
        print(f"\n{subset_name}")
        print("-" * len(subset_name))
        print("Passed :", result["passed"])
        print("Count  :", result["count"])
        print("Duplicates :", len(result["duplicates"]))
        print("Missing IDs:", len(result["missing_ids"]))
        print("Label Errors:", len(result["label_errors"]))


if __name__ == "__main__":
    raise RuntimeError("Import validate() from another script.")