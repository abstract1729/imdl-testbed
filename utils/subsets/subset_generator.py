"""
Generic Dataset Subset Generator

Creates deterministic subsets from a dataset_index.json file.

Output
------
metadata/
    <dataset>/
        subsets/
            real_1000.txt
            fake_1000.txt
"""

from pathlib import Path
import json
import random
from datetime import datetime


def load_dataset_index(metadata_path):
    """
    Load dataset_index.json.
    """
    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with open(metadata_path, "r") as f:
        return json.load(f)


def generate_subset(metadata_path,output_dir,num_real=None,num_fake=None,sampling="random",seed=42):
    """
    Generate deterministic real/fake subsets.

    Parameters
    ----------
    metadata_path : str | Path
        dataset_index.json
    output_dir : str | Path
        Directory where subset txt files are written.
    num_real : int | None
    num_fake : int | None
    sampling : {"random", "first"}
    seed : int
    """

    dataset_index = load_dataset_index(metadata_path)

    real_ids = []
    fake_ids = []
    metadata_path = Path(metadata_path)
    dataset_name = metadata_path.parent.name

    for image_id, sample in dataset_index.items():

        if sample["label"] == 0:
            real_ids.append(image_id)

        elif sample["label"] == 1:
            fake_ids.append(image_id)

        else:
            raise ValueError(f"Invalid label for {image_id}: {sample['label']}")

    real_ids.sort()
    fake_ids.sort()

    rng = random.Random(seed)

    # ------------------------------------------------------------
    # Real
    # ------------------------------------------------------------

    if num_real is not None:

        if num_real > len(real_ids):
            raise ValueError(
                f"Requested {num_real} real samples "
                f"but only {len(real_ids)} exist."
            )

        if sampling == "random":
            real_ids = sorted(
                rng.sample(real_ids, num_real)
            )

        elif sampling == "first":
            real_ids = real_ids[:num_real]

        else:
            raise ValueError(
                "sampling must be 'random' or 'first'"
            )

    # ------------------------------------------------------------
    # Fake
    # ------------------------------------------------------------

    if num_fake is not None:

        if num_fake > len(fake_ids):
            raise ValueError(
                f"Requested {num_fake} fake samples "
                f"but only {len(fake_ids)} exist."
            )

        if sampling == "random":
            fake_ids = sorted(
                rng.sample(fake_ids, num_fake)
            )

        elif sampling == "first":
            fake_ids = fake_ids[:num_fake]

        else:
            raise ValueError(
                "sampling must be 'random' or 'first'"
            )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_file = output_dir / f"real_{len(real_ids)}.txt"
    fake_file = output_dir / f"fake_{len(fake_ids)}.txt"    

    with open(real_file, "w") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        f.write(
            "############################################################\n"
            "# Benchmark Subset Metadata\n"
            "############################################################\n\n"

            f"# Metadata File : {metadata_path.as_posix()}\n"
            f"# Dataset       : {dataset_name}\n"
            f"# Generated On  : {timestamp}\n"
            f"# Sampling      : {sampling}\n"
            f"# Seed          : {seed}\n"
            f"# Selected      : {len(real_ids)}\n\n"

            "############################################################\n\n"
        )

        for image_id in real_ids:
            f.write(f"{image_id}\n")

    with open(fake_file, "w") as f:

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            "############################################################\n"
            "# Benchmark Subset Metadata\n"
            "############################################################\n\n"

            f"# Metadata File : {metadata_path.as_posix()}\n"
            f"# Dataset       : {dataset_name}\n"
            f"# Generated On  : {timestamp}\n"
            f"# Sampling      : {sampling}\n"
            f"# Seed          : {seed}\n"
            f"# Selected      : {len(real_ids)}\n\n"

            "############################################################\n\n"
        )

        for image_id in fake_ids:
            f.write(f"{image_id}\n")

    print("\nSubset generation complete.")

    print(f"Real : {real_file}")
    print(f"Fake : {fake_file}")

    print(
        f"\nSelected "
        f"{len(real_ids)} real "
        f"and {len(fake_ids)} fake samples."
    )


if __name__ == "__main__":
    raise RuntimeError( "Use generate_subset() from another script.")