"""
training_subset_generator.py

Creates balanced training subsets for BR-Gen.

Unlike subset_generator.py, this module performs
hierarchical stratified sampling over

1. Dataset
2. Generator
3. Edit Type

to ensure balanced training subsets.

Author: Arjun Mallick
"""

from pathlib import Path
from collections import defaultdict
import json
import random
from datetime import datetime


# ============================================================
# Utility
# ============================================================

def load_dataset_index(metadata_path):
    """
    Load train_index.json.
    """

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with open(metadata_path, "r") as f:
        return json.load(f)


# ============================================================
# Key Parsing
# ============================================================

def parse_real_key(key):
    """
    real_coco_000000123456

    Returns
    -------
    {
        dataset,
        image_id
    }
    """

    tokens = key.split("_", 2)

    return {
        "dataset": tokens[1],
        "image_id": tokens[2],
    }


def parse_fake_key(key):
    """
    fake_brushnet_background_coco_000000123456

    Returns
    -------
    {
        generator,
        edit_type,
        dataset,
        image_id
    }
    """

    tokens = key.split("_", 4)

    return {
        "generator": tokens[1],
        "edit_type": tokens[2],
        "dataset": tokens[3],
        "image_id": tokens[4],
    }


# ============================================================
# Grouping
# ============================================================

def group_real_samples(dataset_index):
    """
    Group real samples by dataset.

    Output
    ------
    {
        "coco": [...],
        "imagenet": [...],
        "places": [...]
    }
    """

    groups = defaultdict(list)

    for sample_id, sample in dataset_index.items():

        if sample["label"] != 0:
            continue

        info = parse_real_key(sample_id)

        groups[info["dataset"]].append(sample_id)

    return groups


def group_fake_samples(dataset_index):
    """
    Hierarchical grouping

    generator
        ↓
    edit_type
        ↓
    dataset

    Output
    ------

    {
        brushnet:
        {
            background:
            {
                coco: [...],
                imagenet: [...],
                places: [...]
            },

            stuff:
            {
                ...
            }
        }
    }
    """

    groups = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    for sample_id, sample in dataset_index.items():

        if sample["label"] != 1:
            continue

        info = parse_fake_key(sample_id)

        groups[
            info["generator"]
        ][
            info["edit_type"]
        ][
            info["dataset"]
        ].append(sample_id)

    return groups


# ============================================================
# Sampling
# ============================================================

def balanced_random_sample(
    samples,
    n,
    rng,
):
    """
    Deterministic random sampling.

    Parameters
    ----------
    samples : list

    n : int

    rng : random.Random

    Returns
    -------
    sorted(list)
    """

    if n > len(samples):

        raise ValueError(
            f"Requested {n} samples "
            f"but only {len(samples)} available."
        )

    return sorted(
        rng.sample(samples, n)
    )


def allocate_counts(
    total,
    num_groups,
):
    """
    Evenly distribute

    total

    across

    num_groups.

    Example
    -------

    allocate_counts(1000,3)

    ->
    [334,333,333]
    """

    base = total // num_groups

    remainder = total % num_groups

    allocation = [
        base
        for _ in range(num_groups)
    ]

    for i in range(remainder):
        allocation[i] += 1

    return allocation


# ============================================================
# Real Sampling
# ============================================================

def sample_real_groups(
    grouped_reals,
    num_real,
    rng,
):
    """
    Balanced sampling across

    COCO
    ImageNet
    Places
    """

    datasets = sorted(
        grouped_reals.keys()
    )

    allocation = allocate_counts(
        num_real,
        len(datasets),
    )

    selected = []

    for dataset, n in zip(
        datasets,
        allocation,
    ):

        selected.extend(

            balanced_random_sample(
                grouped_reals[dataset],
                n,
                rng,
            )

        )

    return sorted(selected)


# ============================================================
# Fake Sampling
# ============================================================

def sample_fake_groups(
    grouped_fakes,
    num_fake,
    rng,
):
    """
    Balanced over

    Generator
        ↓
    Edit Type
        ↓
    Dataset
    """

    selected = []

    generators = sorted(
        grouped_fakes.keys()
    )

    generator_alloc = allocate_counts(
        num_fake,
        len(generators),
    )

    for generator, gen_count in zip(
        generators,
        generator_alloc,
    ):

        edit_groups = grouped_fakes[generator]

        edit_types = sorted(
            edit_groups.keys()
        )

        edit_alloc = allocate_counts(
            gen_count,
            len(edit_types),
        )

        for edit_type, edit_count in zip(
            edit_types,
            edit_alloc,
        ):

            dataset_groups = edit_groups[
                edit_type
            ]

            datasets = sorted(
                dataset_groups.keys()
            )

            dataset_alloc = allocate_counts(
                edit_count,
                len(datasets),
            )

            for dataset, dataset_count in zip(
                datasets,
                dataset_alloc,
            ):

                selected.extend(

                    balanced_random_sample(
                        dataset_groups[dataset],
                        dataset_count,
                        rng,
                    )

                )

    return sorted(selected)


# ============================================================
# File Writer
# ============================================================

def write_subset_file(
    output_file,
    sample_ids,
    metadata_path,
    sampling,
    seed,
):
    """
    Write subset file.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(output_file, "w") as f:

        f.write(
            "############################################################\n"
            "# BR-Gen Training Subset Metadata\n"
            "############################################################\n\n"
        )

        f.write(
            f"# Metadata File : {metadata_path}\n"
        )

        f.write(
            "# Sampling      : "
            f"{sampling}\n"
        )

        f.write(
            f"# Seed          : {seed}\n"
        )

        f.write(
            "# Selected      : "
            f"{len(sample_ids)}\n"
        )

        f.write(
            f"# Generated On  : {timestamp}\n\n"
        )

        f.write(
            "############################################################\n\n"
        )

        for sample in sample_ids:
            f.write(sample + "\n")


# ============================================================
# Distribution Summary
# ============================================================

def print_real_distribution(
    grouped,
    selected,
):
    """
    Print selected real distribution.
    """

    print("\nReal Distribution")
    print("------------------------------")

    selected = set(selected)

    for dataset in sorted(grouped.keys()):

        count = sum(
            1
            for x in grouped[dataset]
            if x in selected
        )

        print(
            f"{dataset:<12}: {count}"
        )


def print_fake_distribution(
    grouped,
    selected,
):
    """
    Print selected fake distribution.
    """

    print("\nFake Distribution")
    print("------------------------------")

    selected = set(selected)

    for generator in sorted(grouped.keys()):

        print(generator)

        for edit_type in sorted(
            grouped[generator].keys()
        ):

            print(
                f"  {edit_type}"
            )

            for dataset in sorted(
                grouped[generator][edit_type].keys()
            ):

                count = sum(
                    1
                    for x in grouped[
                        generator
                    ][
                        edit_type
                    ][
                        dataset
                    ]
                    if x in selected
                )

                print(
                    f"      {dataset:<10}: {count}"
                )

        print()


# ============================================================
# Validation
# ============================================================

def validate_subset(
    real_subset,
    fake_subset,
):
    """
    Basic sanity checks.
    """

    if len(real_subset) != len(set(real_subset)):
        raise RuntimeError(
            "Duplicate real samples detected."
        )

    if len(fake_subset) != len(set(fake_subset)):
        raise RuntimeError(
            "Duplicate fake samples detected."
        )

    overlap = (
        set(real_subset)
        &
        set(fake_subset)
    )

    if overlap:

        raise RuntimeError(
            "Real/Fake overlap detected."
        )


# ============================================================
# Public API
# ============================================================

def generate_brgen_train_subset(
    metadata_path,
    output_dir,
    num_real,
    num_fake,
    seed=42,
):
    """
    Generate balanced BR-Gen training subsets.

    Parameters
    ----------
    metadata_path
        train_index.json

    output_dir

    num_real

    num_fake

    seed
    """

    dataset_index = load_dataset_index(
        metadata_path
    )

    rng = random.Random(seed)

    # --------------------------------------------------
    # Group
    # --------------------------------------------------

    grouped_reals = group_real_samples(
        dataset_index
    )

    grouped_fakes = group_fake_samples(
        dataset_index
    )

    # --------------------------------------------------
    # Sample
    # --------------------------------------------------

    real_subset = sample_real_groups(
        grouped_reals,
        num_real,
        rng,
    )

    fake_subset = sample_fake_groups(
        grouped_fakes,
        num_fake,
        rng,
    )

    # --------------------------------------------------
    # Validate
    # --------------------------------------------------

    validate_subset(
        real_subset,
        fake_subset,
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    real_file = (
        output_dir
        /
        f"train_real_{len(real_subset)}.txt"
    )

    fake_file = (
        output_dir
        /
        f"train_fake_{len(fake_subset)}.txt"
    )

    write_subset_file(
        output_file=real_file,
        sample_ids=real_subset,
        metadata_path=metadata_path,
        sampling="balanced",
        seed=seed,
    )

    write_subset_file(
        output_file=fake_file,
        sample_ids=fake_subset,
        metadata_path=metadata_path,
        sampling="balanced",
        seed=seed,
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n==========================================")
    print("BR-GEN TRAIN SUBSET")
    print("==========================================")

    print(
        f"Real Selected : {len(real_subset)}"
    )

    print(
        f"Fake Selected : {len(fake_subset)}"
    )

    print_real_distribution(
        grouped_reals,
        real_subset,
    )

    print_fake_distribution(
        grouped_fakes,
        fake_subset,
    )

    print(
        "\nSubset generation complete."
    )

    print(
        f"\nReal : {real_file}"
    )

    print(
        f"Fake : {fake_file}"
    )

    return (
        real_subset,
        fake_subset,
    )



# ============================================================
# Debug Utilities
# ============================================================

def print_dataset_statistics(metadata_path):
    """
    Print the complete distribution of the training index.

    Useful for verifying that the dataset itself
    is balanced before sampling.
    """

    dataset_index = load_dataset_index(metadata_path)

    grouped_reals = group_real_samples(dataset_index)
    grouped_fakes = group_fake_samples(dataset_index)

    print("\n==========================================")
    print("TRAIN DATASET STATISTICS")
    print("==========================================")

    print("\nReal Images")
    print("------------------------------")

    total_real = 0

    for dataset in sorted(grouped_reals.keys()):

        count = len(grouped_reals[dataset])

        total_real += count

        print(f"{dataset:<12}: {count}")

    print(f"\nTotal Real : {total_real}")

    print("\nFake Images")
    print("------------------------------")

    total_fake = 0

    for generator in sorted(grouped_fakes.keys()):

        print(generator)

        generator_total = 0

        for edit_type in sorted(
            grouped_fakes[generator].keys()
        ):

            print(f"  {edit_type}")

            edit_total = 0

            for dataset in sorted(
                grouped_fakes[generator][edit_type].keys()
            ):

                count = len(
                    grouped_fakes[generator][edit_type][dataset]
                )

                edit_total += count

                print(
                    f"      {dataset:<10}: {count}"
                )

            print(f"      Total      : {edit_total}")

            generator_total += edit_total

        print(f"  Generator Total : {generator_total}\n")

        total_fake += generator_total

    print(f"Total Fake : {total_fake}")

    print("\n==========================================")


# ============================================================
# Verify Saved Subset
# ============================================================

def verify_subset_file(txt_file):
    """
    Verify subset file.

    Ensures

    - no duplicates
    - prints number of samples
    """

    txt_file = Path(txt_file)

    with open(txt_file, "r") as f:

        ids = [
            line.strip()
            for line in f
            if line.strip()
            and not line.startswith("#")
        ]

    print("\nVerification")
    print("--------------------------")

    print(f"Samples : {len(ids)}")

    print(f"Unique  : {len(set(ids))}")

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate samples detected."
        )

    print("Verification successful.")


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    METADATA_PATH = PROJECT_ROOT/ "metadata"/ "br-gen" /"train_index.json"
    OUTPUT_DIR = PROJECT_ROOT / "metadata" / "br-gen"/ "training_subsets"
    print_dataset_statistics(METADATA_PATH)
    real_subset, fake_subset = generate_brgen_train_subset(metadata_path=METADATA_PATH,output_dir=OUTPUT_DIR,
                                                           num_real=6000,num_fake=6000,seed=42)

    verify_subset_file(Path(OUTPUT_DIR)/ f"train_real_{len(real_subset)}.txt")
    verify_subset_file(Path(OUTPUT_DIR)/ f"train_fake_{len(fake_subset)}.txt")

