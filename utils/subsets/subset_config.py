from pathlib import Path
import yaml


def get_dataset_metadata_path(project_root, dataset, index_file):
    """
    Return the standard dataset metadata index path.

    Expected structure:
        metadata/
            <dataset>/
                dataset_index.json
    """

    project_root = Path(project_root).resolve()
    metadata_path = (project_root/ "metadata"/ dataset/ index_file)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata file not found:\n{metadata_path}"
        )

    return metadata_path


def get_subset_config(project_root,dataset,subset="default",config_path=None):
    """
    Load subset configuration for a dataset.

    Expected YAML structure:

        auto-splice:
          default:
            real:
              - metadata/auto-splice/subsets/real_250.txt
            fake:
              - metadata/auto-splice/subsets/fake_250.txt

    Returns
    -------
    dict
        {
            "real": [Path, ...],
            "fake": [Path, ...]
        }
    """

    project_root = Path(project_root).resolve()

    if config_path is None:
        config_path = project_root / "configs" / "subsets.yaml"
    else:
        config_path = Path(config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Subset configuration file not found:\n{config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if dataset not in config:
        raise KeyError(
            f"Dataset '{dataset}' not found in subset configuration:\n"
            f"{config_path}"
        )

    dataset_config = config[dataset]

    if subset not in dataset_config:
        raise KeyError(
            f"Subset '{subset}' not found for dataset '{dataset}'.\n"
            f"Available subsets: {list(dataset_config.keys())}"
        )

    subset_config = dataset_config[subset]

    if "real" not in subset_config or "fake" not in subset_config:
        raise KeyError(
            f"Subset '{subset}' for dataset '{dataset}' must contain "
            "'real' and 'fake' entries."
        )

    real_files = [
        project_root / Path(path)
        for path in subset_config["real"]
    ]

    fake_files = [
        project_root / Path(path)
        for path in subset_config["fake"]
    ]

    for path in real_files + fake_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Subset file not found:\n{path}"
            )

    return {
        "real": real_files,
        "fake": fake_files,
    }