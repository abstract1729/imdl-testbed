from pathlib import Path
from datetime import datetime


def create_output_csv_path(experiment_file):
    """
    Create output CSV path corresponding to an experiment file.

    Example
    -------
    experiments-pipeline/
        detection/
            csi-imd/
                trufor/
                    experiment1.py

    becomes

    outputs/
        detection/
            csi-imd/
                trufor/
                    experiment1_csi-imd_trufor_20260731_134512.csv
    """

    experiment_file = Path(experiment_file).resolve()
    parts = experiment_file.parts

    try:
        idx = parts.index("experiments-pipeline")
    except ValueError:
        raise ValueError("'experiments-pipeline' not found in experiment path.")
    relative_parts = parts[idx + 1 :]

    if len(relative_parts) != 4:
        raise ValueError(
            "Expected experiment path:\n"
            "experiments-pipeline/<task>/<dataset>/<model>/<experiment>.py")

    task, dataset, model, experiment_file_name = relative_parts
    experiment_name = Path(experiment_file_name).stem
    project_root = Path(*parts[:idx])
    output_dir = (project_root/ "outputs"/ task/ dataset/ model)
    output_dir.mkdir(parents=True,exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = (
        f"{experiment_name}"
        f"_{dataset}"
        f"_{model}"
        f"_{timestamp}.csv"
    )

    return output_dir / output_file



def list_output_csvs(experiment_file):
    """
    Return all output CSVs corresponding to an experiment.

    Example
    -------
    Experiment
        experiments-pipeline/
            detection/
                csi-imd/
                    trufor/
                        experiment1.py

    Searches

        outputs/
            detection/
                csi-imd/
                    trufor/

    for files matching

        experiment1_csi-imd_trufor_*.csv

    Returns
    -------
    list[Path]

        Sorted from newest to oldest.
        Returns an empty list if none exist.
    """

    experiment_file = Path(experiment_file).resolve()

    parts = experiment_file.parts

    try:
        idx = parts.index("experiments-pipeline")
    except ValueError:
        raise ValueError("'experiments-pipeline' not found in experiment path.")

    relative_parts = parts[idx + 1:]

    if len(relative_parts) != 4:
        raise ValueError(
            "Expected experiment path:\n"
            "experiments-pipeline/<task>/<dataset>/<model>/<experiment>.py"
        )

    task, dataset, model, experiment_filename = relative_parts
    experiment_name = Path(experiment_filename).stem
    project_root = Path(*parts[:idx])
    output_dir = (project_root/ "outputs"/ task/ dataset/ model)
    if not output_dir.exists():
        return []

    pattern = (
        f"{experiment_name}"
        f"_{dataset}"
        f"_{model}"
        "_*.csv"
    )

    csv_files = sorted(output_dir.glob(pattern),key=lambda p: p.stat().st_mtime,reverse=True)
    return csv_files


def get_latest_output_csv(experiment_file):
    """
    Return the latest CSV for an experiment.

    Returns
    -------
    Path | None
    """

    csv_files = list_output_csvs(experiment_file)

    if not csv_files:
        return None

    return csv_files[0]