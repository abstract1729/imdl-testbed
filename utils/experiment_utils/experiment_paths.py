from pathlib import Path


def parse_experiment_path(file_path):
    """
    Parse task, dataset, model, and experiment name from an experiment file path.

    Expected structure:
        experiments-pipeline/
            <task>/
                <dataset>/
                    <model>/
                        experimentK.py

    Returns
    -------
    tuple[str, str, str, str]
        task, dataset, model, experiment
    """

    file_path = Path(file_path).resolve()
    parts = file_path.parts

    try:
        idx = parts.index("experiments-pipeline")
    except ValueError:
        raise ValueError(
            "'experiments-pipeline' not found in experiment path."
        )

    relative_parts = parts[idx + 1:]

    if len(relative_parts) != 4:
        raise ValueError(
            "Expected experiment path:\n"
            "experiments-pipeline/<task>/<dataset>/<model>/<experiment>.py"
        )

    task, dataset, model, experiment_file = relative_parts

    experiment = Path(experiment_file).stem

    return task, dataset, model, experiment