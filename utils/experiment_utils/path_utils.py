from pathlib import Path

def get_project_root():
    path = Path(__file__).resolve()

    while path.name != "IMDL-TESTBED":
        path = path.parent

    return path