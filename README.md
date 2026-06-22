# IMDL-Testbed

IMDL-Testbed is a research-oriented experimental framework for Image Manipulation Detection and Localization (IMDL). The repository is designed to support reproducible evaluation of multiple state-of-the-art forensic models across diverse datasets while keeping individual model repositories as independent black boxes. Experiment-specific code, metrics, visualizations, and analyses are maintained separately from model implementations, making the framework easily extensible to new datasets and methods.

## Directory Structure

- `datasets/`
  - Contains datasets, masks, metadata files, and dataset-specific analysis notebooks.

- `models/`
  - Contains the original repositories of IMDL methods. Model repositories are treated as immutable black boxes except for minimal compatibility fixes.

- `experiments/`
  - Stores task-specific experiments organized as:



Each experiment contains code for inference, evaluation, metrics computation, visualizations, and analysis.

### Currently implemented:

- **Tasks**
    - Localization
    - Detection

- **Datasets**
    - CSI-IMD

- **Models**
    - TruFor

The framework is intended to scale to additional datasets and models with minimal changes.