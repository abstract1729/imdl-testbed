from pathlib import Path
import time
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BFREE_REPO_PATH = PROJECT_ROOT / "models" / "b-free" / "code"

if str(BFREE_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(BFREE_REPO_PATH))

from main_bfree_single import get_config

# ============================================================
# Project utilities
# ============================================================

from utils.experiment_utils.output_manager import create_output_csv_path
from utils.experiment_utils.csv_logger import CSVLogger
from utils.experiment_utils.device import get_device
from utils.experiment_utils.experiment_paths import parse_experiment_path
from utils.subsets.subset_config import (get_dataset_metadata_path,get_subset_config)

###### Dataset ######
from src.dataset.deal300k import DEAL300KDataset

###### Model ######
from src.model.b_free.preprocessing import BFreeTransform
from src.model.b_free.wrapper import BFreeInference
from src.model.b_free.postprocessing import extract_detection_output

# ============================================================
# Configuration
# ============================================================
DEVICE = get_device()      

####################################
TASK, DATASET, MODEL, EXPERIMENT = parse_experiment_path(__file__)

INDEX_FILE = "dataset_index.json"
METADATA_FILE = get_dataset_metadata_path(PROJECT_ROOT,DATASET, index_file = INDEX_FILE)
SUBSETS = get_subset_config(PROJECT_ROOT, DATASET, subset="default")

real_subset_files = SUBSETS["real"]
fake_subset_files = SUBSETS["fake"]

SUBSET_FILES = real_subset_files + fake_subset_files
####################################

WEIGHTS_DIR = PROJECT_ROOT / "models" / "b-free" / "weights"

###### Output Files #######
output_csv = create_output_csv_path(__file__)


####### Preprocessing #######
_, _, _, norm_type = get_config("BFREE_dino2reg4",weights_dir=str(WEIGHTS_DIR))
image_transform = BFreeTransform(norm_type = norm_type)
mask_transform = transforms.ToTensor()


# ============================================================
# Dataset and DataLoader
# ============================================================
dataset = DEAL300KDataset(project_root= PROJECT_ROOT, metadata_path=METADATA_FILE,  subset_files=SUBSET_FILES,
                            transform= image_transform,target_transform= mask_transform)

dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0, pin_memory=False)
progress_bar_desc = f"Running {MODEL} on {DATASET}"

# ============================================================
# Model
# ============================================================
model = BFreeInference(model_name="BFREE_dino2reg4", weights_dir=str(WEIGHTS_DIR), device=str(DEVICE))

###### CSV Logger ######
logger = CSVLogger(output_csv)

# ============================================================
# Inference
# ============================================================

for sample in tqdm(dataloader, desc = progress_bar_desc, unit="image"):
    image_id = sample["image_id"][0]
    file_name = sample["file_name"][0]

    label = sample["label"].item()
    image_path = sample["image_path"][0]
    image = sample["image"]

    try:
        start = time.perf_counter()
        raw_output = model.predict(image)
        detection = extract_detection_output(raw_output)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.log_success(image_id=image_id,file_name=file_name,label=label,image_path=image_path,
                           image_height=image.shape[-2],image_width=image.shape[-1],score=detection["score"],inference_time_ms=elapsed_ms)

    except Exception as e:
        logger.log_failure(image_id=image_id,file_name=file_name,label=label,image_path=image_path,
                           image_height=None,image_width=None,error_type=type(e).__name__,error_message=str(e))

# # ============================================================
# # Save
# # ============================================================

logger.save()
print(f"Results saved to: {output_csv}")