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

# ============================================================
# Project utilities
# ============================================================

from utils.experiment_utils.output_manager import create_output_csv_path
from utils.experiment_utils.csv_logger import CSVLogger
from utils.experiment_utils.device import get_device
from utils.experiment_utils.experiment_paths import parse_experiment_path
from utils.subsets.subset_config import (get_dataset_metadata_path,get_subset_config)

###### Dataset ######
from src.dataset.csi_imd import CSIIMDDataset

###### Model ######
from src.model.pgc_net.preprocessing import PGCTransform
from src.model.pgc_net.wrapper import PGCInference
from src.model.pgc_net.postprocessing import extract_detection_output

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

CHECKPOINT_DINO = PROJECT_ROOT / "models" / "pgc-net" / "pretrained_dino" / "dinov2-large"
MODEL_CKPT = PROJECT_ROOT / "models" / "pgc-net" / "pretrained_model" / "PGC_train_sdv1_4_ckpt.pth"

###### Output Files #######
output_csv = create_output_csv_path(__file__)


####### Preprocessing #######
image_transform = PGCTransform()
mask_transform = transforms.ToTensor()


# ============================================================
# Dataset and DataLoader
# ============================================================
dataset = CSIIMDDataset(project_root= PROJECT_ROOT, metadata_path=METADATA_FILE,  subset_files=SUBSET_FILES,
                            transform= image_transform,target_transform= mask_transform)

dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0, pin_memory=False)
progress_bar_desc = f"Running {MODEL} on {DATASET}"

# ============================================================
# Model
# ============================================================
model = PGCInference(checkpoint_path=str(MODEL_CKPT),dino_variant="dinov2-large",pretrained_root=str(CHECKPOINT_DINO))

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

        # progress_bar.set_postfix( score=f"{detection['score']:.4f}", time=f"{elapsed_ms:.1f} ms")

    except Exception as e:
        logger.log_failure(image_id=image_id,file_name=file_name,label=label,image_path=image_path,
                           image_height=None,image_width=None,error_type=type(e).__name__,error_message=str(e))
        # progress_bar.set_postfix(error=type(e).__name__)

# # ============================================================
# # Save
# # ============================================================

logger.save()
print(f"Results saved to: {output_csv}")