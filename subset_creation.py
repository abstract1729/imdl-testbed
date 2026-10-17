from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.subsets.subset_generator import generate_subset
from utils.validation.dataset_validation import (validate,print_report)

PROJECT_ROOT = Path(__file__).resolve().parent
# Ask the user for input
user_choice = input("Do you want to create the subsets? (Y/N): ")

# Run the block only if the user types 'y' or 'Y'
if user_choice.lower() == 'y':
    print("Creating subsets...")
    ############ CSI-IMD Dataset
    metadata_path = (PROJECT_ROOT/ "metadata"/ "csi-imd"/ "dataset_index.json")
    subset_dir = (PROJECT_ROOT/ "metadata"/ "csi-imd"/ "subsets")

    generate_subset(metadata_path=metadata_path,output_dir=subset_dir,num_real=None,num_fake=None,sampling="random",seed=42)
    report = validate(metadata_path= metadata_path,subset_dir=subset_dir, raise_on_error = True)
    print_report(report)



    ############ DEAL-300K Dataset
    metadata_path = (PROJECT_ROOT/ "metadata"/ "deal-300k"/ "dataset_index.json")
    subset_dir = (PROJECT_ROOT/ "metadata"/ "deal-300k"/ "subsets")

    generate_subset(metadata_path=metadata_path,output_dir=subset_dir,num_real=3500,num_fake=3500,sampling="random",seed=42)
    report = validate(metadata_path= metadata_path,subset_dir=subset_dir, raise_on_error = True)
    print_report(report)



    ############ IMG-EDIT Dataset
    metadata_path = (PROJECT_ROOT/ "metadata"/ "img-edit" / "dataset_index.json")
    subset_dir = (PROJECT_ROOT/ "metadata"/ "img-edit" / "subsets")

    generate_subset(metadata_path=metadata_path,output_dir=subset_dir,num_real=3500,num_fake=3500,sampling="random",seed=42)
    report = validate(metadata_path= metadata_path,subset_dir=subset_dir, raise_on_error = True)
    print_report(report)



    ############ Auto-Splice Dataset
    metadata_path = (PROJECT_ROOT/ "metadata"/ "auto-splice" / "dataset_index.json")
    subset_dir = (PROJECT_ROOT/ "metadata"/ "auto-splice" / "subsets")

    generate_subset(metadata_path=metadata_path,output_dir=subset_dir,num_real=None,num_fake=None,sampling="random",seed=42)
    report = validate(metadata_path= metadata_path,subset_dir=subset_dir, raise_on_error = True)
    print_report(report)


    ############ SID-Set Dataset
    metadata_path = (PROJECT_ROOT/ "metadata"/ "sid-set" / "dataset_index.json")
    subset_dir = (PROJECT_ROOT/ "metadata"/ "sid-set" / "subsets")

    generate_subset(metadata_path=metadata_path,output_dir=subset_dir,num_real=None,num_fake=None,sampling="random",seed=42)
    report = validate(metadata_path= metadata_path,subset_dir=subset_dir, raise_on_error = True)
    print_report(report)


    ############ BR-Gen Dataset
    metadata_path = (PROJECT_ROOT/ "metadata"/ "br-gen" / "test_index.json")
    subset_dir = (PROJECT_ROOT/ "metadata"/ "br-gen" / "subsets")

    generate_subset(metadata_path=metadata_path,output_dir=subset_dir,num_real=None,num_fake=3000,sampling="random",seed=42)
    report = validate(metadata_path= metadata_path,subset_dir=subset_dir, raise_on_error = True)
    print_report(report)