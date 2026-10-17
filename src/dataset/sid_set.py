from pathlib import Path
import json

from PIL import Image
from torch.utils.data import Dataset


class SIDSetDataset(Dataset):
    """
    SID-Set Dataset (Detection)

    Expected Metadata Structure
    ---------------------------
    metadata/
        sid-set/
            dataset_index.json
            subsets/
                real_100.txt
                fake_100.txt
                ...

    Parameters
    ----------
    metadata_path : str | Path
        Path to dataset_index.json.

    subset_files : str | Path | list | None
        None -> load the complete dataset.

        Otherwise one or more subset txt files.

    transform
        Image transform.

    target_transform
        Reserved for future localization support.
    """

    def __init__(self,project_root,metadata_path,subset_files=None,transform=None,target_transform=None):

        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found:\n{self.metadata_path}")

        self.project_root = project_root
        self.transform = transform
        self.target_transform = target_transform

        # --------------------------------------------------------
        # Load Metadata
        # --------------------------------------------------------

        with open(self.metadata_path, "r") as f:
            self.dataset_index = json.load(f)

        # --------------------------------------------------------
        # Build Sample List
        # --------------------------------------------------------

        if subset_files is None:
            image_ids = sorted(self.dataset_index.keys())
        else:
            image_ids = self._load_subset_ids(subset_files)

        self.samples = []
        for image_id in image_ids:
            if image_id not in self.dataset_index:
                raise KeyError(f"{image_id} not present in dataset_index.json")

            sample = self.dataset_index[image_id].copy()
            sample["image_id"] = image_id
            self.samples.append(sample)

    # ------------------------------------------------------------

    def _load_subset_ids(self, subset_files):
        if isinstance(subset_files, (str, Path)):
            subset_files = [subset_files]

        image_ids = []
        for subset_file in subset_files:
            subset_file = Path(subset_file)
            if not subset_file.exists():
                raise FileNotFoundError(subset_file)

            with open(subset_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("#"):
                        continue

                    image_ids.append(line)

        # Preserve order while removing duplicates

        image_ids = list(dict.fromkeys(image_ids))
        return image_ids

    # ------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.project_root / sample["image_path"]
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # --------------------------------------------------------
        # Dummy Mask (Detection Only)
        # --------------------------------------------------------

        mask = Image.new(mode="L",size=image.size,color=0)

        # --------------------------------------------------------
        # Transforms
        # --------------------------------------------------------

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        # --------------------------------------------------------

        return {
            "image_id": sample["image_id"],
            "file_name": sample["file_name"],
            "image": image,
            "mask": mask,
            "label": sample["label"],
            "image_path": str(image_path),
            "mask_path": sample["mask_path"],
        }

    # ------------------------------------------------------------

    @property
    def num_real(self):

        return sum(
            sample["label"] == 0
            for sample in self.samples
        )

    # ------------------------------------------------------------

    @property
    def num_fake(self):

        return sum(
            sample["label"] == 1
            for sample in self.samples
        )

    # ------------------------------------------------------------

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  metadata='{self.metadata_path}',\n"
            f"  real_images={self.num_real},\n"
            f"  fake_images={self.num_fake},\n"
            f"  total_images={len(self.samples)}\n"
            f")"
        )