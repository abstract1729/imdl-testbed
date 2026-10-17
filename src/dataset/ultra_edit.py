from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class UltraEditDataset(Dataset):
    """
    UltraEdit Dataset

    Directory Structure
    -------------------
    ultra-edit/
    ├── input_images/
    ├── edited_images/
    ├── masks/
    └── metadata/

    Parameters
    ----------
    root : str | Path
        Path to dataset root.

    split : str
        Present only for API compatibility.

    task : str
        One of:
            "detection"
            "localization"

    max_real : int | None
        Number of pristine images to load.

    max_fake : int | None
        Number of edited images to load.

    seed : int

    transform
        Image transform.

    target_transform
        Mask transform.
    """

    def __init__(self,root,split="train",task="detection",max_real=None,max_fake=None,
                 seed=42,sampling = "random",transform=None,target_transform=None):

        if task not in ("detection", "localization"):
            raise ValueError("task must be either 'detection' or 'localization'.")

        self.root = Path(root)
        self.sampling = sampling
        
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        self.split = split
        self.task = task

        self.max_real = max_real
        self.max_fake = max_fake
        self.seed = seed

        self.transform = transform
        self.target_transform = target_transform


        self.real_dir = self.root / "input_images"
        self.fake_dir = self.root / "edited_images"
        self.mask_dir = self.root / "masks"
        self.mask_lookup = self._build_mask_lookup()

        self.samples = []

        self._build_index()

    # ------------------------------------------------------------
    # Build Dataset Index
    # ------------------------------------------------------------

    def _build_index(self):

        rng = random.Random(self.seed)

        self.samples.extend(
            self._build_real_index(rng)
        )

        self.samples.extend(
            self._build_fake_index(rng)
        )

    # ------------------------------------------------------------
    # Build Real Index
    # ------------------------------------------------------------

    def _build_real_index(self, rng):

        image_paths = sorted( p for p in self.real_dir.iterdir() if p.is_file())

        if self.max_real is not None:
            if self.max_real > len(image_paths):
                raise ValueError(
                    f"Requested {self.max_real} real images "
                    f"but only {len(image_paths)} exist."
                )

            if self.sampling == "random":
                image_paths = sorted(rng.sample(image_paths, self.max_real))

            elif self.sampling == "first":
                image_paths = image_paths[:self.max_real]

            else:
                raise ValueError("sampling must be either 'random' or 'first'")

        samples = []

        for image_path in tqdm(image_paths,desc="Indexing input images",unit="files",):
            samples.append(
                {
                    "image_id": f"real_{image_path.stem}",
                    "file_name": image_path.name,
                    "image_path": image_path,
                    "mask_path": None,
                    "label": 0,
                }
            )

        return samples
    

    # ------------------------------------------------------------
    # Build Fake Index
    # ------------------------------------------------------------

    def _build_fake_index(self, rng):

        image_paths = sorted(p for p in self.fake_dir.iterdir()if p.is_file())
        if self.max_fake is not None:
            if self.max_fake > len(image_paths):
                raise ValueError(
                    f"Requested {self.max_fake} fake images "
                    f"but only {len(image_paths)} exist."
                )

            if self.sampling == "random":
                image_paths = sorted(rng.sample(image_paths, self.max_fake))

            elif self.sampling == "first":
                image_paths = image_paths[:self.max_fake]

            else:
                raise ValueError("sampling must be either 'random' or 'first'")

        samples = []
        self.missing_masks = []
        for image_path in tqdm(image_paths,desc="Indexing edited images",unit="files",):
            mask_path = self.mask_lookup.get(image_path.stem)

            if mask_path is None:
                self.missing_masks.append(
                    {   "image_id": f"fake_{image_path.stem}",
                        "image_path": str(image_path)
                    })
                continue

            samples.append({  
                    "image_id": f"fake_{image_path.stem}",
                    "file_name": image_path.name,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "label": 1
                })
        

        if len(self.missing_masks) > 0:
            print(
                f"\nWarning: {len(self.missing_masks)} manipulated images "
                "do not have corresponding masks."
            )

        return samples

    # ------------------------------------------------------------
    # Build Mask Lookup
    # ------------------------------------------------------------

    def _build_mask_lookup(self):

        mask_lookup = {}

        valid_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }

        mask_files = sorted(
            p
            for p in self.mask_dir.iterdir()
            if p.is_file()
        )

        for mask_path in tqdm(
            mask_files,
            desc="Building mask lookup",
            unit="files",
        ):

            if mask_path.suffix.lower() in valid_extensions:
                mask_lookup[mask_path.stem] = mask_path

        return mask_lookup

    # ------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------

    def __getitem__(self, idx):

        sample = self.samples[idx]

        with Image.open(sample["image_path"]) as img:
            image = img.convert("RGB")

        # --------------------------------------------------------
        # Mask
        # --------------------------------------------------------

        if sample["mask_path"] is None:

            mask = Image.new(
                mode="L",
                size=image.size,
                color=0,
            )

        else:

            with Image.open(sample["mask_path"]) as m:
                mask = m.convert("L")

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
            "image_path": str(sample["image_path"]),
            "mask_path": (
                str(sample["mask_path"])
                if sample["mask_path"] is not None
                else ""
            ),
        }

    # ------------------------------------------------------------

    def __repr__(self):

        num_real = sum(
            s["label"] == 0
            for s in self.samples
        )

        num_fake = sum(
            s["label"] == 1
            for s in self.samples
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"  split={self.split},\n"
            f"  task={self.task},\n"
            f"  real_images={num_real},\n"
            f"  fake_images={num_fake},\n"
            f"  total_images={len(self.samples)}\n"
            f")"
        )