from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import hashlib
import json
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# 1. CONFIGURATION

PROJECT_ROOT = Path(__file__).resolve().parent

ZIP_PATH = PROJECT_ROOT / "data/raw/UTKFace.zip"
METADATA_DIR = PROJECT_ROOT / "data/metadata"
OUTPUT_DIR = PROJECT_ROOT / "data/preprocessing"

IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# Age groups (option 2): 0-12, 13-19, 20-29, 30-39, 40-49, 50-59, 60+
AGE_GROUP_EDGES = [13, 20, 30, 40, 50, 60]
NUM_AGE_GROUPS = len(AGE_GROUP_EDGES) + 1
AGE_GROUP_NAMES = ["0-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60+"]

SPLIT_NAMES = ("train", "val", "test")
COLUMNS = ["member", "age", "gender", "race"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 2. LOAD AND VALIDATE SPLITS FROM FILE 1

def load_splits(metadata_dir=METADATA_DIR):
    metadata_dir = Path(metadata_dir)

    return {
        split: pd.read_csv(
            metadata_dir / f"{split}.csv",
            usecols=list(COLUMNS),
            dtype={
                "member": "string",
                "age": "float32",
                "gender": "int64"
            },
            encoding="utf-8-sig"
        )
        for split in SPLIT_NAMES
    }

def compute_age_groups(ages):
    return np.digitize(np.asarray(ages, dtype="float64"), AGE_GROUP_EDGES)


def compute_group_weights(train_data):
    """Weights from the TRAIN split only. 1/sqrt(count), normalized so the
    sample-weighted mean is 1 (keeps the loss scale unchanged)."""
    groups = compute_age_groups(train_data["age"])
    counts = np.bincount(groups, minlength=NUM_AGE_GROUPS).astype("float64")
    weights = 1.0 / np.sqrt(np.clip(counts, 1, None))
    weights = weights / ((weights * counts).sum() / counts.sum())
    return torch.tensor(weights, dtype=torch.float32)

# 3. FIND IMAGES IN ZIP

def create_zip_index(zip_path, members):
    zip_path = Path(zip_path)

    if not zip_path.is_file():
        raise FileNotFoundError(f"UTKFace ZIP not found: {zip_path}")

    requested_members = {
        str(member).strip().replace("\\", "/")
        for member in members
    }

    with ZipFile(zip_path) as archive:
        zip_members = set(archive.namelist())

    missing = requested_members - zip_members

    if missing:
        raise FileNotFoundError(
            f"ZIP is missing {len(missing)} images. Example: {sorted(missing)[:3]}"
        )

    return {member: member for member in requested_members}

# 4. RESIZE, AUGMENTATION AND NORMALIZATION

def create_transform(model_type, training, image_size=IMAGE_SIZE):
    if model_type not in ["scratch", "resnet18"]:
        raise ValueError("model_type must be either scratch or resnet18.")

    steps = [
        transforms.Resize((image_size, image_size))
    ]

    if training:
        steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1
            )
        ])

    steps.append(transforms.ToTensor())

    if model_type == "resnet18":
        steps.append(
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        )

    return transforms.Compose(steps)

# 5. DATASET: READ AN IMAGE AND ITS LABELS

class UTKFaceDataset(Dataset):
    def __init__(self, data, zip_path, index, transform):
        self.data = data.reset_index(drop=True)
        self.age_groups = compute_age_groups(self.data["age"])
        self.group_weights = None  # set in build_loaders for the train split
        self.zip_path = Path(zip_path)
        self.index = index
        self.transform = transform
        self.archive = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, position):
        row = self.data.iloc[position]

        if self.archive is None:
            self.archive = ZipFile(self.zip_path)

        content = self.archive.read(self.index[row["member"]])

        with Image.open(BytesIO(content)) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "image": image,
            "age": torch.tensor(
                [float(row["age"])],
                dtype=torch.float32
            ),
            "gender": torch.tensor(
                int(row["gender"]),
                dtype=torch.long
            ),
            "age_group": torch.tensor(
                int(self.age_groups[position]),
                dtype=torch.long
            ),
            "member": row["member"]
        }

    def close(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None

# 6. DATALOADER: GROUP IMAGES INTO BATCHES

def build_loaders(
    model_type="scratch",
    metadata_dir=METADATA_DIR,
    zip_path=ZIP_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
):
    if image_size < 32 or batch_size < 1:
        raise ValueError("image_size >= 32 and batch_size >= 1.")

    splits = load_splits(metadata_dir)

    members = pd.concat(splits.values())["member"]
    index = create_zip_index(zip_path, members)

    loaders = {}
    group_weights = compute_group_weights(splits["train"])

    generator = torch.Generator()
    generator.manual_seed(seed)

    for name, data in splits.items():
        transform = create_transform(
            model_type=model_type,
            training=(name == "train"),
            image_size=image_size
        )

        dataset = UTKFaceDataset(
            data=data,
            zip_path=zip_path,
            index=index,
            transform=transform
        )

        dataset.group_weights = group_weights

        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=False,
            num_workers=2,
            generator=generator if name == "train" else None
        )

    return loaders

# 7. CHECK BATCHES AND AUGMENTATION

def check_batches(loaders, model_type, image_size=IMAGE_SIZE):
    for name, loader in loaders.items():
        batch = next(iter(loader))
        size = len(batch["member"])

        assert batch["image"].shape == (size, 3, image_size, image_size)
        assert batch["age"].shape == (size, 1)
        assert batch["gender"].shape == (size,)
        assert batch["age_group"].shape == (size,)
        assert batch["age_group"].ge(0).all()
        assert batch["age_group"].lt(NUM_AGE_GROUPS).all()

        assert batch["image"].dtype == torch.float32
        assert batch["age"].dtype == torch.float32
        assert batch["gender"].dtype == torch.int64

        assert torch.isfinite(batch["image"]).all()
        assert batch["age"].ge(1).all() and batch["age"].le(116).all()
        assert ((batch["gender"] == 0) | (batch["gender"] == 1)).all()

        if model_type == "scratch":
            assert batch["image"].min() >= 0
            assert batch["image"].max() <= 1

        print(model_type, name, tuple(batch["image"].shape))

    for name in ["val", "test"]:
        dataset = loaders[name].dataset

        assert torch.equal(
            dataset[0]["image"],
            dataset[0]["image"]
        )


def save_augmentation_preview(loader, model_type, output_path):
    dataset = loader.dataset

    figure, axes = plt.subplots(1, 4, figsize=(12, 3))

    for axis in axes:
        image = dataset[0]["image"].permute(1, 2, 0)

        if model_type == "resnet18":
            image = (
                image * torch.tensor(IMAGENET_STD)
                + torch.tensor(IMAGENET_MEAN)
            )

        axis.imshow(image.clamp(0, 1).numpy())
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

# 8. SAVE CONFIGURATION FOR INFERENCE TO MATCH TRAINING

def save_config(output_dir):
    config = {
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "color": "RGB",
        "batch_size": BATCH_SIZE,
        "gender_encoding": {"0": "Male", "1": "Female"},
        "age_unit": "years",
        "scratch": {"pixel_range": [0, 1]},
        "resnet18": {
            "pixel_range_before_normalize": [0, 1],
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD
        },
        "train_augmentation": {
            "horizontal_flip_probability": 0.5,
            "brightness": 0.1,
            "contrast": 0.1
        },
        "validation_test_augmentation": False,
        "age_group_edges": AGE_GROUP_EDGES,
        "age_group_names": AGE_GROUP_NAMES,
        "identity_disjoint_verified": False,
        "split_sha256": {
            name: hashlib.sha256(
                (METADATA_DIR / f"{name}.csv").read_bytes()
            ).hexdigest()
            for name in ["train", "val", "test"]
        }
    }

    path = output_dir / "preprocessing_config.json"
    path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )

# 9. RUN FULL PREPROCESSING

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(prefix="run_", dir=OUTPUT_DIR)
    )

    save_config(run_dir)

    for model_type in ["scratch", "resnet18"]:
        loaders = build_loaders(model_type=model_type)

        try:
            check_batches(loaders, model_type)

            save_augmentation_preview(
                loaders["train"],
                model_type,
                run_dir / f"augmentation_{model_type}.png"
            )
        finally:
            for loader in loaders.values():
                loader.dataset.close()

    print("Preprocessing completed.")
    print("Output:", run_dir)
    print("Create new DataLoader before training each model.")


if __name__ == "__main__":
    main()