import os
import random

from PIL import Image
from torch.utils.data import Dataset

from utils import load_json, save_json


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def is_image_path(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in IMAGE_EXTENSIONS


def load_records(path):
    """
    Load records JSON.

    Expected format:

    [
        {
            "image_path": "path/to/image.png",
            "asset_id": "shirt_001"
        },
        ...
    ]
    """
    return load_json(path)


def build_label_map(records):
    """
    Build mapping from asset_id to integer class index.
    """
    asset_ids = sorted({
        record["asset_id"]
        for record in records
        if "asset_id" in record
    })

    label_map = {
        asset_id: idx
        for idx, asset_id in enumerate(asset_ids)
    }

    return label_map


def save_label_map(label_map, path):
    save_json(label_map, path)


def filter_records_by_label_map(records, label_map):
    """
    Keep only records whose asset_id exists in label_map.
    """
    filtered = []

    for record in records:
        asset_id = record.get("asset_id")

        if asset_id in label_map:
            filtered.append(record)

    return filtered


def split_records(records, val_frac=0.1, test_frac=0.1, seed=42):
    """
    Split records into train, validation, and test records.
    """
    records = list(records)

    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)

    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_records = records[:n_test]
    val_records = records[n_test:n_test + n_val]
    train_records = records[n_test + n_val:]

    return train_records, val_records, test_records


class ShirtDataset(Dataset):
    """
    Dataset for shirt images.

    Each record should contain:

        image_path
        asset_id
    """

    def __init__(
        self,
        records,
        label_to_id=None,
        transform=None,
        return_asset_id=False
    ):
        self.records = records
        self.label_to_id = label_to_id
        self.transform = transform
        self.return_asset_id = return_asset_id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        image_path = record.get("image_path") or record.get("path")
        asset_id = record.get("asset_id")

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.label_to_id is not None:
            label = self.label_to_id[asset_id]

            if self.return_asset_id:
                return image, label, asset_id

            return image, label

        return image, asset_id