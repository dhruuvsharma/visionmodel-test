import json
import os
import random

import numpy as np
import torch


def ensure_dir(path):
    """
    Create directory if it does not exist.
    If path is empty, do nothing.
    """
    if not path:
        return

    os.makedirs(path, exist_ok=True)


def load_json(path):
    """
    Load JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    """
    Save object as JSON.
    """
    dirname = os.path.dirname(path)
    ensure_dir(dirname)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_device(name="auto"):
    """
    Return torch device.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(name)


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """
    Simple meter for tracking average values.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.sum / self.count