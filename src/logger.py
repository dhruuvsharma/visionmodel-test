import json
import os
from datetime import datetime

from utils import ensure_dir


def log_metrics(log_path, metrics):
    """
    Append metrics to a JSON Lines file.

    Example:
        logs/experiments.jsonl
    """
    dirname = os.path.dirname(log_path)
    ensure_dir(dirname)

    record = {
        "timestamp": datetime.now().isoformat(),
        **metrics
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")