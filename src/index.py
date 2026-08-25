import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from PIL import Image

from utils import (
    load_json,
    save_json,
    ensure_dir,
    get_device
)

from transforms import get_val_transform
from model import ShirtEncoder


def load_model_from_checkpoint(config, checkpoint_path, device):
    """
    Load trained ShirtEncoder from checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    label_to_id = ckpt.get("label_to_id", None)

    if label_to_id is None:
        label_map_path = config["output"].get("label_map_path")

        if label_map_path and os.path.exists(label_map_path):
            label_to_id = load_json(label_map_path)

    if label_to_id is None:
        label_to_id = {}

    num_classes = max(1, len(label_to_id))

    model = ShirtEncoder(
        num_classes=num_classes,
        embedding_dim=config["model"].get("embedding_dim", 128),
        backbone_name=config["model"].get("backbone", "resnet18")
    )

    state_dict = ckpt.get("model", ckpt)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, label_to_id


def encode_image(model, image_path, transform, device):
    """
    Encode one image into an embedding.
    """
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        embedding, _ = model(image)

    embedding = embedding.squeeze(0)
    embedding = embedding.cpu().numpy()

    return embedding


def build_index_from_asset_views(
    model,
    asset_views,
    transform,
    device
):
    """
    Build embedding index from asset views.

    asset_views format:

    {
        "shirt_001": [
            "data/asset_views/shirt_001/front.png",
            "data/asset_views/shirt_001/left.png",
            "data/asset_views/shirt_001/right.png",
            "data/asset_views/shirt_001/back.png"
        ],
        ...
    }
    """
    embeddings = []
    asset_ids = []

    for asset_id, image_paths in asset_views.items():
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"Warning: image not found: {image_path}")
                continue

            try:
                embedding = encode_image(
                    model=model,
                    image_path=image_path,
                    transform=transform,
                    device=device
                )

                embeddings.append(embedding)
                asset_ids.append(asset_id)

            except Exception as e:
                print(f"Warning: failed to encode {image_path}: {e}")

    if len(embeddings) == 0:
        raise ValueError("No asset images were successfully encoded.")

    embeddings = np.array(embeddings, dtype=np.float32)

    return embeddings, asset_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None
    )
    args = parser.parse_args()

    config = load_json(args.config)

    device = get_device(
        config["training"].get("device", "auto")
    )

    checkpoint_path = args.checkpoint

    if checkpoint_path is None:
        checkpoint_path = config["output"].get(
            "best_checkpoint",
            "checkpoints/best.pt"
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the model first."
        )

    asset_views_path = config["data"].get("asset_views")

    if not asset_views_path or not os.path.exists(asset_views_path):
        raise FileNotFoundError(
            f"Asset views file not found: {asset_views_path}. "
            "Run scripts/make_manifest.py first."
        )

    asset_views = load_json(asset_views_path)

    model, _ = load_model_from_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device
    )

    transform = get_val_transform(
        image_size=config["data"].get("image_size", 128)
    )

    embeddings, asset_ids = build_index_from_asset_views(
        model=model,
        asset_views=asset_views,
        transform=transform,
        device=device
    )

    embeddings_path = config["output"].get(
        "index_embeddings_path",
        "outputs/index_embeddings.npy"
    )

    asset_ids_path = config["output"].get(
        "index_asset_ids_path",
        "outputs/index_asset_ids.json"
    )

    ensure_dir(os.path.dirname(embeddings_path))
    ensure_dir(os.path.dirname(asset_ids_path))

    np.save(embeddings_path, embeddings)
    save_json(asset_ids, asset_ids_path)

    print(f"Index embeddings saved to: {embeddings_path}")
    print(f"Index asset IDs saved to: {asset_ids_path}")
    print(f"Total indexed vectors: {len(asset_ids)}")


if __name__ == "__main__":
    main()