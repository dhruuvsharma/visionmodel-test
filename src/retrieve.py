import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from utils import (
    load_json,
    save_json,
    get_device
)

from transforms import get_val_transform

from index import (
    load_model_from_checkpoint,
    encode_image,
    build_index_from_asset_views
)

from search import search_top_k


def load_or_build_index(config, model, transform, device):
    """
    Load index if available.
    Otherwise build it from asset views.
    """
    embeddings_path = config["output"].get(
        "index_embeddings_path",
        "outputs/index_embeddings.npy"
    )

    asset_ids_path = config["output"].get(
        "index_asset_ids_path",
        "outputs/index_asset_ids.json"
    )

    if os.path.exists(embeddings_path) and os.path.exists(asset_ids_path):
        index_embeddings = np.load(embeddings_path)
        index_asset_ids = load_json(asset_ids_path)

        return index_embeddings, index_asset_ids

    asset_views_path = config["data"].get("asset_views")

    if not asset_views_path or not os.path.exists(asset_views_path):
        raise FileNotFoundError(
            f"Asset views file not found: {asset_views_path}. "
            "Run scripts/make_manifest.py first or build index first."
        )

    asset_views = load_json(asset_views_path)

    index_embeddings, index_asset_ids = build_index_from_asset_views(
        model=model,
        asset_views=asset_views,
        transform=transform,
        device=device
    )

    np.save(embeddings_path, index_embeddings)
    save_json(index_asset_ids, asset_ids_path)

    return index_embeddings, index_asset_ids


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

    parser.add_argument(
        "--query",
        type=str,
        required=True
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10
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

    if not os.path.exists(args.query):
        raise FileNotFoundError(
            f"Query image not found: {args.query}"
        )

    model, _ = load_model_from_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device
    )

    transform = get_val_transform(
        image_size=config["data"].get("image_size", 128)
    )

    index_embeddings, index_asset_ids = load_or_build_index(
        config=config,
        model=model,
        transform=transform,
        device=device
    )

    query_embedding = encode_image(
        model=model,
        image_path=args.query,
        transform=transform,
        device=device
    )

    results = search_top_k(
        query_embedding=query_embedding,
        index_embeddings=index_embeddings,
        index_asset_ids=index_asset_ids,
        top_k=args.top_k
    )

    print("Top results:")

    for rank, item in enumerate(results, start=1):
        asset_id = item["asset_id"]
        score = item["score"]

        print(f"{rank}. {asset_id}  score={score:.4f}")


if __name__ == "__main__":
    main()