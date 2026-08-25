import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_json,
    save_json,
    get_device
)

from logger import log_metrics

from transforms import get_val_transform

from index import (
    load_model_from_checkpoint,
    encode_image
)

from retrieve import load_or_build_index

from search import find_rank


def evaluate_records(
    model,
    records,
    index_embeddings,
    index_asset_ids,
    transform,
    device,
    top_k_values=(1, 5, 10)
):
    hits = {
        k: 0
        for k in top_k_values
    }

    ranks = []
    total = 0

    for record in records:
        image_path = record.get("image_path") or record.get("path")
        correct_asset_id = record.get("asset_id")

        if not image_path or not correct_asset_id:
            continue

        if not os.path.exists(image_path):
            print(f"Warning: image not found: {image_path}")
            continue

        query_embedding = encode_image(
            model=model,
            image_path=image_path,
            transform=transform,
            device=device
        )

        rank = find_rank(
            query_embedding=query_embedding,
            index_embeddings=index_embeddings,
            index_asset_ids=index_asset_ids,
            correct_asset_id=correct_asset_id
        )

        ranks.append(rank)
        total += 1

        for k in top_k_values:
            if rank <= k:
                hits[k] += 1

    metrics = {}

    for k in top_k_values:
        metrics[f"recall_at_{k}"] = hits[k] / max(1, total)

    metrics["mean_rank"] = sum(ranks) / max(1, total)
    metrics["num_queries"] = total

    return metrics


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

    test_records_path = config["data"].get("test_records")

    if not test_records_path or not os.path.exists(test_records_path):
        raise FileNotFoundError(
            f"Test records not found: {test_records_path}. "
            "Run scripts/make_manifest.py first."
        )

    test_records = load_json(test_records_path)

    if len(test_records) == 0:
        raise ValueError("Test records are empty.")

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

    metrics = evaluate_records(
        model=model,
        records=test_records,
        index_embeddings=index_embeddings,
        index_asset_ids=index_asset_ids,
        transform=transform,
        device=device,
        top_k_values=(1, 5, 10)
    )

    evaluation_path = config["output"].get(
        "evaluation_metrics_path",
        "outputs/evaluation_metrics.json"
    )

    save_json(metrics, evaluation_path)

    log_metrics(
        log_path=config["output"].get("log_path", "logs/experiments.jsonl"),
        metrics={
            "event": "evaluation",
            **metrics
        }
    )

    print("Evaluation metrics:")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"Saved metrics to: {evaluation_path}")


if __name__ == "__main__":
    main()