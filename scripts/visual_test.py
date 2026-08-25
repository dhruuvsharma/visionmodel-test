import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from utils import load_json, get_device, ensure_dir
from transforms import get_val_transform
from index import load_model_from_checkpoint, encode_image
from retrieve import load_or_build_index
from search import search_top_k


def safe_name(value):
    return "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in str(value)
    )


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
        "--num_queries",
        type=int,
        default=10
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/visual_test"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
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

    asset_views_path = config["data"].get("asset_views")

    if not asset_views_path or not os.path.exists(asset_views_path):
        raise FileNotFoundError(
            f"Asset views file not found: {asset_views_path}"
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

    index_embeddings, index_asset_ids = load_or_build_index(
        config=config,
        model=model,
        transform=transform,
        device=device
    )

    rng = random.Random(args.seed)

    if len(test_records) > args.num_queries:
        selected_records = rng.sample(test_records, args.num_queries)
    else:
        selected_records = test_records

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)

    ensure_dir(run_dir)

    html_parts = []

    html_parts.append("<html>")
    html_parts.append("<head>")
    html_parts.append("<title>Shirt Retrieval Visual Test</title>")
    html_parts.append("<style>")
    html_parts.append("body { font-family: Arial; margin: 20px; }")
    html_parts.append(".query-block { border: 1px solid #ccc; padding: 15px; margin-bottom: 30px; }")
    html_parts.append(".images { display: flex; gap: 15px; flex-wrap: wrap; }")
    html_parts.append(".image-card { text-align: center; width: 180px; }")
    html_parts.append(".image-card img { width: 160px; height: 200px; object-fit: contain; border: 1px solid #ddd; }")
    html_parts.append("</style>")
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append("<h1>Shirt Retrieval Visual Test</h1>")

    results_log = []

    for record_index, record in enumerate(selected_records):
        query_image_path = record.get("image_path") or record.get("path")
        correct_asset_id = record.get("asset_id", "unknown")

        if not query_image_path or not os.path.exists(query_image_path):
            print(f"Warning: query image not found: {query_image_path}")
            continue

        query_embedding = encode_image(
            model=model,
            image_path=query_image_path,
            transform=transform,
            device=device
        )

        results = search_top_k(
            query_embedding=query_embedding,
            index_embeddings=index_embeddings,
            index_asset_ids=index_asset_ids,
            top_k=args.top_k
        )

        query_ext = os.path.splitext(query_image_path)[1] or ".jpg"
        query_copy_name = f"query_{record_index:03d}{query_ext}"
        query_copy_path = os.path.join(run_dir, query_copy_name)

        shutil.copy(query_image_path, query_copy_path)

        html_parts.append("<div class='query-block'>")
        html_parts.append(
            f"<h2>Query {record_index + 1} | Expected asset: {correct_asset_id}</h2>"
        )

        html_parts.append("<div class='images'>")

        html_parts.append("<div class='image-card'>")
        html_parts.append(f"<img src='{query_copy_name}'>")
        html_parts.append("<p><b>Query</b></p>")
        html_parts.append("</div>")

        query_result_log = {
            "query_image": query_image_path,
            "expected_asset_id": correct_asset_id,
            "results": []
        }

        for rank, item in enumerate(results, start=1):
            asset_id = item["asset_id"]
            score = item["score"]

            asset_image_paths = asset_views.get(asset_id, [])

            if len(asset_image_paths) == 0:
                continue

            asset_image_path = asset_image_paths[0]

            if not os.path.exists(asset_image_path):
                continue

            asset_ext = os.path.splitext(asset_image_path)[1] or ".jpg"
            asset_copy_name = (
                f"query_{record_index:03d}_rank_{rank:02d}_"
                f"{safe_name(asset_id)}{asset_ext}"
            )

            asset_copy_path = os.path.join(run_dir, asset_copy_name)

            shutil.copy(asset_image_path, asset_copy_path)

            html_parts.append("<div class='image-card'>")
            html_parts.append(f"<img src='{asset_copy_name}'>")
            html_parts.append(f"<p>{rank}. {asset_id}<br>score={score:.4f}</p>")
            html_parts.append("</div>")

            query_result_log["results"].append({
                "rank": rank,
                "asset_id": asset_id,
                "score": score
            })

        html_parts.append("</div>")
        html_parts.append("</div>")

        results_log.append(query_result_log)

    html_parts.append("</body>")
    html_parts.append("</html>")

    html_path = os.path.join(run_dir, "index.html")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    json_path = os.path.join(run_dir, "results.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)

    print("Visual test complete.")
    print(f"HTML report: {html_path}")
    print(f"JSON results: {json_path}")


if __name__ == "__main__":
    main()