import argparse
import json
import os
import random


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def scan_classification_root(root):
    """
    Expected structure:

    root/
        shirt_001/
            img1.png
            img2.png
        shirt_002/
            img1.png
            img2.png
    """
    records = []

    if not root or not os.path.isdir(root):
        return records

    for asset_id in sorted(os.listdir(root)):
        asset_dir = os.path.join(root, asset_id)

        if not os.path.isdir(asset_dir):
            continue

        for filename in sorted(os.listdir(asset_dir)):
            if not is_image_file(filename):
                continue

            image_path = os.path.abspath(
                os.path.join(asset_dir, filename)
            )

            records.append({
                "image_path": image_path,
                "asset_id": asset_id
            })

    return records


def scan_asset_views_root(root):
    """
    Expected structure:

    root/
        shirt_001/
            front.png
            left.png
            right.png
            back.png
        shirt_002/
            front.png
            left.png
            right.png
            back.png
    """
    asset_views = {}

    if not root or not os.path.isdir(root):
        return asset_views

    for asset_id in sorted(os.listdir(root)):
        asset_dir = os.path.join(root, asset_id)

        if not os.path.isdir(asset_dir):
            continue

        image_paths = []

        for filename in sorted(os.listdir(asset_dir)):
            if not is_image_file(filename):
                continue

            image_path = os.path.abspath(
                os.path.join(asset_dir, filename)
            )

            image_paths.append(image_path)

        if len(image_paths) > 0:
            asset_views[asset_id] = image_paths

    return asset_views


def asset_views_to_records(asset_views):
    records = []

    for asset_id, image_paths in asset_views.items():
        for image_path in image_paths:
            records.append({
                "image_path": image_path,
                "asset_id": asset_id
            })

    return records


def split_records(records, val_frac=0.1, test_frac=0.1, seed=42):
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


def build_label_map(records):
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_root",
        type=str,
        default=None,
        help="Folder containing synthetic or training shirt images. Each subfolder is one asset_id."
    )

    parser.add_argument(
        "--asset_root",
        type=str,
        default=None,
        help="Folder containing official asset views. Each subfolder is one asset_id."
    )

    parser.add_argument(
        "--test_root",
        type=str,
        default=None,
        help="Optional folder containing test query images. Each subfolder is one asset_id."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/records"
    )

    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    train_records = []

    if args.train_root:
        train_records = scan_classification_root(args.train_root)

    asset_views = {}

    if args.asset_root:
        asset_views = scan_asset_views_root(args.asset_root)

    # If no training root is given, use asset views as training records.
    if len(train_records) == 0 and len(asset_views) > 0:
        print("No train_root provided. Using asset views as training records.")
        train_records = asset_views_to_records(asset_views)

    if len(train_records) == 0 and len(asset_views) == 0:
        raise ValueError(
            "No data found. Provide --train_root or --asset_root."
        )

    # -------------------------------------------------------------------
    # Test records
    # -------------------------------------------------------------------
    test_records = []

    if args.test_root:
        test_records = scan_classification_root(args.test_root)

    # -------------------------------------------------------------------
    # Split train records if no explicit test root is given
    # -------------------------------------------------------------------
    if len(test_records) == 0:
        train_records, val_records, test_records = split_records(
            train_records,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed
        )
    else:
        train_records, val_records, _ = split_records(
            train_records,
            val_frac=args.val_frac,
            test_frac=0.0,
            seed=args.seed
        )

    # -------------------------------------------------------------------
    # Label map
    # -------------------------------------------------------------------
    if len(train_records) > 0:
        label_map = build_label_map(train_records)
    else:
        asset_ids = sorted(asset_views.keys())
        label_map = {
            asset_id: idx
            for idx, asset_id in enumerate(asset_ids)
        }

    # -------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------
    train_records_path = os.path.join(
        args.output_dir,
        "train_records.json"
    )

    val_records_path = os.path.join(
        args.output_dir,
        "val_records.json"
    )

    test_records_path = os.path.join(
        args.output_dir,
        "test_records.json"
    )

    asset_views_path = os.path.join(
        args.output_dir,
        "asset_views.json"
    )

    label_map_path = os.path.join(
        args.output_dir,
        "label_map.json"
    )

    save_json(train_records, train_records_path)
    save_json(val_records, val_records_path)
    save_json(test_records, test_records_path)
    save_json(asset_views, asset_views_path)
    save_json(label_map, label_map_path)

    print("Manifest generation complete.")
    print(f"Train records: {len(train_records)}")
    print(f"Val records: {len(val_records)}")
    print(f"Test records: {len(test_records)}")
    print(f"Asset view groups: {len(asset_views)}")
    print(f"Number of classes: {len(label_map)}")

    print(f"Saved: {train_records_path}")
    print(f"Saved: {val_records_path}")
    print(f"Saved: {test_records_path}")
    print(f"Saved: {asset_views_path}")
    print(f"Saved: {label_map_path}")


if __name__ == "__main__":
    main()