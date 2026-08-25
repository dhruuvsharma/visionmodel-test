import argparse
import os
import random
import re
from collections import defaultdict, Counter

from datasets import load_dataset


PRODUCT_ID_RE = re.compile(r"(.*?id_\d+)", re.IGNORECASE)

VIEW_TOKENS = [
    "front",
    "back",
    "side",
    "left",
    "right",
    "additional"
]


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def safe_folder_name(name):
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def category_is_top(category_value, keywords):
    if category_value is None:
        return False

    c = str(category_value).lower()
    return any(k in c for k in keywords)


def get_identifier(row):
    """
    Some rows may store the useful filename-like ID in item_ID,
    others may store it in text or description.

    We prefer any field that contains something like:
        id_00000123
    """
    for key in ["item_ID", "text", "description"]:
        value = row.get(key)

        if value is None:
            continue

        value_str = str(value)

        if re.search(r"id_\d+", value_str, re.IGNORECASE):
            return value_str

    # Fallback
    return str(row.get("item_ID") or row.get("text") or "")


def derive_product_id(raw_identifier):
    """
    Convert an image-level identifier into a product-level identifier.

    Example:
        MEN_Shirts_id_00000123_01_1_front
        ->
        MEN_Shirts_id_00000123
    """
    if raw_identifier is None:
        return None

    s = str(raw_identifier).strip()

    if not s:
        return None

    s = s.replace("\\", "/")

    match = PRODUCT_ID_RE.search(s)

    if match:
        product_id = match.group(1)
        product_id = product_id.replace("/", "_")
        return safe_folder_name(product_id)

    # Fallback: remove extension and view suffixes
    s = os.path.splitext(s)[0]

    for token in VIEW_TOKENS:
        s = re.sub(
            rf"_{token}.*$",
            "",
            s,
            flags=re.IGNORECASE
        )

    return safe_folder_name(s)


def assign_splits(idxs, rng):
    """
    Split one product's images into:

        view_idxs   -> asset views
        train_idxs  -> training variations
        test_idxs   -> held-out query images

    This works even if a product has only 4-6 images.
    For prototype testing, some view images may be reused for training.
    """
    idxs = list(idxs)
    rng.shuffle(idxs)

    n = len(idxs)

    if n >= 6:
        test_idxs = idxs[-2:]
        pool = idxs[:-2]

        view_idxs = pool[:4]
        train_idxs = pool[4:]

        if len(train_idxs) == 0:
            train_idxs = view_idxs[:2]

    elif n == 5:
        test_idxs = idxs[-1:]
        pool = idxs[:-1]

        view_idxs = pool[:4]
        train_idxs = pool[:2]

    elif n == 4:
        test_idxs = idxs[-1:]
        view_idxs = idxs[:3]
        train_idxs = idxs[:2]

    else:
        test_idxs = idxs[-1:]
        view_idxs = idxs[:-1]
        train_idxs = idxs[:-1]

    return view_idxs, train_idxs, test_idxs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="Marqo/deepfashion-inshop"
    )

    parser.add_argument(
        "--num_items",
        type=int,
        default=300,
        help="Number of shirt products to sample."
    )

    parser.add_argument(
        "--min_images_per_item",
        type=int,
        default=4,
        help="Minimum images required per product."
    )

    parser.add_argument(
        "--max_train_per_item",
        type=int,
        default=15
    )

    parser.add_argument(
        "--category_keywords",
        type=str,
        default="shirt,tee,blouse,top,sweater,cardigan,graphic"
    )

    parser.add_argument(
        "--output_base",
        type=str,
        default="data/hf"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--print_categories",
        action="store_true",
        help="Only print available category2 values and exit."
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------
    print("Loading dataset metadata (first run downloads data)...")

    try:
        full = load_dataset(args.dataset, split="train")
    except Exception:
        loaded = load_dataset(args.dataset)
        split_name = list(loaded.keys())[0]
        full = loaded[split_name]

    if args.print_categories:
        cats = sorted({str(row.get("category2")) for row in full})
        print("Available category2 values:")
        for c in cats:
            print(" -", c)
        return

    # Metadata-only view so we do not decode all images now
    meta = full.remove_columns(["image"])

    keywords = [
        k.strip().lower()
        for k in args.category_keywords.split(",")
        if k.strip()
    ]

    print(f"Using category keywords: {keywords}")

    # -------------------------------------------------------------------
    # Group images by derived product ID
    # -------------------------------------------------------------------
    item_indices = defaultdict(list)

    sample_printed = 0

    for idx, row in enumerate(meta):
        category2 = row.get("category2")

        if not category_is_top(category2, keywords):
            continue

        raw_identifier = get_identifier(row)
        product_id = derive_product_id(raw_identifier)

        if not product_id:
            continue

        if sample_printed < 10:
            print(f"Example: raw={raw_identifier}")
            print(f"         product_id={product_id}")
            sample_printed += 1

        item_indices[product_id].append(idx)

    print(f"Total matching products after grouping: {len(item_indices)}")

    count_dist = Counter(len(v) for v in item_indices.values())

    print("Images-per-product distribution:")

    for count in sorted(count_dist.keys()):
        print(f"  {count} images -> {count_dist[count]} products")

    eligible = [
        (product_id, idxs)
        for product_id, idxs in item_indices.items()
        if len(idxs) >= args.min_images_per_item
    ]

    eligible.sort(key=lambda x: x[0])
    selected = eligible[:args.num_items]

    print(f"Eligible shirt products: {len(eligible)}")
    print(f"Selected shirt products: {len(selected)}")

    if len(selected) == 0:
        raise ValueError(
            "No eligible products found. Check the example raw IDs printed above. "
            "If they do not contain something like id_00000123, share a few sample values."
        )

    train_root = os.path.join(args.output_base, "synthetic_shirts")
    asset_root = os.path.join(args.output_base, "asset_views")
    test_root = os.path.join(args.output_base, "test_shirts")

    rng = random.Random(args.seed)

    total_train = 0
    total_views = 0
    total_test = 0

    # -------------------------------------------------------------------
    # Save images to disk in your prototype folder structure
    # -------------------------------------------------------------------
    for product_id, idxs in selected:
        folder = safe_folder_name(product_id)

        view_idxs, train_idxs, test_idxs = assign_splits(idxs, rng)

        train_idxs = train_idxs[:args.max_train_per_item]

        for i, idx in enumerate(view_idxs):
            try:
                img = full[idx]["image"].convert("RGB")
            except Exception as e:
                print(f"Warning: failed view image {product_id}: {e}")
                continue

            path = os.path.join(asset_root, folder, f"view_{i}.jpg")
            ensure_dir(os.path.dirname(path))
            img.save(path, quality=90)
            total_views += 1

        for i, idx in enumerate(train_idxs):
            try:
                img = full[idx]["image"].convert("RGB")
            except Exception as e:
                print(f"Warning: failed train image {product_id}: {e}")
                continue

            path = os.path.join(train_root, folder, f"var_{i:02d}.jpg")
            ensure_dir(os.path.dirname(path))
            img.save(path, quality=90)
            total_train += 1

        for i, idx in enumerate(test_idxs):
            try:
                img = full[idx]["image"].convert("RGB")
            except Exception as e:
                print(f"Warning: failed test image {product_id}: {e}")
                continue

            path = os.path.join(test_root, folder, f"query_{i:02d}.jpg")
            ensure_dir(os.path.dirname(path))
            img.save(path, quality=90)
            total_test += 1

    print("Dataset preparation complete.")
    print(f"Asset view images: {total_views}")
    print(f"Training images: {total_train}")
    print(f"Test query images: {total_test}")

    print("\nNext commands:")
    print(
        "python scripts/make_manifest.py "
        f"--train_root {train_root} "
        f"--asset_root {asset_root} "
        f"--test_root {test_root} "
        "--output_dir data/records --val_frac 0.1 --test_frac 0.0"
    )
    print("python src/train.py --config configs/config.json")
    print("python src/index.py --config configs/config.json")
    print("python src/evaluate.py --config configs/config.json")


if __name__ == "__main__":
    main()