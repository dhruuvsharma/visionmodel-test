import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import gradio as gr

from utils import load_json, get_device
from transforms import get_val_transform
from index import load_model_from_checkpoint, encode_image
from retrieve import load_or_build_index
from search import search_top_k


CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/config.json")

config = load_json(CONFIG_PATH)

device = get_device(
    config["training"].get("device", "auto")
)

checkpoint_path = config["output"].get(
    "best_checkpoint",
    "checkpoints/best.pt"
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

asset_views_path = config["data"].get("asset_views")
asset_views = load_json(asset_views_path)


def retrieve(image_path, top_k=5):
    if not image_path:
        return []

    query_embedding = encode_image(
        model=model,
        image_path=image_path,
        transform=transform,
        device=device
    )

    results = search_top_k(
        query_embedding=query_embedding,
        index_embeddings=index_embeddings,
        index_asset_ids=index_asset_ids,
        top_k=int(top_k)
    )

    gallery = []

    for item in results:
        asset_id = item["asset_id"]
        score = item["score"]

        paths = asset_views.get(asset_id, [])

        if len(paths) == 0:
            continue

        image_path_result = paths[0]

        if not os.path.exists(image_path_result):
            continue

        caption = f"{asset_id} ({score:.3f})"

        gallery.append((image_path_result, caption))

    return gallery


demo = gr.Interface(
    fn=retrieve,
    inputs=[
        gr.Image(type="filepath", label="Query shirt crop"),
        gr.Slider(
            minimum=1,
            maximum=20,
            value=5,
            step=1,
            label="Top K"
        )
    ],
    outputs=gr.Gallery(label="Top Matching Shirts"),
    title="Shirt Retrieval Prototype",
    description="Upload a shirt crop and retrieve the closest shirt assets."
)

demo.launch()