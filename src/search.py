import numpy as np


def l2_normalize(x, eps=1e-12):
    """
    L2 normalize one or more vectors.
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def rank_assets(query_embedding, index_embeddings, index_asset_ids):
    """
    Rank assets by maximum cosine similarity over asset view embeddings.

    Returns:
        [
            (asset_id, score),
            ...
        ]
    """
    query_embedding = l2_normalize(query_embedding)[0]
    index_embeddings = l2_normalize(index_embeddings)

    scores = index_embeddings @ query_embedding

    best_scores = {}

    for score, asset_id in zip(scores, index_asset_ids):
        score = float(score)

        if asset_id not in best_scores:
            best_scores[asset_id] = score
        else:
            if score > best_scores[asset_id]:
                best_scores[asset_id] = score

    ranked = sorted(
        best_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked


def search_top_k(
    query_embedding,
    index_embeddings,
    index_asset_ids,
    top_k=10
):
    """
    Return top-K assets.
    """
    ranked = rank_assets(
        query_embedding=query_embedding,
        index_embeddings=index_embeddings,
        index_asset_ids=index_asset_ids
    )

    results = []

    for asset_id, score in ranked[:top_k]:
        results.append({
            "asset_id": asset_id,
            "score": float(score)
        })

    return results


def find_rank(
    query_embedding,
    index_embeddings,
    index_asset_ids,
    correct_asset_id
):
    """
    Find rank of the correct asset.

    If not found, return len(ranked_assets) + 1.
    """
    ranked = rank_assets(
        query_embedding=query_embedding,
        index_embeddings=index_embeddings,
        index_asset_ids=index_asset_ids
    )

    for rank, item in enumerate(ranked, start=1):
        asset_id, _ = item

        if asset_id == correct_asset_id:
            return rank

    return len(ranked) + 1