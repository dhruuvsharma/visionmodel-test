import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=0.07
):
    """
    Simple supervised contrastive loss.

    Positive pairs:
        embeddings with the same label

    Negative pairs:
        embeddings with different labels
    """
    device = embeddings.device

    embeddings = F.normalize(embeddings, dim=1)

    batch_size = embeddings.size(0)

    if batch_size <= 1:
        return embeddings.sum() * 0.0

    labels = labels.contiguous().view(-1, 1)

    # Mask where same labels are positive pairs
    mask = torch.eq(labels, labels.T).float().to(device)

    # Similarity matrix
    similarity = torch.matmul(embeddings, embeddings.T) / temperature

    # Remove self-similarity
    self_mask = 1.0 - torch.eye(batch_size, device=device)
    mask = mask * self_mask

    positives = mask.sum(dim=1)
    valid = positives > 0

    if not valid.any():
        return embeddings.sum() * 0.0

    # Numerical stability
    logits_max, _ = similarity.max(dim=1, keepdim=True)
    logits = similarity - logits_max.detach()

    exp_logits = torch.exp(logits) * self_mask
    denom = exp_logits.sum(dim=1).clamp(min=1e-12)

    log_prob = logits - torch.log(denom)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1)
    mean_log_prob_pos = mean_log_prob_pos / positives.clamp(min=1.0)

    loss = -mean_log_prob_pos[valid].mean()

    return loss