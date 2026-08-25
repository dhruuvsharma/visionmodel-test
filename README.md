# Shirt Retrieval Prototype

This is a shirt-only visual retrieval prototype.

It trains a small image encoder from scratch and retrieves the closest matching shirt assets using cosine similarity.

This prototype does not require:

- FAISS
- MLflow
- Albumentations
- pretrained weights

---

## Expected data layout

### Training images

Synthetic shirt images should be arranged like this:

```text
data/synthetic_shirts/
    shirt_001/
        var_0001.png
        var_0002.png
        var_0003.png
    shirt_002/
        var_0001.png
        var_0002.png