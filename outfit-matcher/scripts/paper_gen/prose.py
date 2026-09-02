"""Long-form prose for the whitepaper (imported by build_pdf.py)."""

ABSTRACT_TEXT = (
    "We present Outfit Matcher, a garment-retrieval system that maps a photograph of a dressed "
    "person to the exact matching garments in a 3D render catalog. The visual encoder is a Vision "
    "Transformer (ViT-S/16, 21.9M parameters) implemented and trained entirely from scratch - no "
    "pretrained weights are used at any stage. Training follows the supervised contrastive "
    "(SupCon) paradigm with multi-view positives: the four canonical render angles of one garment "
    "are treated as positive pairs, which pulls all views of a garment onto a single point of the "
    "256-dimensional unit hypersphere while pushing distinct garments apart. A key challenge is "
    "the domain gap between clean catalog renders (T-posed garment on white) and real photographs; "
    "we address it with heavy domain randomization - background keying via border flood fill, "
    "procedural background composites, perspective and affine warps, occlusion, blur, and noise. "
    "At inference, catalog views are embedded once and a query photo is matched by cosine "
    "similarity with per-garment max-pooling over views, yielding a ranked top-K list in "
    "milliseconds. We describe the architecture, mathematics, training objective, data pipeline, "
    "distributed two-GPU optimization, evaluation protocol (leave-one-view-out Recall@K and MRR "
    "with garment-level holdout), and full operational usage."
)

INTRO_TEXT = (
    "Our 3D pipeline composes garments combinatorially: garments are authored as separate "
    "meshes rather than single fused outfits. Ten shirts, ten pairs of pants, and ten hats "
    "already span one thousand unique outfits. Today, finding which catalog garments match a "
    "given reference photograph of a dressed person is a manual, slow, error-prone lookup "
    "task performed by artists. We automate it with learned visual retrieval."
)

PROBLEM_TEXT = (
    "Formally: given a query image q containing a person wearing an unknown shirt, and a "
    "catalog C of N garments where each garment c is described by K = 4 canonical render views "
    "(front, back, left, right), output a ranking of catalog garments by visual match to q. "
    "This is an image-retrieval problem, not a classification problem: the catalog is dynamic "
    "(garments are added continuously), so the model must generalize to garments it has never "
    "seen and match by appearance similarity rather than memorized identity. Metric learning "
    "with a contrastive objective is the standard, well-matched tool for this setting: the "
    "encoder is trained to place same-garment views close together and different-garment views "
    "far apart in a metric space, after which nearest-neighbor lookup over precomputed catalog "
    "embeddings solves the ranking step without any retraining when the catalog changes."
)

APPROACH_TEXT = (
    "The system has three stages. (1) Offline catalog encoding: each garment's four render "
    "views are embedded once with the trained encoder and cached. (2) Query encoding: a "
    "photograph is embedded with the same encoder. (3) Matching: cosine similarities between "
    "the query embedding and all catalog view embeddings are computed; per-garment scores use "
    "the maximum over that garment's views; the top-K garments are returned with scores. "
    "Because both encodings live on the unit hypersphere, cosine similarity is a dot product, "
    "and the whole match step is a single matrix multiplication - milliseconds for catalogs of "
    "thousands of garments on CPU, and directly portable to GPU vector indexes (e.g. Faiss) "
    "at tens of millions of items."
)

VIT_TEXT = (
    "The encoder is a standard ViT-S/16. An image is divided into a 14 x 14 grid of "
    "non-overlapping 16 x 16 patches; each patch is linearly embedded into a 384-dimensional "
    "token by a convolutional stem. A learnable classification token (CLS) is prepended, and a "
    "learnable absolute positional embedding is added so that the transformer, which is "
    "permutation-invariant over tokens, can reason about spatial layout. Twelve pre-norm "
    "transformer blocks with six attention heads and MLP ratio 4 process the 197-token "
    "sequence. The final LayerNorm output's CLS token serves as the global image "
    "representation. All weights are initialized from scratch (truncated normal, std 0.02) - "
    "the point of this project is that nothing is imported from a pretrained checkpoint."
)

ATTN_TEXT = (
    "Self-attention lets every token gather information from every other token; heads "
    "specialize in different relations (local texture, hem-to-sleeve correspondence, global "
    "shape). With 197 tokens, full attention costs O(N^2) ~ 38,809 pairwise scores per head - "
    "cheap at ViT-S scale and fully parallel on GPU."
)

BLOCK_TEXT = (
    "Pre-norm (LayerNorm inside the residual branch, not on the trunk) is the modern default: "
    "it keeps the residual stream clean, which stabilizes from-scratch training at high "
    "learning rates without LayerNorm reparameterization or careful warmup tricks. The MLP "
    "expands 384 to 1536 (ratio 4) with GELU nonlinearity. Stochastic depth (DropPath) "
    "randomly drops entire residual branches per sample during training, which acts as an "
    "ensemble regularizer and is scheduled linearly from 0 (first block) to 0.1 (last block)."
)

HEAD_TEXT = (
    "Contrastive methods consistently improve when the loss operates on a projected "
    "representation rather than the raw backbone features (SimCLR, SupCon). We use a "
    "two-layer MLP head (384 to 384, GELU, 384 to 256) whose output is L2-normalized onto "
    "the unit hypersphere S^255. Retrieval at inference uses the same normalized head output, "
    "so training and deployment geometries are identical."
)

SUPCON_TEXT = (
    "In each training batch we place P garments with V views each (defaults: P = 32 per GPU, "
    "V = 2, global batch 256 across two GPUs). Each view is independently augmented, so two "
    "views of the same garment are two different-looking images. The supervised contrastive "
    "loss then operates on the normalized embeddings z_i:"
)

SUPCON_TEXT_2 = (
    "Intuitively: for every anchor, at least one other batch element is the same garment seen "
    "differently (augmented, different angle). The loss maximizes the log-probability, under a "
    "softmax over pairwise similarities, of those positives relative to all other elements. "
    "Same-garment views are pulled to a point; other garments are pushed away. Multi-view "
    "positives (Khosla et al. show 2+ views per class improves over single-positive InfoNCE) "
    "and in-batch negatives from the other 31 garments provide the gradient signal. Because "
    "views differ in angle and augmentation, the encoder cannot rely on view-specific cues - "
    "it must learn garment-invariant appearance, which is exactly what cross-view retrieval "
    "against real photos requires."
)

LOSS_WHY = (
    "Why SupCon alone is not enough: with few hundred to a thousand garments, pure metric "
    "learning can wander - all garments can collapse toward similar directions while still "
    "locally ordering neighbors. The auxiliary linear classifier on the CLS token adds an "
    "explicit identity signal: features must also be linearly separable per training "
    "garment, which sharpens the space. Empirically in the contrastive-literature this "
    "combination (metric + auxiliary CE) is a robust default; our weight of 0.5 keeps SupCon "
    "dominant. The classifier head is discarded at inference."
)

DATA_TEXT = (
    "Training data is the existing catalog renders: each garment directory contains front, "
    "back, left, and right views of the T-posed garment on a plain white background, without "
    "a human model. A preparation script scans the render tree, resolves angle names "
    "(front/f, back/b, left/l/side, right/r), and writes JSONL manifests. Garments are split "
    "by identity (default 5% held out): held-out garments never appear in training, so "
    "evaluation measures generalization to unseen garments, which mirrors production."
)

AUG_TEXT = (
    "The renders are too clean: uniform background, perfect lighting, centered framing. A "
    "model trained on them verbatim will not transfer to photographs. Domain randomization "
    "closes this gap by training on synthetic disturbances of the renders (Tobin et al.): if "
    "the training distribution already spans backgrounds, lighting, viewpoints, and "
    "occlusions, the real-photo distribution falls inside it. Concretely (Figure 3, Table 2): "
    "the near-white background is keyed by flood fill from the image border (only "
    "border-connected white is removed, so white fabric interior is preserved); the garment "
    "is composited onto procedurally painted backgrounds; mild perspective and affine warps "
    "simulate off-axis cameras; photometric jitter, blur, and noise simulate consumer "
    "photography; random rectangles simulate arms and objects partially covering the "
    "subject; and scale-jittered crops around the garment bounding box simulate framing "
    "variation. Horizontal flips are applied - for garments this is safe and doubles data."
)

SAMPLER_TEXT = (
    "SupCon needs positives in every batch, so batches are constructed by a balanced "
    "multi-view sampler rather than shuffled indices: each batch draws P garments, and for "
    "each garment samples V of its available views, guaranteeing every batch contains P "
    "positive groups of size V. In the two-GPU setup the garment set is sharded by rank so "
    "the two GPUs see different garments - the effective global batch has 2P garments and "
    "2PV images, and the SupCon implementation gathers embeddings across ranks so negatives "
    "span both GPUs."
)

OPT_TEXT = (
    "From-scratch ViTs need modern optimization to be stable: AdamW (decoupled weight decay, "
    "applied to weights but not biases or norm parameters), a peak learning rate around "
    "1e-3 for batch 256, 5 epochs of linear warmup, and cosine decay to 1e-5. Gradients are "
    "clipped at global L2 norm 1.0. Mixed precision uses bfloat16 autocast (native on "
    "RTX 5090) with float32 loss computation. In addition, an exponential moving average of "
    "all weights (decay 0.999) is maintained; all evaluation and the exported checkpoint "
    "use EMA weights, which is a free, consistent accuracy gain."
)

DDP_TEXT = (
    "Training runs on two RTX 5090 GPUs with PyTorch DistributedDataParallel. On Windows "
    "the NCCL backend is unavailable, so we use gloo. Two subtleties matter: (i) the SupCon "
    "loss gathers features and labels across ranks via all_gather (on gloo this round-trips "
    "through CPU tensors), with gradients flowing only through the local rank's block; (ii) "
    "the auxiliary classifier is not wrapped by DDP, so its gradients are manually "
    "all-reduced and averaged. The launcher is a small custom Python script "
    "(scripts/launch_ddp.py) that spawns one worker per GPU with MASTER_ADDR/MASTER_PORT/"
    "RANK/LOCAL_RANK environment variables rather than torchrun. Reason: several torch "
    "Windows builds (including 2.5.1) ship without libuv, and torchrun's elastic rendezvous "
    "then fails at TCPStore creation; init_process_group('env://') with USE_LIBUV=0 works "
    "correctly, so the custom launcher sidesteps the broken component. Each worker binds to "
    "its own GPU via LOCAL_RANK."
)

EVAL_TEXT = (
    "Evaluation answers: for a garment never seen in training, does the model rank its other "
    "views correctly against every other held-out garment's views? All held-out garments' "
    "views are embedded. Each view takes a turn as query; the gallery is every other view "
    "(same-garment views excluded from the gallery only when the query is that garment's own "
    "duplicate - in practice each of the 4 angles queries against all views of all other "
    "held-out garments). A retrieval is correct when the top-ranked gallery view belongs to "
    "the same garment (necessarily a different angle, since the query view itself is "
    "excluded from its own gallery). We report Recall@1/5/10/20 and MRR. This protocol is "
    "strict: it measures cross-angle matching, generalization to unseen garments, and "
    "discrimination among near-duplicate garments simultaneously. A real-photo validation "
    "set (labeled photos matched to garment IDs) can be dropped in as an additional manifest; "
    "the code paths are identical."
)

INFER_TEXT = (
    "The catalog is embedded once: each garment contributes its four (or fewer) view "
    "embeddings, L2-normalized, to a matrix. A query photo is preprocessed identically to "
    "training input statistics (resize, center-crop to 224) and embedded. Matching is a dot "
    "product against the catalog matrix. Per-garment scores take the max over that garment's "
    "views, because a photo typically resembles one canonical angle more than others, and "
    "garments are ranked by that best-view score. Output is a top-K list of garment IDs with "
    "cosine similarities."
)

OPS_TEXT = (
    "Encoding the whole catalog takes seconds on GPU and the artifact is a single .pt file; "
    "re-embedding is only needed when the catalog changes (and can be incremental - append "
    "new garments' rows). Matching itself is one matrix product; for very large catalogs the "
    "matrix should move to Faiss or a vector database, with the same embeddings. Query "
    "preprocessing (resize + center-crop) matches eval, not the random training crops - "
    "intentionally, because real photos are already naturally framed. If a query contains "
    "multiple garments (full outfit), run the pipeline per detected garment region or per "
    "garment category classifier first; per-category models (shirts now, pants and hats "
    "later) keep each space clean, since a shirt should never be matched against pants."
)

USAGE_PREP = """cd outfit-matcher
pip install -r requirements.txt

python -m outfit_matcher.data.prepare_data ^
    --data-root D:\\renders\\shirts ^
    --out-dir  D:\\data\\shirts ^
    --val-fraction 0.05 --seed 42"""

USAGE_TRAIN = """scripts\\train_ddp.bat D:\\data\\shirts configs\\shirts.yaml
:: equivalent: python scripts/launch_ddp.py --nproc 2 --config configs/shirts.yaml ^
::                                        --data-override D:/data/shirts"""

USAGE_TRAIN_NOTE = (
    "Checkpoints land in runs/shirts_v1: checkpoint_last.pt (resumable, includes optimizer "
    "and EMA state), periodic snapshots every 10 epochs, checkpoint_final.pt, and JSONL "
    "histories (loss per epoch; Recall@K/MRR every 5 epochs). Resume with train.resume in "
    "the YAML or --resume."
)

USAGE_MATCH = """python -m outfit_matcher.match ^
    --config configs/shirts.yaml ^
    --checkpoint runs\\shirts_v1\\checkpoint_final.pt ^
    --catalog D:\\data\\shirts\\manifest_train.jsonl ^
    --catalog-cache runs\\shirts_v1\\catalog_emb.pt ^
    --query C:\\photos\\someone.jpg ^
    --topk 5"""

USAGE_TESTS = """python -m pytest tests/ -q
:: generates a synthetic garment dataset, trains a tiny model end-to-end,
:: runs eval + retrieval + the prepare-data CLI. Green in ~1 min on CPU."""

LIMITS_TEXT = (
    "First, the model has never seen a human body: renders are T-posed garments without a "
    "wearer, so sleeves drape differently in real photos. Heavy augmentation mitigates but "
    "does not remove this; the strongest fix is compositing garments onto our existing 3D "
    "human models at varied poses and re-rendering - the training pipeline already accepts "
    "any per-garment view images, so this is a data change, not a code change. Second, "
    "scale: 100-1000 garments is small for from-scratch training; the balanced sampler and "
    "strong regularization are chosen for it, but accuracy will scale with catalog size. "
    "Third, occlusion by other garments (jackets over shirts) is only simulated "
    "synthetically. Fourth, this first model is shirt-only by design; pants and hats should "
    "be trained as separate category models and, optionally, distilled later into one "
    "unified embedding space. Finally, evaluation on real photos is pending labeled data; "
    "the leave-one-view-out numbers measure render-side generalization only."
)
