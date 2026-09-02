# Outfit Matcher — from-scratch ViT garment retrieval

Takes a photo of a person wearing a garment and retrieves the matching garment from
your 3D-render catalog. Built entirely from scratch: custom ViT, supervised-contrastive
(SupCon) training, no pretrained weights.

## Architecture
- **ViT-S/16 from scratch** (~22M params): conv patch stem, learnable CLS + absolute
  pos-embed, 12 pre-norm blocks, 2-layer MLP projection head (256-d).
- **SupCon loss** (Khosla et al. 2020) + auxiliary garment-classification CE on CLS.
  Multi-view positives: the 4 render angles of the same garment.
- **Domain randomization** to close the render→photo gap: white-background keying
  (border flood fill), procedural background composites, affine/perspective warp,
  color jitter, blur, noise, occluders, scale-jitter crops, flips.
- **EMA weights** for eval/export; cosine LR with warmup; AdamW; bf16 autocast;
  2×RTX 5090 via DDP (gloo on Windows) with rank-sharded balanced batches
  (P garments × V views per GPU → global batch 256).

## Retrieval at inference
Encode catalog renders (all 4 angles) once → query photo embedding → cosine
nearest-neighbor → dedupe by garment → top-K with similarity scores.

## Quickstart (training PC, Windows, 2×5090)
```bat
cd outfit-matcher
pip install -r requirements.txt

:: 1. point configs\shirts.yaml data.manifest to your manifests (or use --data-override)
python -m outfit_matcher.data.prepare_data --data-root D:\renders\shirts --out-dir D:\data\shirts

:: 2. launch 2-GPU DDP training
scripts\train_ddp.bat D:\data\shirts configs\shirts.yaml

:: 3. match a real photo against the catalog
python -m outfit_matcher.match --config configs\shirts.yaml ^
    --checkpoint runs\shirts_v1\checkpoint_final.pt ^
    --catalog D:\data\shirts\manifest_train.jsonl ^
    --query C:\photos\someone.jpg --topk 5 ^
    --catalog-cache runs\shirts_v1\catalog_emb.pt
```

**Launcher note**: `scripts\train_ddp.bat` uses `scripts\launch_ddp.py` (custom 2-process
launcher), NOT torchrun. Reason: several torch Windows builds (incl. 2.5.1) ship without
libuv, and torchrun's elastic TCPStore rendezvous then fails with
"use_libuv was requested but PyTorch was built without libuv support".
The custom launcher sets `USE_LIBUV=0` and spawns workers that call
`init_process_group("env://")` directly — verified to work on Windows.

## Evaluation protocol
- **Garment-level holdout**: 5% of garments fully excluded from training.
- **Leave-one-view-out**: each val view is a query; correct hit = any other view
  (different angle) of the same garment.
- Metrics: Recall@1/5/10/20, MRR.

## Repo layout
```
outfit_matcher/
  model/vit.py          # ViT backbone + SupCon projection head (from scratch)
  losses/supcon.py      # SupCon with cross-GPU gather (gloo-safe, CPU gather)
  data/
    prepare_data.py     # scan render tree -> manifests (+train/val garment split)
    dataset.py          # dataset + balanced multi-view batch sampler
    eval_dataset.py     # deterministic eval preprocessing
    transforms.py       # domain randomization: keying, bg composite, warps, occluders
    synth_data.py       # synthetic garment generator (smoke tests)
  engine.py             # DDP init, EMA, AdamW, cosine-warmup schedule
  train.py              # training loop, checkpoints, history
  evaluate.py           # leave-one-view-out Recall@K / MRR, catalog encoding
  match.py              # production: photo -> top-K catalog garments
configs/
  shirts.yaml           # real training config (2x5090)
  shirts_front.yaml     # front-only prototype (train.views: [front])
  smoke.yaml            # tiny CPU config for tests
scripts/
  prepare_data.bat      # manifest generation on the training PC
  train_ddp.bat         # torchrun 2-GPU launch (gloo)
tests/
  test_smoke.py         # end-to-end smoke tests on synthetic data
```

## Tuning knobs for accuracy
- `views_per_garment` (V): 2 (fast) → 3-4 (better SupCon with your 4 angles).
- `garments_per_batch` (P): 32/GPU → 64/GPU if VRAM allows (global batch 512).
- `p_composite` / occlusion probabilities: raise if real photos are cluttered.
- First real-photo evals: try `temperature` 0.07 vs 0.1 vs 0.2.

## Front-only prototype
Train on front renders only (faster first iteration; catalog + eval still use all angles):
```bat
scripts\train_ddp.bat D:\data\shirts configs\shirts_front.yaml
```
- `train.views: [front]` in the YAML filters **training** to front images; SupCon
  positives become two independent augmentations of the same front render
  (SimCLR-style). Eval still runs leave-one-view-out over all 4 angles — a
  meaningful "can a front-only model match a back view?" metric.
- Alternative: generate front-only manifests with
  `python -m outfit_matcher.data.prepare_data ... --views front` (also accepts
  multiple: `--views front back`).
- When done prototyping, switch back to `configs/shirts.yaml` (all views).

## Notes
- No real photos needed to start; the domain randomization is designed to
  bridge the render→photo gap. Once you have labeled real photos, add them as
  extra manifest rows (same garment_id) — dataset + sampler handle it natively.
- Windows DDP uses gloo: CPU-gathered SupCon (labels stay small) + NCCL-style
  separate loss reduces are all handled in losses/supcon.py.
