"""All paper content: sections, equations (LaTeX strings), tables, references.

Numbers here mirror the shipped code (outfit_matcher/). Sections are written in a
research-paper voice with equations in STIX mathtext.
"""

TITLE = "Garment Retrieval with a From-Scratch Vision Transformer"
SUBTITLE = "Supervised Contrastive Learning for Render-to-Photo Matching in a 3D Clothing Pipeline"
AUTHORS = "Computer Vision Team"

# equation source-of-truth: outfit_matcher/losses/supcon.py, model/vit.py, train.py
EQ = {
    "patchify": r"$\mathbf{x} \in \mathbb{R}^{3 \times 224 \times 224} \Rightarrow \mathbf{X}_p \in \mathbb{R}^{196 \times 384}$",
    "z0": (r"$\mathbf{z}_0 = [\,\mathbf{x}_{\mathrm{cls}}\,;\, \mathbf{x}_p^1 \mathbf{E}\,;\, \dots\,;\, \mathbf{x}_p^N \mathbf{E}\,] + \mathbf{E}_{pos},"
           r"\quad \mathbf{E}_{pos} \in \mathbb{R}^{(1+196) \times 384}$"),
    "msa": (r"$\mathrm{MHA}(\mathbf{Z}) = \mathrm{concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\,\mathbf{W}^{O}$"),
    "attention": (r"$\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{\mathbf{Q}_i \mathbf{K}_i^\top}{\sqrt{d_k}}\right)\mathbf{V}_i,"
                  r"\quad \mathbf{Q} = \mathbf{Z}\mathbf{W}^{Q},\ \mathbf{K} = \mathbf{Z}\mathbf{W}^{K},\ \mathbf{V} = \mathbf{Z}\mathbf{W}^{V}$"),
    "block": (r"$\mathbf{z}'_{\ell} = \mathbf{z}_{\ell} + \mathrm{DropPath}(\mathrm{MHA}(\mathrm{LN}(\mathbf{z}_{\ell}))),\qquad "
              r"\mathbf{z}_{\ell+1} = \mathbf{z}'_{\ell} + \mathrm{DropPath}(\mathrm{MLP}(\mathrm{LN}(\mathbf{z}'_{\ell})))$"),
    "mlp": (r"$\mathrm{MLP}(\mathbf{u}) = \mathbf{W}_2\,\mathrm{GELU}(\mathbf{W}_1 \mathbf{u} + \mathbf{b}_1) + \mathbf{b}_2,"
            r"\quad \mathbf{W}_1 \in \mathbb{R}^{1536 \times 384},\ \mathbf{W}_2 \in \mathbb{R}^{384 \times 1536}$"),
    "gelu": r"$\mathrm{GELU}(u) = u\,\Phi(u)$",
    "droppath": (r"$\widetilde{\mathrm{DropPath}}(a) = \frac{\mathbf{m}}{1-p}\,a,\ \ \mathbf{m} \sim \mathrm{Bernoulli}(1-p),\ "
                 r"\ p_{\ell} = 0 + (\ell / 11) \times 0.1$"),
    "head": (r"$f_\theta(x) = \mathrm{normalize}(\mathbf{W}_2\,\mathrm{GELU}(\mathbf{W}_1\,\mathbf{z}_{L,0} + \mathbf{b}_1) + \mathbf{b}_2) \in \mathbb{S}^{255}$"),
    "supcon": (r"$\mathcal{L}_{\mathrm{SupCon}} = \sum_{i \in I}\ \frac{-1}{|P(i)|}\ \sum_{p \in P(i)}\ \log\ \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$"),
    "supcon_defs": (r"$I = \{1 \dots 2N\}$ (index set), $\ A(i) = I \setminus \{i\}$ (all except self), $\ P(i) = \{p \in A(i) : y_p = y_i\}$ (positives = same garment), $\ \tau = 0.1$"),
    "auxce": (r"$\mathcal{L}_{\mathrm{CE}} = -\sum_{i} \log\ \frac{\exp(\mathbf{g}_i^{(y_i)})}{\sum_{c=1}^{C}\exp(\mathbf{g}_i^{(c)})},"
             r"\quad \mathbf{g}_i = \mathbf{W}_{\mathrm{cls}}\,\mathbf{z}_{L,0}^{(i)} \in \mathbb{R}^{C}$"),
    "total": r"$\mathcal{L} = \mathcal{L}_{\mathrm{SupCon}} + 0.5\,\mathcal{L}_{\mathrm{CE}}$",
    "ema": (r"$\theta_{\mathrm{EMA}} \leftarrow m\,\theta_{\mathrm{EMA}} + (1-m)\,\theta,\qquad m = 0.999$"),
    "lr": (r"$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\!\left(1 + \cos\!(\pi\,\frac{t - t_w}{T - t_w})\right),"
           r"\quad t < t_w\!:\ \eta_t = \eta_{\max}\,t / t_w$"),
    "adamw": (r"$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t,\ \ \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^{\odot 2},\ "
              r"\ \mathbf{p}_t \leftarrow \mathbf{p}_{t-1} - \eta_t\!\left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda\,\mathbf{p}_{t-1}\right)$"),
    "nn": (r"$\hat{c}(q) = \mathrm{arg\,max}_{c \in \mathcal{C}}\ \max_{v \in \mathrm{views}(c)}\ \frac{f_\theta(q) \cdot f_\theta(v)}{\|f_\theta(q)\| \|f_\theta(v)\|}$"),
    "topk": (r"$\mathrm{score}(q, c) = \max_{v \in \mathrm{views}(c)}\ s(f_\theta(q), f_\theta(v)),\qquad s(u, w) = u \cdot w$ (unit vectors)"),
    "recall": (r"$\mathrm{Recall}@K = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \mathbf{1}\!\left[\hat{c}(q)\ \mathrm{in\ top-}K\ \wedge\ \hat{c}(q) = c^*(q)\right]$"),
    "mrr": (r"$\mathrm{MRR} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \frac{1}{\mathrm{rank}_q},\quad \mathrm{rank}_q = \text{position of first correct garment}$"),
    "lovo": (r"$\mathcal{Q} = \{\,\text{val views}\,\},\ \ \mathrm{gallery} = \mathcal{Q} \setminus \{\text{query view}\}\ \wedge\ \text{correct} \Leftrightarrow\ \text{same garment, different angle}$"),
    "params": (r"$\mathrm{params} = 294{,}912_{\mathrm{stem}} + 75{,}648_{\mathrm{pos}} + 12 \times 1{,}807{,}680_{\mathrm{block}} + 364{,}800_{\mathrm{head}} + \mathrm{biases} = 21{,}912{,}064$"),
    "teaser": r"$10\,\mathrm{shirts} \times 10\,\mathrm{pants} \times 10\,\mathrm{hats} = 1000$ outfits, one garment at a time: $f_\theta(\mathrm{photo}) \rightarrow \mathrm{top}\!-\!K\ \mathrm{garments}$",
}

PARAM_TABLE = [
    ["Component", "Shape", "Parameters"],
    ["Patch stem (Conv2d)", "384 x 3 x 16 x 16", "294,912"],
    ["Positional embedding", "197 x 384", "75,648"],
    ["CLS token", "1 x 384", "384"],
    ["12 x Block (MHA + MLP)", "12 x (1.81M)", "21,692,160"],
    ["Final LayerNorm", "384", "768"],
    ["Projection head", "384-384-256 MLP", "364,800"],
    ["Total", "", "22,341,392 (backbone 21.9M)"],
]

TRAIN_TABLE = [
    ["Hyperparameter", "Value", "Rationale"],
    ["Optimizer", "AdamW", "decoupled weight decay, ViT standard"],
    ["Base LR", "1e-3", "scaled for batch 256"],
    ["Min LR", "1e-5", "cosine floor"],
    ["Warmup", "5 epochs (linear)", "stabilizes early attention"],
    ["Schedule", "cosine", "smooth decay to floor"],
    ["Weight decay", "0.05", "on weights only; 0 on bias/norm"],
    ["Batch (global)", "256 = 2 GPUs x 32 garments x 2 views", "balanced multi-view SupCon"],
    ["Temperature tau", "0.1", "sharpens softmax over sims"],
    ["SupCon + aux CE", "1.0 : 0.5", "retrieval + classification signal"],
    ["DropPath", "0.0 to 0.1 (linear)", "stochastic depth regularization"],
    ["EMA decay", "0.999", "eval/export weights"],
    ["Epochs", "100", "convergence on small datasets"],
    ["Precision", "bf16 autocast", "RTX 5090 native"],
    ["Grad clip", "1.0 (global L2 norm)", "exploding-gradient guard"],
]

AUG_TABLE = [
    ["Augmentation", "Probability", "Parameters"],
    ["Background keying (flood fill)", "always (85% composite)", "tol=0.08, grow=2px"],
    ["Procedural background", "0.85", "8 palettes + low-freq noise"],
    ["Perspective warp", "0.3", "corner shift up to 15%"],
    ["Rotation + scale", "always", "+/-7 deg, 0.95-1.1"],
    ["Color jitter", "always", "brightness/contrast/sat x0.4"],
    ["Grayscale", "0.05", "-"],
    ["Gaussian blur", "0.2", "sigma 0.5-1.5, k=3/5"],
    ["Gaussian noise", "0.2", "sigma 0.01-0.05"],
    ["Occluder rectangles", "0.5", "2-15% of image, up to 2"],
    ["Garment crop", "always", "scale 0.5-1.0 of bbox"],
    ["Horizontal flip", "0.5", "-"],
]

FILE_TABLE = [
    ["File", "Role"],
    ["outfit_matcher/model/vit.py", "ViT backbone, projection head (no pretrained weights)"],
    ["outfit_matcher/losses/supcon.py", "SupCon loss with cross-GPU gathering"],
    ["outfit_matcher/data/prepare_data.py", "render tree -> manifests + train/val garment split"],
    ["outfit_matcher/data/dataset.py", "dataset + balanced multi-view batch sampler"],
    ["outfit_matcher/data/transforms.py", "domain randomization: keying, warps, occluders"],
    ["outfit_matcher/engine.py", "DDP init (gloo/Windows), EMA, AdamW groups, cosine schedule"],
    ["outfit_matcher/train.py", "training loop, checkpoints, history, eval hooks"],
    ["outfit_matcher/evaluate.py", "leave-one-view-out Recall@K / MRR + catalog encoding"],
    ["outfit_matcher/match.py", "production CLI: photo -> top-K catalog garments"],
    ["scripts/launch_ddp.py", "custom 2-GPU DDP launcher (torchrun-free, libuv-safe)"],
    ["scripts/train_ddp.bat", "one-click dual-5090 training entry"],
]

REFS = [
    "Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.",
    "Khosla, P., et al. (2020). Supervised Contrastive Learning. NeurIPS 2020.",
    "Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). ICML 2020.",
    "Tolstikhin, I., et al. (2021). MLP-Mixer. Appendix: from-scratch ViT training recipes (AdamW, high LR, strong augmentation).",
    "Huang, G., et al. (2016). Deep Networks with Stochastic Depth. ECCV 2016.",
    "Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). ICLR 2019.",
    "He, K., et al. (2016). Deep Residual Learning. Residual connection formulation.",
    "Tobin, J., et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World. IROS 2017.",
    "Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. EMA + strong augmentation practice.",
    "Johnson, J., Douze, M., & Jegou, H. (2019). Billion-scale similarity search with GPUs. IEEE TBDC. (Faiss, nearest-neighbor retrieval.)",
]


