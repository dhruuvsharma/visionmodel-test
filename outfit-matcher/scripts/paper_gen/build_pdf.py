"""Build the research-paper PDF with reportlab.

Layout: single-column (compact technical report style), STIX fonts,
matplotlib-mathtext equation images, numbered sections, tables, figures.
Usage:
    python scripts/paper_gen/build_pdf.py
Writes: docs/OutfitMatcher_Whitepaper.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Frame, Image, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.paper_gen import content as C  # noqa: E402
from scripts.paper_gen import prose as P  # noqa: E402

ABSTRACT_TEXT = P.ABSTRACT_TEXT
INTRO_TEXT = P.INTRO_TEXT
PROBLEM_TEXT = P.PROBLEM_TEXT
APPROACH_TEXT = P.APPROACH_TEXT
VIT_TEXT = P.VIT_TEXT
ATTN_TEXT = P.ATTN_TEXT
BLOCK_TEXT = P.BLOCK_TEXT
HEAD_TEXT = P.HEAD_TEXT
SUPCON_TEXT = P.SUPCON_TEXT
SUPCON_TEXT_2 = P.SUPCON_TEXT_2
LOSS_WHY = P.LOSS_WHY
DATA_TEXT = P.DATA_TEXT
AUG_TEXT = P.AUG_TEXT
SAMPLER_TEXT = P.SAMPLER_TEXT
OPT_TEXT = P.OPT_TEXT
DDP_TEXT = P.DDP_TEXT
EVAL_TEXT = P.EVAL_TEXT
INFER_TEXT = P.INFER_TEXT
OPS_TEXT = P.OPS_TEXT
USAGE_PREP = P.USAGE_PREP
USAGE_TRAIN = P.USAGE_TRAIN
USAGE_TRAIN_NOTE = P.USAGE_TRAIN_NOTE
USAGE_MATCH = P.USAGE_MATCH
USAGE_TESTS = P.USAGE_TESTS
LIMITS_TEXT = P.LIMITS_TEXT

PAGE_W, PAGE_H = A4
M_L, M_R, M_T, M_B = 20 * mm, 20 * mm, 20 * mm, 20 * mm
TEXT_W = PAGE_W - M_L - M_R

# ----------------------------------------------------------------------
# styles
# ----------------------------------------------------------------------
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Times-Roman",
                      fontSize=9.6, leading=13.6, alignment=TA_JUSTIFY, spaceAfter=5)
TITLE = ParagraphStyle("title", parent=BODY, fontName="Times-Bold", fontSize=17.5,
                       leading=21.5, alignment=TA_CENTER, spaceAfter=4)
SUBTITLE = ParagraphStyle("subtitle", parent=BODY, fontName="Times-Italic", fontSize=11,
                          leading=14.5, alignment=TA_CENTER, textColor=colors.HexColor("#333333"))
AUTHOR = ParagraphStyle("author", parent=BODY, alignment=TA_CENTER, fontSize=9.5,
                        spaceAfter=2, textColor=colors.HexColor("#555555"))
H1 = ParagraphStyle("h1", parent=BODY, fontName="Times-Bold", fontSize=13,
                    leading=16, spaceBefore=11, spaceAfter=5, alignment=TA_LEFT,
                    textColor=colors.HexColor("#1a2c47"))
H2 = ParagraphStyle("h2", parent=BODY, fontName="Times-Bold", fontSize=10.8,
                    leading=13.8, spaceBefore=8, spaceAfter=3, alignment=TA_LEFT,
                    textColor=colors.HexColor("#1a2c47"))
CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=8.2, leading=10.5,
                         alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                         spaceBefore=2, spaceAfter=8)
EQ_STYLE = ParagraphStyle("eq", parent=BODY, alignment=TA_CENTER, spaceBefore=3,
                           spaceAfter=5, fontSize=10)
ABSTRACT = ParagraphStyle("abstract", parent=BODY, fontSize=9.2, leading=13.2,
                          leftIndent=6 * mm, rightIndent=6 * mm, spaceAfter=8,
                          spaceBefore=4, textColor=colors.HexColor("#222222"))
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=7.8,
                      leading=10.4, alignment=TA_LEFT, backColor=colors.HexColor("#f4f6fa"),
                      borderColor=colors.HexColor("#d0d7e2"), borderWidth=0.5,
                      borderPadding=6, leftIndent=2, spaceAfter=8)
REF = ParagraphStyle("ref", parent=BODY, fontSize=8.6, leading=11.5,
                     alignment=TA_LEFT, spaceAfter=3, leftIndent=6 * mm,
                     firstLineIndent=-6 * mm)
FOOT = ParagraphStyle("foot", parent=BODY, fontSize=8, alignment=TA_CENTER,
                      textColor=colors.HexColor("#777777"))

# ----------------------------------------------------------------------
# equation rendering via matplotlib mathtext -> PNG
# ----------------------------------------------------------------------

def eq_image(tex: str, path: Path, fontsize=13) -> Path:
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, tex, fontsize=fontsize)
    fig.canvas.draw()  # type: ignore[attr-defined]
    bbox = t.get_window_extent()
    width, height = bbox.width / fig.dpi, bbox.height / fig.dpi
    fig.set_size_inches(max(width, 0.2), max(height, 0.15))
    fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


EQ_FILES: dict = {}


def E(story, key: str, caption: str | None = None, fontsize=13):
    if key not in EQ_FILES:
        eq_dir = Path(__file__).parent / "_eq"
        eq_dir.mkdir(exist_ok=True)
        EQ_FILES[key] = eq_image(C.EQ[key], eq_dir / f"eq_{key}.png", fontsize=fontsize)
    img = Image(str(EQ_FILES[key]))
    scale = 0.52
    iw, ih = img.imageWidth, img.imageHeight  # pixels
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    img.hAlign = "CENTER"
    story.append(img)
    if caption:
        story.append(Paragraph(caption, CAPTION))
    else:
        story.append(Spacer(1, 4))


# ----------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------

def make_table(story, data, col_widths, caption=None, header=True, fs=7.8):
    rows = [[Paragraph(c, ParagraphStyle("th", parent=BODY, fontName="Times-Bold",
                                         fontSize=fs, alignment=TA_LEFT))
             if header and r == 0 else
             Paragraph(c, ParagraphStyle("td", parent=BODY, fontSize=fs, alignment=TA_LEFT))
             for c in row] for r, row in enumerate(data)]
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")) if header else ("BACKGROUND", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2d4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fb")]),
    ]))
    story.append(t)
    if caption:
        story.append(Paragraph(caption, CAPTION))
    else:
        story.append(Spacer(1, 8))


def code_block(story, text):
    for line in text.strip("\n").splitlines():
        story.append(Paragraph(line.replace(" ", "&nbsp;"), CODE))


# ----------------------------------------------------------------------
# document shell (header/footer + title block)
# ----------------------------------------------------------------------

def on_page(canv, doc):
    canv.saveState()
    canv.setFont("Times-Roman", 8)
    canv.setFillColor(colors.HexColor("#777777"))
    canv.drawString(M_L, 12 * mm, "Outfit Matcher - Technical Whitepaper")
    canv.drawRightString(PAGE_W - M_R, 12 * mm, f"Page {doc.page}")
    canv.setStrokeColor(colors.HexColor("#bbbbbb"))
    canv.setLineWidth(0.4)
    canv.line(M_L, 15 * mm, PAGE_W - M_R, 15 * mm)
    if doc.page > 1:
        canv.setFont("Times-Italic", 8)
        canv.drawCentredString(PAGE_W / 2, PAGE_H - 12 * mm, C.TITLE)
    canv.restoreState()


def build(out_pdf: Path, figs: dict):
    from reportlab.platypus import BaseDocTemplate

    doc = BaseDocTemplate(
        str(out_pdf), pagesize=A4,
        leftMargin=M_L, rightMargin=M_R, topMargin=M_T, bottomMargin=M_B,
        title=C.TITLE, author=C.AUTHORS,
    )
    frame = Frame(M_L, M_B, TEXT_W, PAGE_H - M_T - M_B, id="main")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=on_page)])

    S: list = []
    # ---------------- title ----------------
    S.append(Paragraph(C.TITLE, TITLE))
    S.append(Paragraph(C.SUBTITLE, SUBTITLE))
    S.append(Spacer(1, 3))
    S.append(Paragraph(C.AUTHORS, AUTHOR))
    S.append(Paragraph("Internal technical whitepaper - Computer Vision / 3D Assets", AUTHOR))
    S.append(Spacer(1, 6))
    S.append(Paragraph("<b>Abstract.</b> " + ABSTRACT_TEXT, ABSTRACT))
    S.append(Spacer(1, 2))

    # ---------------- 1 intro ----------------
    S.append(Paragraph("1&nbsp;&nbsp;Introduction and Problem Statement", H1))
    S.append(Paragraph(INTRO_TEXT, BODY))
    E(S, "teaser", None, fontsize=11)
    S.append(Paragraph(PROBLEM_TEXT, BODY))

    S.append(Paragraph("2&nbsp;&nbsp;Approach Overview", H1))
    S.append(Paragraph(APPROACH_TEXT, BODY))
    S.append(Paragraph("2.1&nbsp;&nbsp;System diagram", H2))
    S.append(Image(str(figs["arch"]), width=TEXT_W, height=TEXT_W * img_aspect(figs["arch"])))
    S.append(Paragraph("Figure 1: Outfit Matcher encoder architecture. Every component is trained from random initialization - no pretrained weights are used anywhere.", CAPTION))

    # ---------------- 3 vit ----------------
    S.append(Paragraph("3&nbsp;&nbsp;The Vision Transformer, from Scratch", H1))
    S.append(Paragraph(VIT_TEXT, BODY))
    S.append(Paragraph("3.1&nbsp;&nbsp;Patch embedding", H2))
    E(S, "patchify", "Equation 1: Convolutional patchification. A Conv2d with kernel = stride = 16 maps each 16x16 patch to a 384-d token.")
    S.append(Paragraph("3.2&nbsp;&nbsp;Token assembly and positional information", H2))
    E(S, "z0", "Equation 2: Input token sequence. A learnable CLS token is prepended; a learnable absolute positional embedding is added.")
    S.append(Paragraph("3.3&nbsp;&nbsp;Multi-head self-attention", H2))
    E(S, "attention", "Equation 3: Scaled dot-product attention per head (d_k = 64).")
    E(S, "msa", "Equation 4: Multi-head concatenation and output projection (h = 6 heads).")
    S.append(Paragraph(ATTN_TEXT, BODY))
    S.append(Paragraph("3.4&nbsp;&nbsp;Transformer block (pre-norm residual)", H2))
    E(S, "block", "Equation 5: Pre-norm block. LayerNorm before each sublayer; residual streams carry information; DropPath adds stochastic depth.")
    E(S, "mlp", "Equation 6: MLP with GELU.")
    S.append(Paragraph(BLOCK_TEXT, BODY))
    E(S, "droppath", "Equation 7: Stochastic depth (per-sample), linearly scheduled from 0 to 0.1 across the 12 blocks.")
    S.append(Paragraph("3.5&nbsp;&nbsp;Projection head", H2))
    E(S, "head", "Equation 8: Two-layer MLP projection head to the 256-d hypersphere S^255 (L2-normalized output).")
    S.append(Paragraph(HEAD_TEXT, BODY))
    make_table(S, C.PARAM_TABLE, [58 * mm, 52 * mm, 60 * mm],
               "Table 1: Parameter inventory. Backbone 21.9M; head 0.36M. All initialized from scratch (trunc-normal std 0.02).")

    # ---------------- 4 supcon ----------------
    S.append(Paragraph("4&nbsp;&nbsp;Training Objective: Supervised Contrastive Learning", H1))
    S.append(Paragraph(SUPCON_TEXT, BODY))
    E(S, "supcon", "Equation 9: Supervised contrastive loss over the multi-view batch (Khosla et al., 2020).")
    E(S, "supcon_defs", None, fontsize=10.5)
    S.append(Paragraph(SUPCON_TEXT_2, BODY))
    S.append(Paragraph("4.1&nbsp;&nbsp;Auxiliary classification loss", H2))
    E(S, "auxce", "Equation 10: Auxiliary cross-entropy on the CLS token via a linear classifier (one logit per training garment).")
    E(S, "total", "Equation 11: Total loss.")
    S.append(Paragraph("4.2&nbsp;&nbsp;Why these losses", H2))
    S.append(Paragraph(LOSS_WHY, BODY))
    S.append(Paragraph("4.3&nbsp;&nbsp;What the embedding space looks like", H2))
    S.append(Image(str(figs["embed"]), width=TEXT_W * 0.82, height=TEXT_W * 0.82 * img_aspect(figs["embed"])))
    S.append(Paragraph("Figure 2: Supervised contrastive learning pulls all views of one garment together and pushes different garments apart, which is exactly the geometry nearest-neighbor retrieval needs.", CAPTION))

    # ---------------- 5 data ----------------
    S.append(Paragraph("5&nbsp;&nbsp;Data Pipeline and Domain Randomization", H1))
    S.append(Paragraph(DATA_TEXT, BODY))
    S.append(Paragraph("5.1&nbsp;&nbsp;Augmentation pipeline", H2))
    S.append(Image(str(figs["aug"]), width=TEXT_W, height=TEXT_W * img_aspect(figs["aug"])))
    S.append(Paragraph("Figure 3: Actual pipeline stages executed by the shipped augmentation code (outfit_matcher/data/transforms.py) on a sample render: keying, procedural background, perspective + affine warp, occlusion, crop.", CAPTION))
    make_table(S, C.AUG_TABLE, [64 * mm, 30 * mm, 76 * mm],
               "Table 2: Domain randomization inventory (probability = per-sample chance).")
    S.append(Paragraph(AUG_TEXT, BODY))
    S.append(Paragraph("5.2&nbsp;&nbsp;Balanced multi-view batching", H2))
    S.append(Paragraph(SAMPLER_TEXT, BODY))

    # ---------------- 6 training ----------------
    S.append(Paragraph("6&nbsp;&nbsp;Optimization and Infrastructure", H1))
    S.append(Paragraph(OPT_TEXT, BODY))
    E(S, "adamw", "Equation 12: AdamW with decoupled weight decay; bias-corrected moments.", fontsize=11)
    E(S, "lr", "Equation 13: Cosine schedule with 5-epoch linear warmup (eta_max = 1e-3, eta_min = 1e-5).", fontsize=11)
    S.append(Image(str(figs["lr"]), width=TEXT_W * 0.62, height=TEXT_W * 0.62 * img_aspect(figs["lr"])))
    S.append(Paragraph("Figure 4: Learning-rate schedule over 100 epochs.", CAPTION))
    E(S, "ema", "Equation 14: Exponential moving average of weights used for all evaluation and export.", fontsize=11)
    make_table(S, C.TRAIN_TABLE, [40 * mm, 58 * mm, 72 * mm],
               "Table 3: Training hyperparameters (configs/shirts.yaml).")
    S.append(Paragraph("6.1&nbsp;&nbsp;Dual-GPU distributed training", H2))
    S.append(Paragraph(DDP_TEXT, BODY))

    # ---------------- 7 eval ----------------
    S.append(Paragraph("7&nbsp;&nbsp;Evaluation Protocol and Metrics", H1))
    S.append(Paragraph(EVAL_TEXT, BODY))
    E(S, "recall", "Equation 15: Recall@K over the query set.", fontsize=11.5)
    E(S, "mrr", "Equation 16: Mean reciprocal rank.", fontsize=11.5)
    E(S, "lovo", "Equation 17: Leave-one-view-out protocol. A retrieval is correct only if the gallery view belongs to the same garment at a different angle.", fontsize=10.5)

    # ---------------- 8 inference ----------------
    S.append(Paragraph("8&nbsp;&nbsp;Inference: Matching a Photo to the Catalog", H1))
    S.append(Paragraph(INFER_TEXT, BODY))
    E(S, "nn", "Equation 18: Match rule - the best-scoring view determines the garment.", fontsize=11.5)
    E(S, "topk", "Equation 19: Per-garment score for top-K ranking (max over the garment's catalog views).", fontsize=11.5)
    S.append(Image(str(figs["retrieval"]), width=TEXT_W * 0.7, height=TEXT_W * 0.7 * img_aspect(figs["retrieval"])))
    S.append(Paragraph("Figure 5: Top-K retrieval output. Scores are cosine similarities in [-1, 1].", CAPTION))
    S.append(Paragraph("8.1&nbsp;&nbsp;Operational notes", H2))
    S.append(Paragraph(OPS_TEXT, BODY))

    # ---------------- 9 usage ----------------
    S.append(Paragraph("9&nbsp;&nbsp;How to Run It", H1))
    S.append(Paragraph("9.1&nbsp;&nbsp;Prepare manifests", H2))
    code_block(S, USAGE_PREP)
    S.append(Paragraph("9.2&nbsp;&nbsp;Train on two GPUs", H2))
    code_block(S, USAGE_TRAIN)
    S.append(Paragraph(USAGE_TRAIN_NOTE, BODY))
    S.append(Paragraph("9.3&nbsp;&nbsp;Match a photo", H2))
    code_block(S, USAGE_MATCH)
    S.append(Paragraph("9.4&nbsp;&nbsp;Smoke tests", H2))
    code_block(S, USAGE_TESTS)
    make_table(S, C.FILE_TABLE, [72 * mm, 98 * mm],
               "Table 4: Repository map.")

    # ---------------- 10 limitations ----------------
    S.append(Paragraph("10&nbsp;&nbsp;Limitations and Future Work", H1))
    S.append(Paragraph(LIMITS_TEXT, BODY))

    # ---------------- refs ----------------
    S.append(Paragraph("References", H1))
    for i, r in enumerate(C.REFS, 1):
        S.append(Paragraph(f"[{i}]&nbsp;&nbsp;{r}", REF))

    doc.build(S)
    return out_pdf


def img_aspect(p) -> float:
    from PIL import Image as PILImage
    with PILImage.open(p) as im:
        return im.height / im.width


if __name__ == "__main__":
    fig_dir = ROOT / "docs" / "paper_figs"
    synth = ROOT / "tests" / "_smoke_data"
    if not synth.exists():
        print("synthetic smoke data missing - generating...")
        from outfit_matcher.data.synth_data import main as synth_main
        sys.argv = ["synth", "--out-dir", str(synth), "--garments", "12", "--size", "160"]
        synth_main()
    figs = __import__("scripts.paper_gen.figures", fromlist=["build_all"]).build_all(fig_dir, str(synth))
    out = ROOT / "docs" / "OutfitMatcher_Whitepaper.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = build(out, figs)
    print(f"PDF written -> {result}  ({result.stat().st_size/1024:.0f} KB)")
