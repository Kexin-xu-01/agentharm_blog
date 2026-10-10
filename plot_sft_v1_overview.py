#!/usr/bin/env python3
"""Publication-style (ICML/ICLR/NeurIPS) overview figures for the AgentHarm SFT v1 dataset.

Data: hedgehog-quills/benchmarks/agentharm/datasets/sft-toolcalling/combined/v1/
agentharm_sft_toolcalling.jsonl (2,531 single-turn rows = train + test; composition
matches combined/v1/REPORT.md).

Writes five figures (each as .pdf + .png):
  sft_v1_composition   two donuts: rows per category, rows per generator model.
  sft_v1_model_by_category
                       100% stacked bar per generator model, segmented by
                       category (same colours as the donuts), with row totals.
  sft_v1_umap          UMAP of the user prompts, coloured by category and by
                       generator model. Default (--embedding semantic): sentence
                       embeddings (sentence-transformers all-mpnet-base-v2,
                       L2-normalised), precomputed by sky_embed/embed.py, which
                       runs as a SkyPilot job (sky_embed/embed.sky.yaml) because
                       this JupyterHub box has too little memory for torch.
                       --embedding tfidf reproduces the lexical baseline in
                       hedgehog-quills/utils/generate_topic_graph_umap_plot.py
                       (word 1-2 grams, English stopwords, min_df=2). Both use
                       cosine UMAP with n_neighbors=30, min_dist=0.15, seed 42.
                       The tfidf variant is written as sft_v1_umap_tfidf.
  sft_v1_complexity    (a) prompt and (b) response length per generator model,
                       each split into allow and refuse rows; (c) response lexical
                       diversity (MATTR, window 50) per category, one box per model.
  sft_v1_diversity_by_category
                       models compared WITHIN each category (the models got very
                       different category mixes, see sft_v1_model_by_category, so
                       pooled comparisons would mostly reflect the mix):
                       (a) response lexical diversity, MATTR (window 50);
                       (b) prompt semantic diversity: per prompt, the mean cosine
                       distance to every other prompt from the same model in the
                       same category (sentence embeddings above). The all-pairs mean,
                       unlike nearest-neighbour distance, is not biased by group
                       size, but small groups (n printed under each box) are noisy.

Usage (from /home/jovyan/Shared/kexin):
    python agentharm_blog/plot_sft_v1_overview.py --out-dir agentharm_blog/figures
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_DATA = Path("/home/jovyan/Shared/kexin/hedgehog-quills/benchmarks/agentharm/datasets/"
                    "sft-toolcalling/combined/v1/agentharm_sft_toolcalling.jsonl")
TEXTWIDTH_IN = {"neurips": 5.5, "iclr": 5.5, "icml": 6.75}
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e6e5e0"

# Categories: one hue per domain, dark = harmful (refuse), light = benign (allow).
CATEGORIES = [  # (key, label, colour)
    ("cybercrime", "Cybercrime (harmful)", "#1c5cab"),
    ("benign_cybercrime", "Cybercrime (benign)", "#86b6ef"),
    ("fraud", "Fraud (harmful)", "#c24e1e"),
    ("benign_fraud", "Fraud (benign)", "#f4a582"),
    ("benign_general", "General (benign)", "#008300"),
]
# Generator models: hues disjoint from the category hues.
MODELS = [
    ("claude-sonnet-5", "Claude Sonnet 5", "#1baf7a"),
    ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", "#eda100"),
    ("gpt-4o", "GPT-4o", "#e87ba4"),
    ("gpt-4o-mini", "GPT-4o-mini", "#4a3aa7"),
]
CAT_LABEL = {k: l for k, l, _ in CATEGORIES}
CAT_COLOR = {k: c for k, _, c in CATEGORIES}
MOD_LABEL = {k: l for k, l, _ in MODELS}
MOD_COLOR = {k: c for k, _, c in MODELS}


def set_style(base_pt=7.0):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_pt, "axes.titlesize": base_pt + 0.5, "axes.labelsize": base_pt,
        "xtick.labelsize": base_pt - 0.5, "ytick.labelsize": base_pt - 0.5,
        "legend.fontsize": base_pt - 0.5, "legend.frameon": False,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, out_dir, stem, dpi):
    pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
    fig.savefig(pdf, dpi=dpi)
    # Rasterise the PDF with Ghostscript when available: much lower peak memory than
    # matplotlib's Agg renderer at high dpi (this box has a 4 GB memory limit).
    if shutil.which("gs"):
        subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m",
                        f"-r{dpi}", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
                        f"-sOutputFile={png}", str(pdf)], check=True)
    else:
        fig.savefig(png, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_dir / stem}.pdf/.png")


# ------------------------------------------------------------------ text metrics

CODE_RE = re.compile(r"```.*?```", re.S)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
def mattr(text, window=50):
    """Moving-average type-token ratio (length-robust lexical diversity)."""
    toks = [w.lower() for w in WORD_RE.findall(CODE_RE.sub(" ", text))]
    if len(toks) < window:
        return len(set(toks)) / len(toks) if len(toks) >= 10 else np.nan
    counts = Counter(toks[:window])
    ttrs = [len(counts)]
    for i in range(window, len(toks)):
        counts[toks[i]] += 1
        old = toks[i - window]
        counts[old] -= 1
        if not counts[old]:
            del counts[old]
        ttrs.append(len(counts))
    return float(np.mean(ttrs)) / window


def load(path):
    rows = []
    for line in path.open():
        r = json.loads(line)
        msg = {m["role"]: m.get("content") or "" for m in r["messages"]}
        md = r["metadata"]
        rows.append(dict(cat=md["attack_type"], model=md["generator_model"],
                         behavior=md["expected_behavior"], user=msg.get("user", ""),
                         assistant=msg.get("assistant", "")))
    return rows


# ---------------------------------------------------------------------- figures

def fig_composition(rows, w):
    fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.4))
    specs = [
        ("cat", CATEGORIES, "(a) By category"),
        ("model", MODELS, "(b) By generator model"),
    ]
    for ax, (field, spec, title) in zip(axes, specs):
        cnt = Counter(r[field] for r in rows)
        keys = [k for k, _, _ in spec if cnt[k]]
        vals = [cnt[k] for k in keys]
        colors = [c for k, _, c in spec if cnt[k]]
        wedges, _ = ax.pie(vals, colors=colors, startangle=90, counterclock=False,
                           wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1.0))
        total = sum(vals)
        for wdg, k, v in zip(wedges, keys, vals):
            ang = np.deg2rad((wdg.theta1 + wdg.theta2) / 2)
            x, y = np.cos(ang), np.sin(ang)
            label = dict((kk, ll) for kk, ll, _ in spec)[k]
            ax.annotate(f"{label}\n{v:,} ({v / total:.0%})", xy=(0.99 * x, 0.99 * y),
                        xytext=(1.28 * x, 1.2 * y), ha="left" if x >= 0 else "right", va="center",
                        fontsize=6, color=INK,
                        arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.4, shrinkA=0, shrinkB=0))
        ax.text(0, 0, f"{total:,}\nrows", ha="center", va="center", fontsize=6.5, color=INK_2)
        ax.set_title(title, loc="center", pad=8)
        ax.set_aspect("equal")
    fig.subplots_adjust(wspace=0.75)
    return fig


def embed(texts, kind, model_name, emb_dir):
    if kind == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer  # lazy: heavy import
        return TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english",
                               min_df=2).fit_transform(texts)
    # Precomputed by sky_embed/embed.py (row order = the dataset file's order).
    path = emb_dir / f"emb_{model_name.replace('/', '_')}.npy"
    if not path.exists():
        raise SystemExit(f"{path} not found: run `sky jobs launch agentharm_blog/sky_embed/embed.sky.yaml` "
                         "(or embed.py locally, if the machine has the memory)")
    X = np.load(path)
    if len(X) != len(texts):
        raise SystemExit(f"{path} has {len(X)} rows but the dataset has {len(texts)}; re-run the embed job")
    return X


def fig_model_breakdown(rows, w):
    """100% stacked bar per generator model, segmented by category."""
    models = [k for k, _, _ in MODELS]
    cnt = Counter((r["model"], r["cat"]) for r in rows)
    totals = Counter(r["model"] for r in rows)
    fig, ax = plt.subplots(figsize=(w, w * 0.34))
    for i, m in enumerate(models):
        left = 0.0
        for k, _, c in CATEGORIES:
            n = cnt[(m, k)]
            if not n:
                continue
            frac = n / totals[m]
            ax.barh(i, frac, left=left, height=0.62, color=c, edgecolor="white", linewidth=0.8)
            if frac >= 0.07:
                dark = k in ("cybercrime", "fraud", "benign_general")
                ax.text(left + frac / 2, i, f"{frac:.0%}", ha="center", va="center", fontsize=5.8,
                        color="white" if dark else INK)
            left += frac
        ax.text(1.01, i, f"$n$={totals[m]:,}", ha="left", va="center", fontsize=6, color=INK_2)
    ax.set_yticks(range(len(models)), [MOD_LABEL[m] for m in models])
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("share of the model's rows")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(fc=c, ec="none", label=l) for _, l, c in CATEGORIES]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5,
              handlelength=1.0, handletextpad=0.35, columnspacing=0.9, fontsize=5.8)
    return fig


def fig_umap(rows, w, seed, kind="semantic", model_name="all-mpnet-base-v2", emb_dir=Path(".")):
    X = embed([r["user"] for r in rows], kind, model_name, emb_dir)
    import umap  # lazy: numba makes this import heavy
    emb = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine", random_state=seed).fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(w, w * 0.52))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))  # random z-order so no group always sits on top
    for ax, field, spec, title in [
        (axes[0], "cat", CATEGORIES, "(a) Coloured by category"),
        (axes[1], "model", MODELS, "(b) Coloured by generator model"),
    ]:
        cmap = {k: c for k, _, c in spec}
        ax.scatter(emb[order, 0], emb[order, 1], s=2.2, lw=0,
                   c=[cmap[rows[i][field]] for i in order], alpha=0.85, rasterized=True)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9c8c2")
        ax.set_xlabel("UMAP 1", color=INK_2), ax.set_ylabel("UMAP 2", color=INK_2)
        ax.set_title(title, loc="left", pad=3)
        cnt = Counter(r[field] for r in rows)
        handles = [Line2D([], [], ls="", marker="o", ms=3.5, mfc=c, mec="none",
                          label=f"{l} ({cnt[k]:,})") for k, l, c in spec if cnt[k]]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2,
                  handletextpad=0.2, columnspacing=0.8, fontsize=5.8)
    fig.subplots_adjust(wspace=0.12)
    return fig


BOX = dict(patch_artist=True, showfliers=True, medianprops=dict(color=INK, lw=0.9),
           whiskerprops=dict(color=INK_2, lw=0.5), capprops=dict(color=INK_2, lw=0.5),
           flierprops=dict(marker="o", ms=1.2, mfc=INK_3, mec="none", alpha=0.5))


def semantic_diversity(rows, X):
    """Per row: mean cosine distance to every other prompt of the same model in the same category."""
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["cat"], r["model"]), []).append(i)
    for idx in groups.values():
        E = X[idx] / np.linalg.norm(X[idx], axis=1, keepdims=True)
        n = len(idx)
        d = 1 - ((E @ E.T).sum(axis=1) - 1) / max(n - 1, 1) if n > 1 else [np.nan]
        for i, v in zip(idx, d):
            rows[i]["sem_div"] = float(v)


def model_ticks(ax, models):
    ax.set_xticks(range(len(models)), [MOD_LABEL[m].replace(" 2.5 ", " 2.5\n").replace("Sonnet", "\nSonnet")
                                       for m in models], fontsize=5.8)
    ax.set_xlim(-0.6, len(models) - 0.4)


def fig_complexity(rows, w):
    """(a) prompt and (b) response length per model, split into allow / refuse rows;
    (c) response lexical diversity (MATTR) per category, one box per model."""
    for r in rows:
        r["user_words"] = len(r["user"].split())
        r["asst_words"] = len(r["assistant"].split())
        r["mattr"] = mattr(r["assistant"])
    fig = plt.figure(figsize=(w, w * 0.72))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.55, wspace=0.28, top=0.93, bottom=0.12)
    models = [k for k, _, _ in MODELS]

    offs = {"allow": -0.19, "refuse": 0.19}
    for col, (key, title) in enumerate([("user_words", "(a) Prompt length"),
                                        ("asst_words", "(b) Response length")]):
        ax = fig.add_subplot(gs[0, col])
        for i, m in enumerate(models):
            for beh, dx in offs.items():
                v = np.array([r[key] for r in rows if r["model"] == m and r["behavior"] == beh], float)
                if not len(v):
                    continue
                face = MOD_COLOR[m] if beh == "refuse" else mpl.colors.to_rgba(MOD_COLOR[m], 0.35)
                ax.boxplot(v, positions=[i + dx], widths=0.3,
                           boxprops=dict(facecolor=face, edgecolor=INK_2, lw=0.5), **BOX)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(
            lambda v, _: f"{v:g}" if str(int(round(v)))[0] in "25" else ""))
        model_ticks(ax, models)
        ax.set_ylabel("words")
        ax.set_title(title, loc="left", pad=3)
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    handles = [Patch(fc=mpl.colors.to_rgba(INK_3, 0.35), ec=INK_2, lw=0.5, label="allow (benign request)"),
               Patch(fc=INK_3, ec=INK_2, lw=0.5, label="refuse (harmful request)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.945))

    # (c) lexical diversity within each category, so the models' different category
    # mixes don't drive the comparison.
    ax = fig.add_subplot(gs[1, :])
    step, width = 0.2, 0.17
    for g, (cat, clabel, _) in enumerate(CATEGORIES):
        for i, m in enumerate(models):
            x = g + (i - (len(models) - 1) / 2) * step
            v = np.array([r["mattr"] for r in rows if r["model"] == m and r["cat"] == cat], float)
            v = v[np.isfinite(v)]
            if len(v) < 2:
                ax.text(x, 0.5, "\u2013", transform=ax.get_xaxis_transform(), ha="center",
                        va="center", color=INK_3, fontsize=6)
                continue
            ax.boxplot(v, positions=[x], widths=width,
                       boxprops=dict(facecolor=MOD_COLOR[m], edgecolor=INK_2, lw=0.5), **BOX)
            ax.text(x, -0.035, f"{len(v)}", transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=4.6, color=INK_3)
        if g:
            ax.axvline(g - 0.5, color=GRID, lw=0.6, zorder=0)
    ax.text(-0.5, -0.035, "$n$", transform=ax.get_xaxis_transform(), ha="right", va="top",
            fontsize=4.6, color=INK_3)
    ax.set_xticks(range(len(CATEGORIES)), [l for _, l, _ in CATEGORIES], fontsize=6.3)
    ax.tick_params(axis="x", length=0, pad=9)
    ax.set_xlim(-0.5, len(CATEGORIES) - 0.5)
    ax.set_ylabel("MATTR (window 50)")
    ax.set_title("(c) Response lexical diversity by category", loc="left", pad=3)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(fc=c, ec=INK_2, lw=0.5, label=l) for _, l, c in MODELS],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, handlelength=1.0,
              columnspacing=1.2, fontsize=6.3)
    return fig


def fig_diversity_by_category(rows, w, X):
    """Rows: response lexical diversity, prompt semantic diversity. Columns: category. x: model."""
    for r in rows:
        r["mattr"] = mattr(r["assistant"])
    semantic_diversity(rows, X)
    models = [k for k, _, _ in MODELS]
    metrics = [("mattr", "Response lexical\ndiversity (MATTR)"),
               ("sem_div", "Prompt semantic diversity\n(mean cosine distance)")]
    fig, axes = plt.subplots(2, len(CATEGORIES), figsize=(w, w * 0.56), sharey="row")
    for row, (key, ylabel) in enumerate(metrics):
        for col, (cat, clabel, _) in enumerate(CATEGORIES):
            ax = axes[row, col]
            for i, m in enumerate(models):
                v = np.array([r[key] for r in rows if r["model"] == m and r["cat"] == cat], float)
                v = v[np.isfinite(v)]
                if len(v) < 2:
                    ax.text(i, 0.5, "\u2013", transform=ax.get_xaxis_transform(), ha="center",
                            va="center", color=INK_3, fontsize=6)
                    continue
                ax.boxplot(v, positions=[i], widths=0.62,
                           boxprops=dict(facecolor=MOD_COLOR[m], edgecolor=INK_2, lw=0.5), **BOX)
                if row == 1:
                    ax.text(i, -0.06, f"{len(v)}", transform=ax.get_xaxis_transform(), ha="center",
                            va="top", fontsize=4.8, color=INK_3)
            ax.set_xticks([])
            ax.set_xlim(-0.6, len(models) - 0.4)
            ax.grid(axis="y", color=GRID, lw=0.4)
            ax.set_axisbelow(True)
            if col:
                ax.tick_params(axis="y", length=0)
                ax.spines["left"].set_visible(False)
            else:
                ax.set_ylabel(ylabel, fontsize=6.3)
            if row == 0:
                ax.set_title(clabel.replace(" (", "\n("), fontsize=6.3, pad=3)
        axes[row, 0].text(-0.62, 1.04, "(a)" if row == 0 else "(b)", transform=axes[row, 0].transAxes,
                          fontsize=7, fontweight="bold", va="bottom")
    axes[1, 0].text(-0.02, -0.06, "$n$", transform=axes[1, 0].transAxes, ha="right", va="top",
                    fontsize=4.8, color=INK_3)
    axes[1, 0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    handles = [Patch(fc=c, ec=INK_2, lw=0.5, label=l) for _, l, c in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.985),
               handlelength=1.0, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.08, hspace=0.18)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=Path("agentharm_blog/figures"))
    p.add_argument("--venue", choices=TEXTWIDTH_IN, default="neurips")
    p.add_argument("--dpi", type=int, default=600, help="PNG resolution (PDFs are vector)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding", choices=["semantic", "tfidf"], default="semantic")
    p.add_argument("--st-model", default="all-mpnet-base-v2", help="sentence-transformers model")
    p.add_argument("--emb-dir", type=Path, default=Path("agentharm_blog/sky_embed"),
                   help="folder holding emb_<model>.npy from sky_embed/embed.py")
    p.add_argument("--only", choices=["composition", "breakdown", "umap", "complexity", "diversity"], nargs="+",
                   default=["composition", "breakdown", "umap", "complexity", "diversity"])
    a = p.parse_args()

    set_style()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load(a.data)
    print(f"{len(rows)} rows from {a.data}")
    w = TEXTWIDTH_IN[a.venue]
    if "composition" in a.only:
        save(fig_composition(rows, w), a.out_dir, "sft_v1_composition", a.dpi)
    if "breakdown" in a.only:
        save(fig_model_breakdown(rows, w), a.out_dir, "sft_v1_model_by_category", a.dpi)
    if "umap" in a.only:
        stem = "sft_v1_umap" if a.embedding == "semantic" else "sft_v1_umap_tfidf"
        save(fig_umap(rows, w, a.seed, a.embedding, a.st_model, a.emb_dir), a.out_dir, stem, a.dpi)
    if "complexity" in a.only:
        save(fig_complexity(rows, w), a.out_dir, "sft_v1_complexity", a.dpi)
    if "diversity" in a.only:
        X = embed([r["user"] for r in rows], "semantic", a.st_model, a.emb_dir)
        save(fig_diversity_by_category(rows, w, X), a.out_dir, "sft_v1_diversity_by_category", a.dpi)


if __name__ == "__main__":
    main()
