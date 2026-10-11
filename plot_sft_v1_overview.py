#!/usr/bin/env python3
"""Overview figures for the AgentHarm SFT v1 dataset.

Data: hedgehog-quills/benchmarks/agentharm/datasets/sft-toolcalling/combined/v1/
agentharm_sft_toolcalling.jsonl (2,531 single-turn rows = train + test; composition
matches combined/v1/REPORT.md).

Figures (each .pdf + .png):
  sft_v1_composition        rows per category and per generator model (pies).
  sft_v1_model_by_category  category mix of each generator model (100% stacked bars).
  sft_v1_umap               UMAP of the user prompts, coloured by category and by model.
                            Default (--embedding semantic): sentence embeddings
                            (all-mpnet-base-v2, L2-normalised), precomputed by
                            sky_embed/embed.py as a SkyPilot job (this box has too little
                            memory for torch). --embedding tfidf gives the lexical
                            baseline of hedgehog-quills/utils/generate_topic_graph_umap_plot.py
                            (word 1-2 grams, English stopwords, min_df=2), written as
                            sft_v1_umap_tfidf. Both: cosine UMAP, n_neighbors=30,
                            min_dist=0.15, seed 42. 2-D coordinates are cached in
                            --emb-dir as umap_<embedding>_s<seed>.npy.
  sft_v1_complexity         (a) prompt and (b) response length per model, split into
                            allow / refuse rows; (c) response lexical diversity
                            (MATTR, window 50) per category, one box per model.
  sft_v1_diversity_by_category
                            models compared within each category (the models got very
                            different category mixes): response lexical diversity (MATTR)
                            and prompt semantic diversity (per prompt, the mean cosine
                            distance to every other prompt from the same model in the same
                            category; unlike nearest-neighbour distance this is not biased
                            by group size, but small groups are noisy).

Usage (from /home/jovyan/Shared/kexin):
    python agentharm_blog/plot_sft_v1_overview.py --out-dir agentharm_blog/figures
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style
from style import CATEGORIES, MODELS, MODEL_TICKS, WIDTH

DEFAULT_DATA = Path("/home/jovyan/Shared/kexin/hedgehog-quills/benchmarks/agentharm/datasets/"
                    "sft-toolcalling/combined/v1/agentharm_sft_toolcalling.jsonl")
MODEL_KEYS = [k for k, _, _ in MODELS]
MOD_COLOR = {k: c for k, _, c in MODELS}
BOX = dict(patch_artist=True, medianprops=dict(color="black", lw=0.8),
           whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6),
           flierprops=dict(marker=".", ms=2, mfc="#888888", mec="none"))


# ------------------------------------------------------------------ data + metrics

CODE_RE = re.compile(r"```.*?```", re.S)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def mattr(text, window=50):
    """Moving-average type-token ratio (length-robust lexical diversity), code removed."""
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


def embed(texts, kind, model_name, emb_dir):
    if kind == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer  # lazy: heavy import
        return TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english",
                               min_df=2).fit_transform(texts)
    path = emb_dir / f"emb_{model_name.replace('/', '_')}.npy"  # row order = dataset order
    if not path.exists():
        raise SystemExit(f"{path} not found: run `sky jobs launch agentharm_blog/sky_embed/embed.sky.yaml`")
    X = np.load(path)
    if len(X) != len(texts):
        raise SystemExit(f"{path} has {len(X)} rows but the dataset has {len(texts)}; re-run the embed job")
    return X


def umap_coords(rows, kind, model_name, emb_dir, seed):
    cache = emb_dir / f"umap_{kind}_s{seed}.npy"
    if cache.exists():
        Y = np.load(cache)
        if len(Y) == len(rows):
            return Y
    X = embed([r["user"] for r in rows], kind, model_name, emb_dir)
    import umap  # lazy: numba makes this import heavy
    Y = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine", random_state=seed).fit_transform(X)
    np.save(cache, Y)
    return Y


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


def values(rows, key, **match):
    v = np.array([r[key] for r in rows if all(r[k] == m for k, m in match.items())], float)
    return v[np.isfinite(v)]


# ---------------------------------------------------------------------- figures

def fig_composition(rows):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    for ax, field, spec, title in [(axes[0], "cat", CATEGORIES, "(a) Category"),
                                   (axes[1], "model", MODELS, "(b) Generator model")]:
        cnt = Counter(r[field] for r in rows)
        spec = [s for s in spec if cnt[s[0]]]
        _, _, pct = ax.pie([cnt[k] for k, _, _ in spec], colors=[c for _, _, c in spec],
                           startangle=90, counterclock=False, pctdistance=0.72,
                           autopct=lambda v: f"{v:.0f}%" if v >= 5 else "",
                           wedgeprops=dict(edgecolor="white", linewidth=0.8), textprops=dict(fontsize=6.5))
        for t, (k, _, c) in zip(pct, spec):
            t.set_color("white" if c in (style.TOL["indigo"], style.TOL["wine"], style.TOL["green"],
                                         style.TOL["purple"]) else "black")
        ax.set_title(title)
        ax.legend([Patch(fc=c) for _, _, c in spec], [f"{l} ({cnt[k]:,})" for k, l, _ in spec],
                  loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, columnspacing=0.8)
        ax.set_aspect("equal")
    fig.tight_layout(w_pad=2)
    return fig


def fig_model_breakdown(rows):
    cnt = Counter((r["model"], r["cat"]) for r in rows)
    totals = np.array([sum(cnt[(m, k)] for k, _, _ in CATEGORIES) for m in MODEL_KEYS])
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 2.3))
    x = np.arange(len(MODEL_KEYS))
    bottom = np.zeros(len(MODEL_KEYS))
    for k, label, color in CATEGORIES:
        pct = 100 * np.array([cnt[(m, k)] for m in MODEL_KEYS]) / totals
        ax.bar(x, pct, 0.6, bottom=bottom, color=color, label=label)
        bottom += pct
    ax.set_xticks(x, MODEL_TICKS)
    ax.tick_params(axis="x", length=0, labelsize=6.5)
    ax.set_ylabel("Share of rows (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    return fig


def fig_umap(rows, Y, seed):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    order = np.random.default_rng(seed).permutation(len(rows))  # no group always drawn on top
    for ax, field, spec, title in [(axes[0], "cat", CATEGORIES, "(a) Category"),
                                   (axes[1], "model", MODELS, "(b) Generator model")]:
        color = {k: c for k, _, c in spec}
        ax.scatter(Y[order, 0], Y[order, 1], s=3.5, lw=0, c=[color[rows[i][field]] for i in order],
                   rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(title)
        present = {r[field] for r in rows}
        ax.legend([Line2D([], [], ls="", marker="o", ms=3, mfc=c, mec="none") for k, _, c in spec if k in present],
                  [l for k, l, _ in spec if k in present],
                  loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, columnspacing=0.8, handletextpad=0.2)
    fig.tight_layout(w_pad=1.5)
    return fig


def fig_complexity(rows):
    for r in rows:
        r["user_words"] = len(r["user"].split())
        r["asst_words"] = len(r["assistant"].split())
        r["mattr"] = mattr(r["assistant"])
    fig = plt.figure(figsize=(WIDTH, 4.4))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.3)
    x = np.arange(len(MODEL_KEYS))

    for col, (key, title, ticks) in enumerate([("user_words", "(a) Prompt length", [5, 10, 20, 50]),
                                               ("asst_words", "(b) Response length", [10, 100, 1000])]):
        ax = fig.add_subplot(gs[0, col])
        for i, m in enumerate(MODEL_KEYS):
            for beh, dx, alpha in (("allow", -0.18, 0.45), ("refuse", 0.18, 1.0)):
                b = ax.boxplot(values(rows, key, model=m, behavior=beh), positions=[i + dx], widths=0.3, **BOX)
                b["boxes"][0].set_facecolor(mpl.colors.to_rgba(MOD_COLOR[m], alpha))
        ax.set_yscale("log")
        ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xticks(x, MODEL_TICKS)
        ax.tick_params(axis="x", length=0, labelsize=6.5)
        ax.set_ylabel("Words")
        ax.set_title(title)
    fig.axes[0].legend([Patch(fc="#888888", alpha=0.45), Patch(fc="#888888")],
                       ["Benign (allow)", "Harmful (refuse)"], loc="upper right")

    ax = fig.add_subplot(gs[1, :])
    step = 0.2
    for g, (cat, _, _) in enumerate(CATEGORIES):
        for i, m in enumerate(MODEL_KEYS):
            v = values(rows, "mattr", model=m, cat=cat)
            if len(v) < 2:
                continue
            b = ax.boxplot(v, positions=[g + (i - 1.5) * step], widths=0.16, **BOX)
            b["boxes"][0].set_facecolor(MOD_COLOR[m])
    ax.set_xticks(range(len(CATEGORIES)), [l.replace(" (", "\n(") for _, l, _ in CATEGORIES])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(CATEGORIES) - 0.5)
    ax.set_ylabel("MATTR")
    ax.set_title("(c) Response lexical diversity")
    ax.legend([Patch(fc=c) for _, _, c in MODELS], [l for _, l, _ in MODELS],
              loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4)
    return fig


def fig_diversity_by_category(rows, X):
    for r in rows:
        r["mattr"] = mattr(r["assistant"])
    semantic_diversity(rows, X)
    fig, axes = plt.subplots(2, len(CATEGORIES), figsize=(WIDTH, 3.4), sharey="row")
    for row, (key, ylabel) in enumerate([("mattr", "Lexical diversity\n(MATTR)"),
                                         ("sem_div", "Semantic diversity\n(cosine distance)")]):
        for col, (cat, clabel, _) in enumerate(CATEGORIES):
            ax = axes[row, col]
            for i, m in enumerate(MODEL_KEYS):
                v = values(rows, key, model=m, cat=cat)
                if len(v) < 2:
                    continue
                b = ax.boxplot(v, positions=[i], widths=0.6, **BOX)
                b["boxes"][0].set_facecolor(MOD_COLOR[m])
            ax.set_xticks([])
            ax.set_xlim(-0.6, len(MODEL_KEYS) - 0.4)
            if col == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(axis="y", length=0)
            if row == 0:
                ax.set_title(clabel.replace(" (", "\n("), fontsize=7)
    fig.legend([Patch(fc=c) for _, _, c in MODELS], [l for _, l, _ in MODELS],
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.8, w_pad=0.4)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=Path("agentharm_blog/figures"))
    p.add_argument("--dpi", type=int, default=600, help="PNG resolution (PDFs are vector)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding", choices=["semantic", "tfidf"], default="semantic")
    p.add_argument("--st-model", default="all-mpnet-base-v2", help="sentence-transformers model")
    p.add_argument("--emb-dir", type=Path, default=Path("agentharm_blog/sky_embed"),
                   help="folder holding emb_<model>.npy from sky_embed/embed.py")
    figs = ["composition", "breakdown", "umap", "complexity", "diversity"]
    p.add_argument("--only", choices=figs, nargs="+", default=figs)
    a = p.parse_args()

    style.use()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load(a.data)
    print(f"{len(rows)} rows from {a.data}")
    out = lambda stem: str(a.out_dir / stem)  # noqa: E731
    if "composition" in a.only:
        style.save(fig_composition(rows), out("sft_v1_composition"), a.dpi)
    if "breakdown" in a.only:
        style.save(fig_model_breakdown(rows), out("sft_v1_model_by_category"), a.dpi)
    if "umap" in a.only:
        Y = umap_coords(rows, a.embedding, a.st_model, a.emb_dir, a.seed)
        stem = "sft_v1_umap" if a.embedding == "semantic" else "sft_v1_umap_tfidf"
        style.save(fig_umap(rows, Y, a.seed), out(stem), a.dpi)
    if "complexity" in a.only:
        style.save(fig_complexity(rows), out("sft_v1_complexity"), a.dpi)
    if "diversity" in a.only:
        X = embed([r["user"] for r in rows], "semantic", a.st_model, a.emb_dir)
        style.save(fig_diversity_by_category(rows, X), out("sft_v1_diversity_by_category"), a.dpi)


if __name__ == "__main__":
    main()
