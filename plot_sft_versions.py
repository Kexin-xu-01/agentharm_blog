#!/usr/bin/env python3
"""AgentHarm SFT v1 to v5: category coverage, topic diversity, the word "fake", length,
lexical and semantic diversity, and a joint UMAP of the user prompts.
Five-version extension of plot_sft_v1_vs_v2.py (same panels, metrics and style).

Data (hedgehog-quills/benchmarks/agentharm/datasets/sft-toolcalling/combined/):
  SFT v1  v1/agentharm_sft_toolcalling.jsonl       2,531 rows, 4 generator models
  SFT v2  v2.1/agentharm_sft_toolcalling_v2.1.jsonl 1,702 rows (v2 + benign publictest gap-fill)
  SFT v3  v3/agentharm_sft_toolcalling_v3.jsonl       666 rows (v3 + targeted fraud v3.1)
  SFT v4  the 51 rows v2_v3_v4/ adds on top of v2_and_v3/ (48 fraud harmful, 3 general benign;
          v4 has no standalone file). Its benign_general_capability rows count as General (benign).
  SFT v5  v5/agentharm_sft_toolcalling_v5.jsonl       743 rows
Each version is the batch generated for that version, not the cumulative training mix.
v2-v5 are Claude Sonnet 5 only; v1 mixes four generator models.

Figures (each .pdf + .png):
  sft_versions_composition  share of prompts per category: one donut per version and (f) all
                         versions together, as in fig_composition of plot_sft_v1_overview.py.
  sft_versions_coverage  (a) rows, (b) distinct topic-graph nodes, (c) % prompts containing "fake".
  sft_versions_summary   (a) rows, (b) "fake", (c) prompt / (d) response length, (e) lexical
                         diversity (distinct bigrams in 3,000-word samples; left blank for
                         categories with fewer words), (f) semantic diversity (mean cosine distance
                         to the other prompts of the same category and version; groups of fewer
                         than 10 prompts are left out).
  sft_versions_umap      one UMAP fitted on all five versions (all-mpnet-base-v2, cosine,
                         n_neighbors=30, min_dist=0.15, seed 42); one panel per version coloured
                         by category over the others in grey, and (f) all versions together,
                         coloured by category.
Embeddings and the joint UMAP come from sky_embed/embed_v3_v4_v5.sky.yaml (a SkyPilot job),
reusing the v1 and v2.1 embeddings from the earlier jobs.

Usage (from /home/jovyan/Shared/kexin):
    python agentharm_blog/plot_sft_versions.py --out-dir agentharm_blog/figures
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style
from style import CATEGORIES, TOL, WIDTH

DATA = Path("/home/jovyan/Shared/kexin/hedgehog-quills/benchmarks/agentharm/datasets/"
            "sft-toolcalling/combined")
EMB = Path("/home/jovyan/Shared/kexin/agentharm_blog/sky_embed")
GROUPS = [  # key, label, colour
    ("v1", "SFT v1", TOL["cyan"]),
    ("v2", "SFT v2", TOL["indigo"]),
    ("v3", "SFT v3", TOL["teal"]),
    ("v4", "SFT v4", TOL["sand"]),
    ("v5", "SFT v5", TOL["purple"]),
]
GW = 0.16  # bar / box width per version within a category
OFF = lambda g: (g - (len(GROUPS) - 1) / 2) * GW  # noqa: E731
CAT_TICKS = ["Cyber harmful", "Cyber benign", "Fraud harmful", "Fraud benign", "General benign"]
FAKE_RE = re.compile(r"\bfake", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
BOX = dict(patch_artist=True, medianprops=dict(color="black", lw=0.8),
           whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6),
           flierprops=dict(marker=".", ms=1.5, mfc="#888888", mec="none"))


def load(path, exclude=None):
    """Rows of a combined jsonl; `exclude` drops rows whose messages also appear in that file."""
    key = lambda r: json.dumps(r["messages"], sort_keys=True)  # noqa: E731
    skip = {key(json.loads(l)) for l in exclude.open()} if exclude else set()
    rows = []
    for line in path.open():
        r = json.loads(line)
        if key(r) in skip:
            continue
        msg = {m["role"]: m.get("content") or "" for m in r["messages"]}
        md = r["metadata"]
        cat = "benign_general" if md["attack_type"] == "benign_general_capability" else md["attack_type"]
        rows.append(dict(cat=cat, model=md["generator_model"], topic=md["topic_id"],
                         user=msg["user"], assistant=msg["assistant"]))
    return rows


def distinct2(texts, budget=3000, reps=50, seed=0):
    """Distinct bigrams / bigrams in random samples of whole prompts totalling `budget` words."""
    toks = [[w.lower() for w in WORD_RE.findall(t)] for t in texts]
    rng = random.Random(seed)
    if sum(max(len(t) - 1, 0) for t in toks) < budget:
        return np.nan, np.nan  # too few words: the ratio would not be comparable (it rises as text shrinks)
    out = []
    for _ in range(reps):
        grams = []
        for i in rng.sample(range(len(toks)), len(toks)):
            grams += zip(toks[i], toks[i][1:])
            if len(grams) >= budget:
                break
        out.append(len(set(grams)) / len(grams))
    return np.mean(out), np.std(out)


def semantic_diversity(E):
    n = len(E)
    return 1 - ((E @ E.T).sum(1) - 1) / (n - 1)


def grouped_bars(ax, getter, width=GW):
    for g, (gk, gl, gc) in enumerate(GROUPS):
        x = np.arange(len(CATEGORIES)) + OFF(g)
        ax.bar(x, [getter(gk, c) for c, _, _ in CATEGORIES], width, color=gc, label=gl)
    ax.set_xticks(range(len(CATEGORIES)), CAT_TICKS, rotation=30, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, labelsize=6)


def grouped_boxes(ax, getter, width=GW):
    for g, (gk, gl, gc) in enumerate(GROUPS):
        for i, (c, _, _) in enumerate(CATEGORIES):
            v = getter(gk, c)
            if len(v) < 2:
                continue
            b = ax.boxplot(v, positions=[i + OFF(g)], widths=width * 0.85, **BOX)
            b["boxes"][0].set_facecolor(gc)
    ax.set_xticks(range(len(CATEGORIES)), CAT_TICKS, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(CATEGORIES) - 0.5)
    ax.tick_params(axis="x", length=0, labelsize=6)


def legend_below(fig, y=-0.01):
    fig.legend([Patch(fc=c) for _, _, c in GROUPS],
               [l for _, l, _ in GROUPS], loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(GROUPS))


def fig_composition(G):
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.9))
    panels = [(G[g], l) for g, l, _ in GROUPS] + [([r for g, _, _ in GROUPS for r in G[g]], "All versions")]
    for p, (ax, (rows, label)) in enumerate(zip(axes.ravel(), panels)):
        cnt = Counter(r["cat"] for r in rows)
        spec = [(k, c) for k, _, c in CATEGORIES if cnt[k]]
        vals = [cnt[k] for k, _ in spec]
        wedges, _ = ax.pie(vals, colors=[c for _, c in spec], startangle=90, counterclock=False,
                           wedgeprops=dict(width=0.38, edgecolor="white", linewidth=0.8))
        for w, v in zip(wedges, vals):
            frac = v / len(rows)
            ang = np.deg2rad((w.theta1 + w.theta2) / 2)
            if frac < 0.04:  # slice too thin for an inside label: put it just outside the ring
                ax.text(1.17 * np.cos(ang), 1.17 * np.sin(ang), f"{frac:.0%}", ha="center", va="center",
                        fontsize=6)
                continue
            ax.text(0.81 * np.cos(ang), 0.81 * np.sin(ang), f"{frac:.0%}", ha="center", va="center",
                    fontsize=6, color="white" if w.get_facecolor()[:3] < (0.6, 0.6, 0.6) else "black")
        ax.text(0, 0, f"{len(rows):,}\nprompts", ha="center", va="center", fontsize=6.5)
        ax.set_title(f"({'abcdef'[p]}) {label}", pad=2)
        ax.set_aspect("equal")
    fig.legend([Patch(fc=c) for _, _, c in CATEGORIES], [l for _, l, _ in CATEGORIES],
               loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=5, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.2, w_pad=0.4)
    return fig


def fig_coverage(G):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.2))
    sel = lambda g, c: [r for r in G[g] if r["cat"] == c]  # noqa: E731
    grouped_bars(axes[0], lambda g, c: len(sel(g, c)))
    axes[0].set_ylabel("Rows")
    axes[0].set_title("(a) Rows")
    grouped_bars(axes[1], lambda g, c: len({r["topic"] for r in sel(g, c)}))
    axes[1].set_ylabel("Distinct topics")
    axes[1].set_title("(b) Topic-graph nodes used")
    grouped_bars(axes[2], lambda g, c: 100 * np.mean([bool(FAKE_RE.search(r["user"])) for r in sel(g, c)])
                 if sel(g, c) else np.nan)
    axes[2].set_ylabel("Prompts (%)")
    axes[2].set_title('(c) Prompts containing "fake"')
    legend_below(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.2)
    return fig


def fig_summary(G, E):
    fig = plt.figure(figsize=(WIDTH, 6.3))
    gs = fig.add_gridspec(3, 2, hspace=0.62, wspace=0.28)
    sel = lambda g, c, k: np.array([r[k] for r in G[g] if r["cat"] == c and k in r], float)  # noqa: E731
    rows = lambda g, c: [r for r in G[g] if r["cat"] == c]  # noqa: E731
    ax = fig.add_subplot(gs[0, 0])
    grouped_bars(ax, lambda g, c: len(rows(g, c)))
    ax.set_ylabel("Rows")
    ax.set_title("(a) Rows")
    ax = fig.add_subplot(gs[0, 1])
    grouped_bars(ax, lambda g, c: 100 * np.mean([bool(FAKE_RE.search(r["user"])) for r in rows(g, c)])
                 if rows(g, c) else np.nan)
    ax.set_ylabel("Prompts (%)")
    ax.set_title('(b) Prompts containing "fake"')

    for col, (key, title, ticks) in enumerate([("user_words", "(c) Prompt length", [5, 10, 20, 50, 100]),
                                               ("asst_words", "(d) Response length", [10, 100, 1000])]):
        ax = fig.add_subplot(gs[1, col])
        grouped_boxes(ax, lambda g, c: sel(g, c, key))
        ax.set_yscale("log")
        ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_ylabel("Words")
        ax.set_title(title)

    ax = fig.add_subplot(gs[2, 0])
    for g, (gk, gl, gc) in enumerate(GROUPS):
        for i, (c, _, _) in enumerate(CATEGORIES):
            texts = [r["user"] for r in G[gk] if r["cat"] == c]
            if not texts:
                continue
            m, s = distinct2(texts)
            ax.bar(i + OFF(g), m, GW, yerr=s, color=gc, error_kw=dict(lw=0.6))
    ax.set_xticks(range(len(CATEGORIES)), CAT_TICKS, rotation=30, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, labelsize=6)
    ax.set_ylim(0.35, None)
    ax.set_ylabel("Distinct bigrams")
    ax.set_title("(e) Prompt lexical diversity")

    for gk in E:
        for c, _, _ in CATEGORIES:
            idx = [i for i, r in enumerate(G[gk]) if r["cat"] == c]
            if len(idx) >= 10:  # smaller groups are too noisy to compare (v4 general benign has 3)
                for i, v in zip(idx, semantic_diversity(E[gk][idx])):
                    G[gk][i]["sem_div"] = float(v)
    ax = fig.add_subplot(gs[2, 1])
    grouped_boxes(ax, lambda g, c: sel(g, c, "sem_div"))
    ax.set_ylabel("Mean cosine distance")
    ax.set_title("(f) Prompt semantic diversity")
    legend_below(fig, y=-0.005)
    fig.subplots_adjust(bottom=0.115, top=0.97)
    return fig


def fig_umap(G, Y, seed):
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.3), sharex=True, sharey=True)
    axes = axes.ravel()
    color = {k: c for k, _, c in CATEGORIES}
    rng = np.random.default_rng(seed)
    allY = np.vstack([Y[g] for g, _, _ in GROUPS])
    for p, (ax, (fg, label, _)) in enumerate(zip(axes, GROUPS)):
        others = np.vstack([Y[g] for g, _, _ in GROUPS if g != fg])
        ax.scatter(others[:, 0], others[:, 1], s=2, lw=0, c="#E4E4E4", rasterized=True)
        order = rng.permutation(len(G[fg]))
        ax.scatter(Y[fg][order, 0], Y[fg][order, 1], s=3.5 if len(G[fg]) > 100 else 7, lw=0,
                   c=[color[G[fg][i]["cat"]] for i in order], rasterized=True)
        ax.set_title(f"({'abcde'[p]}) {label}")
    ax = axes[5]
    ccol = np.array([color[r["cat"]] for g, _, _ in GROUPS for r in G[g]])
    order = rng.permutation(len(allY))
    ax.scatter(allY[order, 0], allY[order, 1], s=2, lw=0, c=ccol[order], rasterized=True)
    ax.set_title("(f) All versions")
    for i, ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP-1" if i >= 3 else "")
        ax.set_ylabel("UMAP-2" if i % 3 == 0 else "")
    dot = lambda c: Line2D([], [], ls="", marker="o", ms=3, mfc=c, mec="none")  # noqa: E731
    fig.legend([dot(c) for _, _, c in CATEGORIES] + [dot("#D0D0D0")],
               [l for _, l, _ in CATEGORIES] + ["Other versions"],
               loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=6, columnspacing=0.8, handletextpad=0.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6, h_pad=0.8)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out-dir", type=Path, default=Path("agentharm_blog/figures"))
    p.add_argument("--dpi", type=int, default=600, help="PNG resolution (PDFs are vector)")
    p.add_argument("--seed", type=int, default=42)
    figs = ["composition", "coverage", "summary", "umap"]
    p.add_argument("--only", choices=figs, nargs="+", default=figs)
    a = p.parse_args()

    style.use()
    d = a.data
    G = {
        "v1": load(d / "v1/agentharm_sft_toolcalling.jsonl"),
        "v2": load(d / "v2.1/agentharm_sft_toolcalling_v2.1.jsonl"),
        "v3": load(d / "v3/agentharm_sft_toolcalling_v3.jsonl"),
        "v4": load(d / "v2_v3_v4/agentharm_sft_toolcalling_v2_v3_v4.jsonl",
                   exclude=d / "v2_and_v3/agentharm_sft_toolcalling_v2_and_v3.jsonl"),
        "v5": load(d / "v5/agentharm_sft_toolcalling_v5.jsonl"),
    }
    for rows in G.values():
        for r in rows:
            r["user_words"] = len(r["user"].split())
            r["asst_words"] = len(r["assistant"].split())
    print({k: len(v) for k, v in G.items()})
    out = lambda stem: str(a.out_dir / stem)  # noqa: E731

    if "composition" in a.only:
        style.save(fig_composition(G), out("sft_versions_composition"), a.dpi)
    if "coverage" in a.only:
        style.save(fig_coverage(G), out("sft_versions_coverage"), a.dpi)
    if not {"summary", "umap"} & set(a.only):
        return
    # Embeddings: v1 in dataset order; v2.1 matched by prompt text; v3/v4/v5 in the order of
    # sky_embed/texts_<v>.jsonl, which was written from these same rows (checked below).
    X = {"v1": np.load(EMB / "emb_all-mpnet-base-v2.npy")}
    texts21 = [json.loads(l) for l in open(EMB / "texts_v2.1.jsonl")]
    pos = {t: i for i, t in enumerate(texts21)}
    idx2 = np.array([pos[r["user"]] for r in G["v2"]])
    X["v2"] = np.load(EMB / "emb_all-mpnet-base-v2_v2.1.npy")[idx2]
    for v in ("v3", "v4", "v5"):
        if [json.loads(l) for l in open(EMB / f"texts_{v}.jsonl")] != [r["user"] for r in G[v]]:
            raise SystemExit(f"sky_embed/texts_{v}.jsonl is out of sync with the {v} data; re-run the embed job")
        X[v] = np.load(EMB / f"emb_all-mpnet-base-v2_{v}.npy")
    if "summary" in a.only:
        style.save(fig_summary(G, X), out("sft_versions_summary"), a.dpi)
    if "umap" in a.only:
        Yall = np.load(EMB / f"umap_semantic_v1_to_v5_s{a.seed}.npy")
        sizes = [len(G["v1"]), len(texts21), len(G["v3"]), len(G["v4"]), len(G["v5"])]
        parts = np.split(Yall, np.cumsum(sizes)[:-1])
        Y = {"v1": parts[0], "v2": parts[1][idx2], "v3": parts[2], "v4": parts[3], "v5": parts[4]}
        style.save(fig_umap(G, Y, a.seed), out("sft_versions_umap"), a.dpi)


if __name__ == "__main__":
    main()
