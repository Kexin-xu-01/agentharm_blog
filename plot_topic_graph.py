#!/usr/bin/env python3
"""Publication-style (ICML/ICLR/NeurIPS) figure of a DeepFabric topic graph's structure.

Input: a DeepFabric config YAML (the graph path is read from `topics.save_as`) or
the graph JSON directly. Output: a two-panel vector figure.

  (a) Radial dendrogram. Root at centre, one ring per depth level, alternating
      shaded sectors numbered by depth-1 branch. Leaves are coloured by lexical
      redundancy = max TF-IDF cosine similarity to any other leaf (1.0 = duplicate).
  (b) Per-branch redundancy distribution (strip + median), same branch numbering,
      with graph-wide exact/near-duplicate rates in the corner.

Only structure and similarity statistics are drawn; topic strings are not rendered.

Usage (from /home/jovyan/Shared/kexin):
    python agentharm_blog/plot_topic_graph.py \
        hedgehog-quills/benchmarks/agentharm/deepfabric/fraud/malicious.yaml \
        --out agentharm_blog/figures/fraud_malicious_topic_graph
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import LineCollection
from matplotlib.patches import Wedge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEXTWIDTH_IN = {"neurips": 5.5, "iclr": 5.5, "icml": 6.75}
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
EDGE, SECTOR = "#bdbcb6", "#f1f0ec"
CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def set_style(base_pt=7.5):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_pt, "axes.titlesize": base_pt + 0.5, "axes.labelsize": base_pt,
        "xtick.labelsize": base_pt - 1, "ytick.labelsize": base_pt - 1,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def resolve_graph_path(path, repo_root):
    if path.suffix in (".yaml", ".yml"):
        save_as = Path(yaml.safe_load(path.read_text())["topics"]["save_as"])
        return save_as if save_as.is_absolute() else repo_root / save_as
    return path


def load_graph(path):
    d = json.loads(path.read_text())
    nodes = {int(k): v for k, v in d["nodes"].items()}
    root = int(d.get("root_id", 0))
    topics = {i: n["topic"] for i, n in nodes.items()}
    # Graph mode allows multiple parents: lay out a BFS spanning tree, keep the rest as cross-links.
    depth, kids, order, cross = {root: 0}, {i: [] for i in nodes}, [root], []
    for u in order:
        for c in map(int, nodes[u].get("children", [])):
            if c in depth:
                cross.append((u, c))
            else:
                depth[c] = depth[u] + 1
                kids[u].append(c)
                order.append(c)
    return root, topics, kids, depth, cross


def leaves_of(u, kids):
    out, stack = [], [u]
    while stack:
        v = stack.pop()
        if kids[v]:
            stack.extend(reversed(kids[v]))
        else:
            out.append(v)
    return out


def redundancy(leaf_ids, topics):
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(
        [" ".join(topics[i].lower().split()) for i in leaf_ids])
    S = cosine_similarity(X)
    np.fill_diagonal(S, -1)
    return dict(zip(leaf_ids, S.max(axis=1).clip(0, 1)))


def polar(r, t):  # clockwise from 12 o'clock
    return r * np.sin(t), r * np.cos(t)


def draw_radial(ax, root, kids, depth, cross, red, norm):
    leaves = leaves_of(root, kids)
    n = len(leaves)
    theta = {l: 2 * np.pi * (k + 0.5) / n for k, l in enumerate(leaves)}

    def place(u):
        if u not in theta:
            theta[u] = float(np.mean([place(c) for c in kids[u]]))
        return theta[u]

    place(root)
    max_d = max(depth.values())
    half = np.pi / n

    for k, b in enumerate(kids[root]):
        bl = leaves_of(b, kids)
        t0, t1 = theta[bl[0]] - half, theta[bl[-1]] + half
        if k % 2 == 0:
            ax.add_patch(Wedge((0, 0), max_d + 0.6, 90 - np.degrees(t1), 90 - np.degrees(t0),
                               width=max_d + 0.1, facecolor=SECTOR, edgecolor="none", zorder=0))
        ax.text(*polar(max_d + 0.85, theta[b]), str(k + 1), ha="center", va="center",
                fontsize=6.5, color=INK)

    segs, widths = [], []
    for u, cs in kids.items():
        if not cs:
            continue
        lw = {0: 0.8, 1: 0.55}.get(depth[u], 0.12)
        r_mid = depth[u] + 0.5
        ts = [theta[c] for c in cs]
        arc = np.linspace(min(ts), max(ts), max(6, int(80 * (max(ts) - min(ts)))))
        if u != root:
            segs.append(np.array([polar(depth[u], theta[u]), polar(r_mid, theta[u])]))
            widths.append(lw)
        segs.append(np.column_stack(polar(r_mid, arc)))
        widths.append(lw)
        for c in cs:
            segs.append(np.array([polar(r_mid, theta[c]), polar(depth[c], theta[c])]))
            widths.append(lw)
    ax.add_collection(LineCollection(segs, colors=EDGE, linewidths=widths, capstyle="round", zorder=1))
    if cross:
        ax.add_collection(LineCollection(
            [np.array([polar(depth[a], theta[a]), polar(depth[b], theta[b])]) for a, b in cross],
            colors=INK_3, linewidths=0.3, linestyles=(0, (2, 1)), zorder=1))

    internal = [u for u in kids if kids[u]]
    ax.scatter(*polar(np.array([depth[u] for u in internal], float), np.array([theta[u] for u in internal])),
               s=[{0: 12, 1: 6}.get(depth[u], 2.5) for u in internal],
               c="white", edgecolors=INK_2, linewidths=0.4, zorder=3)
    ls = sorted(red, key=red.get)
    sc = ax.scatter(*polar(np.array([depth[u] for u in ls], float), np.array([theta[u] for u in ls])),
                    s=2.6, c=[red[u] for u in ls], cmap=CMAP, norm=norm, linewidths=0, zorder=4)
    for d in range(1, max_d + 1):
        ax.text(0.04, d + 0.02, f"$d={d}$", fontsize=5, color=INK_3, ha="left", va="bottom", zorder=5)
    lim = max_d + 1.05
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
    ax.axis("off")
    return sc


def draw_strip(ax, root, kids, red, norm, near):
    rng = np.random.default_rng(0)
    for k, b in enumerate(kids[root]):
        v = np.array([red[l] for l in leaves_of(b, kids)])
        y = k + rng.uniform(-0.28, 0.28, len(v))
        ax.scatter(v, y, s=2.2, c=v, cmap=CMAP, norm=norm, linewidths=0, zorder=2)
        med = np.median(v)
        ax.plot([med, med], [k - 0.38, k + 0.38], color=INK, lw=0.9, solid_capstyle="butt", zorder=3)
    ax.axvline(near, color=INK_3, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
    ax.text(near + 0.01, -0.55, f"near-dup. $\\geq${near:.1f}", fontsize=5,
            color=INK_3, ha="left", va="top")
    ax.set_yticks(range(len(kids[root])), [str(k + 1) for k in range(len(kids[root]))])
    ax.set_ylim(len(kids[root]) - 0.4, -0.9)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("max cosine sim. to any other leaf")
    ax.set_ylabel("depth-1 branch")
    ax.grid(axis="x", color="#e6e5e0", lw=0.4, zorder=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("graph", type=Path, help="DeepFabric config YAML or graph JSON")
    p.add_argument("--repo-root", type=Path, default=Path("/home/jovyan/Shared/kexin/hedgehog-quills"))
    p.add_argument("--out", type=Path, required=True, help="output stem; writes .pdf and .png")
    p.add_argument("--venue", choices=TEXTWIDTH_IN, default="neurips")
    p.add_argument("--near-dup", type=float, default=0.8, help="near-duplicate threshold")
    p.add_argument("--title", default=None, help="optional suptitle (usually omit; use the LaTeX caption)")
    a = p.parse_args()

    set_style()
    path = resolve_graph_path(a.graph, a.repo_root)
    root, topics, kids, depth, cross = load_graph(path)
    leaves = leaves_of(root, kids)
    red = redundancy(leaves, topics)
    norm = mpl.colors.Normalize(0, 1)

    w = TEXTWIDTH_IN[a.venue]
    fig = plt.figure(figsize=(w, w * 0.47))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.28)
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    sc = draw_radial(ax_a, root, kids, depth, cross, red, norm)
    draw_strip(ax_b, root, kids, red, norm, a.near_dup)

    cax = ax_a.inset_axes([0.08, -0.06, 0.84, 0.03])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5)
    cb.set_label("leaf redundancy", fontsize=6, labelpad=1)

    vals = np.array(list(red.values()))
    n_nodes = len(depth)
    ax_a.set_title(f"(a) Topic tree ($N$={n_nodes:,}, {len(leaves):,} leaves)",
                   loc="left", pad=2)
    ax_b.set_title(f"(b) Leaf redundancy by branch", loc="left", pad=2)
    ax_b.text(0.99, 0.02, f"exact dup. {np.mean(vals >= 0.999):.1%}\nnear dup. {np.mean(vals >= a.near_dup):.1%}",
              transform=ax_b.transAxes, ha="right", va="bottom", fontsize=5.5, color=INK_2,
              bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none"))
    if a.title:
        fig.suptitle(a.title, y=1.02)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(a.out.with_suffix(f".{ext}"), dpi=300)
    print(f"graph: {path}\nnodes={n_nodes} leaves={len(leaves)} cross_links={len(cross)} "
          f"exact={np.mean(vals >= 0.999):.3f} near={np.mean(vals >= a.near_dup):.3f}\n"
          f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
