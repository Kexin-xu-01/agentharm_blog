#!/usr/bin/env python3
"""Schematic of a DeepFabric topic graph, publication style (ICML/ICLR/NeurIPS).

Left-to-right tree: the root (seed prompt), every depth-1 topic with its text, and
for a few chosen branches their first depth-2 children with text, then a fan of
dots for the depth-3 leaves beneath each. The remaining branches are shown
collapsed. Topic strings come from the graph JSON; long ones are shortened.

Usage (from /home/jovyan/Shared/kexin):
    python agentharm_blog/plot_topic_tree_schematic.py \
        hedgehog-quills/benchmarks/agentharm/deepfabric/fraud/malicious.yaml \
        --out agentharm_blog/figures/fraud_malicious_topic_tree
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

from style import TOL

TEXTWIDTH_IN = {"neurips": 5.5, "iclr": 5.5, "icml": 6.75}
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8983"
EDGE = "#a9a8a2"
FILL = {0: TOL["indigo"], 1: "#E1EEF6", 2: "#ffffff"}  # depth-1: pale tint of Tol cyan
TEXT_ON = {0: "#ffffff", 1: INK, 2: INK}
LEAF = TOL["teal"]


def set_style(base_pt=7.0):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_pt,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def resolve_graph_path(path, repo_root):
    if path.suffix in (".yaml", ".yml"):
        save_as = Path(yaml.safe_load(path.read_text())["topics"]["save_as"])
        return save_as if save_as.is_absolute() else repo_root / save_as
    return path


def shorten(s, n):
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1].rstrip(" ,.;:") + "…"


def curve(ax, p0, p1, lw=0.5, color=EDGE):
    """Horizontal S-curve (cubic Bezier) from p0 to p1."""
    (x0, y0), (x1, y1) = p0, p1
    xm = x0 + 0.5 * (x1 - x0)
    path = MPath([(x0, y0), (xm, y0), (xm, y1), (x1, y1)],
                 [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw, capstyle="round", zorder=1))


def box(ax, x, y, text, depth, fs):
    return ax.text(x, y, text, ha="left", va="center", fontsize=fs, color=TEXT_ON[depth], zorder=3,
                   bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.2", fc=FILL[depth],
                             ec=FILL[0] if depth else "none", lw=0.4))


def right_edge(t, ax, renderer):
    bb = t.get_bbox_patch().get_window_extent(renderer).transformed(ax.transData.inverted())
    return bb.x1, bb.x0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("graph", type=Path, help="DeepFabric config YAML or graph JSON")
    p.add_argument("--repo-root", type=Path, default=Path("/home/jovyan/Shared/kexin/hedgehog-quills"))
    p.add_argument("--out", type=Path, required=True, help="output stem; writes .pdf and .png")
    p.add_argument("--venue", choices=TEXTWIDTH_IN, default="neurips")
    p.add_argument("--dpi", type=int, default=600, help="PNG resolution (the PDF is vector)")
    p.add_argument("--expand", type=int, nargs="+", default=[0, 5], help="depth-1 branch indices to expand")
    p.add_argument("--n-children", type=int, default=3, help="depth-2 examples shown per expanded branch")
    p.add_argument("--root-label", default=None, help='default: "<category> seed prompt"')
    p.add_argument("--chars1", type=int, default=52, help="max chars for depth-1 text")
    p.add_argument("--chars2", type=int, default=46, help="max chars for depth-2 text")
    a = p.parse_args()

    set_style()
    gpath = resolve_graph_path(a.graph, a.repo_root)
    d = json.loads(gpath.read_text())
    nodes = {int(k): v for k, v in d["nodes"].items()}
    root = int(d.get("root_id", 0))
    kids = {i: [int(c) for c in n.get("children", [])] for i, n in nodes.items()}
    topic = {i: n["topic"] for i, n in nodes.items()}

    # Level sizes for the column headers.
    level, frontier = [1], [root]
    while True:
        frontier = [c for u in frontier for c in kids[u]]
        if not frontier:
            break
        level.append(len(frontier))

    # Rows: a collapsed branch takes one row; an expanded one takes n_children + 1.
    branches = kids[root]
    expand = {i for i in a.expand if 0 <= i < len(branches)}
    rows, y = [], 0.0
    for i, b in enumerate(branches):
        if i in expand:
            shown = kids[b][: a.n_children]
            ys = [y + j for j in range(len(shown) + (len(kids[b]) > len(shown)))]
            rows.append((b, float(np.mean(ys)), list(zip(shown, ys)), len(kids[b]) - len(shown), ys[-1]))
            y = ys[-1] + 1.25
        else:
            rows.append((b, y, [], len(kids[b]), None))
            y += 1.0
    n_rows = y - 0.25

    w = TEXTWIDTH_IN[a.venue]
    fig, ax = plt.subplots(figsize=(w, 0.19 * n_rows + 0.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(n_rows - 0.4, -1.3)
    ax.axis("off")
    fs1, fs2 = 6.2, 5.8
    X0, X1, X2, X3 = 0.0, 0.125, 0.545, 0.955

    label = a.root_label or f"{gpath.parent.name.capitalize()} seed prompt"
    root_y = (rows[0][1] + rows[-1][1]) / 2
    t_root = box(ax, X0, root_y, label.replace(" ", "\n", 1), 0, fs1)
    t_root.set_ha("left")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    root_x1, _ = right_edge(t_root, ax, renderer)

    for b, yb, shown, rest, y_more in rows:
        t1 = box(ax, X1, yb, shorten(topic[b], a.chars1), 1, fs1)
        fig.canvas.draw()
        x1r, x1l = right_edge(t1, ax, renderer)
        curve(ax, (root_x1 + 0.004, root_y), (x1l - 0.002, yb), lw=0.6)
        if not shown:  # collapsed: stub + child count
            ax.plot([x1r + 0.006, x1r + 0.03], [yb, yb], color=EDGE, lw=0.5, zorder=1)
            ax.text(x1r + 0.034, yb, f"{rest} subtopics", fontsize=5, color=INK_3, va="center")
            continue
        for c, yc in shown:
            t2 = box(ax, X2, yc, shorten(topic[c], a.chars2), 2, fs2)
            fig.canvas.draw()
            x2r, x2l = right_edge(t2, ax, renderer)
            curve(ax, (x1r + 0.004, yb), (x2l - 0.002, yc), lw=0.45)
            # Fan of leaves under this depth-2 node.
            nl = len(kids[c])
            if nl:
                ly = yc + np.linspace(-0.42, 0.42, nl)
                for yy in ly:
                    curve(ax, (x2r + 0.004, yc), (X3, yy), lw=0.25)
                ax.scatter(np.full(nl, X3 + 0.004), ly, s=1.6, c=LEAF, lw=0, zorder=3)
                ax.text(X3 + 0.012, yc, f"{nl}", fontsize=5, color=INK_3, va="center")
        if rest > 0:
            curve(ax, (x1r + 0.004, yb), (X2 - 0.002, y_more), lw=0.45, color="#d3d2cc")
            ax.text(X2 + 0.004, y_more, f"… {rest} more subtopics", fontsize=5.2,
                    color=INK_3, va="center", style="italic")

    heads = [(X0, "Root"), (X1, f"Depth 1  ({level[1]:,})")]
    if len(level) > 2:
        heads.append((X2, f"Depth 2  ({level[2]:,})"))
    if len(level) > 3:
        heads.append((1.0, f"Depth 3  ({level[3]:,})"))
    for x, h in heads:
        ax.text(x, -1.05, h, fontsize=6.3, color=INK_2, ha="right" if x >= 1 else "left", va="center", fontweight="bold")
    ax.plot([0, 1], [-0.72, -0.72], color="#dcdbd5", lw=0.4)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(a.out.with_suffix(f".{ext}"), dpi=a.dpi)
    print(f"graph: {gpath}\nlevels={level}\nwrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
