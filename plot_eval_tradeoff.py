#!/usr/bin/env python3
"""AgentHarm safety / capability trade-off: harmful refusal rate (x) vs benign score (y).

One panel per prompt mode (neutral, refusal prompt, jailbreak), with every hosted model and the three
Qwen3-14B rows of plot_recent_runs_eval.py --eval-all / --selected (same sources, same test_public
Fraud + Cybercrime set, 44 harmful + 44 benign tasks). Hosted models are open circles, the Qwen rows
filled in their SFT version colour. Top right is better (refuses harmful tasks, still completes
benign ones).

Usage (from /home/jovyan/Shared/kexin):
    python agentharm_blog/plot_eval_tradeoff.py --out-dir agentharm_blog/figures              # -> eval_tradeoff_all
    python agentharm_blog/plot_eval_tradeoff.py --selected --out-dir agentharm_blog/figures   # -> eval_tradeoff_selected
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_recent_runs_eval as ev
import style

CF_COLOR = style.TOL["rose"]  # --selected: hosted models whose API runs a content filter
MODES = [("neutral", "Neutral"), ("refusal", "Refusal prompt"), ("jailbreak", "Jailbreak")]
QWEN_LABEL = {"Base Qwen3-14B": "Base Qwen3-14B", "SFT-v2 + RL": "Qwen3 SFT-v2 + RL", "SFT-v3 + RL": "Qwen3 SFT-v3 + RL"}
# label positions tried in order: rings of growing radius (points) around the marker; the first spot
# free of other labels and markers wins, labels away from their marker get a leader line
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
OFFSETS = [(r * dx, r * dy * 0.8) for r in (4, 10, 17, 25, 34, 45) for dx, dy in DIRS]

# dense clusters: labels stacked in a column (data coords, horizontal alignment) with leader lines
COLUMN = {
    "refusal": (0.82, "right", [("GPT-4o-mini", 0.701), ("GPT-5.5", 0.674), ("Gemini 3.8 Flash", 0.640)]),
    "jailbreak": (0.16, "left", [("Base Qwen3-14B", 0.745), ("GPT-4o-mini", 0.724),
                                 ("Qwen3 SFT-v3 + RL", 0.665), ("Gemini 2.5 Pro", 0.644)]),
}


def points(mode, selected):
    hosted = ev.HOSTED_MODES[mode]
    if selected:
        hosted = [m for m in hosted if m[0] not in ev.SELECTED_DROP]
    out = [(l, h, b, ev.FRONTIER_COLOR) for l, h, b in hosted]
    for l, _, h, b, c, _ in (r for r in ev.RUNS if r[0] in ev.FRONTIER_QWEN):
        if mode != "neutral":
            h, b = ev.QWEN_MODES[mode][l]
        out.append((QWEN_LABEL[l], h, b, c))
    rows = []
    for l, h, b, c in out:
        H, B = ev.results(h), ev.results(b)
        assert len(H) == len(B) == 44
        rows.append(dict(label=l, col=c, x=np.mean([float(r["refused"]) for r in H]),
                         y=np.mean([float(r["reward"]) for r in B])))
    return rows


def place_labels(ax, rows, fig, mode):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    marks = [ax.transData.transform((r["x"], r["y"])) for r in rows]
    taken = []
    axbox = ax.get_window_extent(rend)
    fixed = set()
    if mode in COLUMN:
        cx, ha, slots = COLUMN[mode]
        names = [r["label"] for r in rows]
        for lab, cy in slots:
            if lab not in names:
                continue
            r = rows[names.index(lab)]
            t = ax.annotate(lab, (r["x"], r["y"]), xytext=(cx, cy), textcoords="data", ha=ha, va="center", fontsize=5.5,
                            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.35, shrinkA=1, shrinkB=2.5))
            taken.append(t.get_window_extent(rend).expanded(1.04, 1.15))
            fixed.add(lab)

    lines = []  # sampled points of the leader lines drawn so far
    segs = []  # leader lines drawn so far, (start, end) in display coords

    def crosses(p1, p2, q1, q2):
        d = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return d(p1, p2, q1) * d(p1, p2, q2) < 0 and d(q1, q2, p1) * d(q1, q2, p2) < 0

    def cost(bb, i, dx, dy):
        c = sum(5e3 for o in taken if bb.overlaps(o))
        c += sum(3e3 for j, m in enumerate(marks)
                 if j != i and bb.x0 - 4 < m[0] < bb.x1 + 4 and bb.y0 - 4 < m[1] < bb.y1 + 4)
        c += sum(1e3 for p in lines if bb.x0 < p[0] < bb.x1 and bb.y0 < p[1] < bb.y1)
        # ambiguity: the label's attachment point must be clearly closest to its own marker
        a = marks[i] + np.array([dx, dy]) * fig.dpi / 72
        own = np.hypot(*(a - marks[i]))
        c += sum(200 for j, m in enumerate(marks) if j != i and np.hypot(*(a - m)) < own + 6 * fig.dpi / 72)
        if abs(dx) > 6 or abs(dy) > 6:  # leader line must not cross labels or markers
            seg = [marks[i] + f * (a - marks[i]) for f in np.linspace(0.15, 0.95, 12)]
            c += sum(2e3 for p in seg for o in taken if o.x0 < p[0] < o.x1 and o.y0 < p[1] < o.y1)
            c += sum(1e3 for p in seg for j, m in enumerate(marks) if j != i and np.hypot(*(p - m)) < 5 * fig.dpi / 72)
            c += sum(3e3 for q1, q2 in segs if crosses(marks[i], a, q1, q2))
            c += 3 * np.hypot(dx, dy)  # prefer short leaders
        return c

    for i in sorted(range(len(rows)), key=lambda i: (-rows[i]["x"], -rows[i]["y"])):
        r = rows[i]
        if r["label"] in fixed:
            continue
        best = None
        for dx, dy in OFFSETS:
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
            t = ax.annotate(r["label"], (r["x"], r["y"]), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=5.5)
            bb = t.get_window_extent(rend).expanded(1.04, 1.15)
            t.remove()
            if not (axbox.x0 <= bb.x0 and bb.x1 <= axbox.x1 and axbox.y0 <= bb.y0 and bb.y1 <= axbox.y1):
                continue
            c = cost(bb, i, dx, dy)
            if best is None or c < best[0]:
                best = (c, dx, dy, ha, va, bb)
        _, dx, dy, ha, va, bb = best
        far = abs(dx) > 6 or abs(dy) > 6
        ax.annotate(r["label"], (r["x"], r["y"]), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    fontsize=5.5, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.35, shrinkA=0, shrinkB=2.5) if far else None)
        if far:
            a = marks[i] + np.array([dx, dy]) * fig.dpi / 72
            lines += [marks[i] + f * (a - marks[i]) for f in np.linspace(0.1, 1, 12)]
            segs.append((marks[i], a))
        taken.append(bb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="agentharm_blog/figures")
    ap.add_argument("--selected", action="store_true", help="drop SELECTED_DROP (-> eval_tradeoff_selected)")
    args = ap.parse_args()
    style.use()
    fig, axes = plt.subplots(3, 1, figsize=(style.WIDTH, 8.4), sharex=True)
    for ax, (mode, title) in zip(axes, MODES):
        rows = points(mode, args.selected)
        for r in rows:
            hosted = r["col"] == ev.FRONTIER_COLOR
            col = CF_COLOR if args.selected and r["label"] in ev.CONTENT_FILTER else r["col"]
            ax.plot(r["x"], r["y"], marker="o", ms=4 if hosted else 5, mfc=col, mec="black", mew=0.5, ls="",
                    zorder=3)
            print(f"{mode:9s} {r['label']:22s} refusal {r['x']:.0%}  benign {r['y']:.3f}")
        ax.set_xlim(-0.04, 1.17)  # room for labels right of the 90% points
        ax.spines["bottom"].set_bounds(0, 1)
        ax.set_ylim(0.28, 0.87)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"])
        ax.grid(color=style.TOL["pale"], lw=0.4)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Benign score ↑")
        place_labels(ax, rows, fig, mode)
    axes[-1].set_xlabel("Harmful refusal rate ↑")
    if args.selected:
        mk = dict(marker="o", ls="", mec="black", mew=0.5)
        handles = [plt.Line2D([], [], ms=4, mfc=CF_COLOR, label="Hosted, provider content filter", **mk),
                   plt.Line2D([], [], ms=4, mfc="white", label="Hosted, no filter", **mk),
                   plt.Line2D([], [], ms=5, mfc=style.TOL["indigo"], label="Qwen3-14B (ours, no filter)", **mk)]
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(h_pad=0.8, rect=(0, 0.025 if args.selected else 0, 1, 1))
    for ax, (mode, _) in zip(axes, MODES):  # re-place after tight_layout moved the axes
        for t in ax.texts[:]:
            t.remove()
        place_labels(ax, points(mode, args.selected), fig, mode)
    style.save(fig, str(Path(args.out_dir) / f"eval_tradeoff_{'selected' if args.selected else 'all'}"))


if __name__ == "__main__":
    main()
