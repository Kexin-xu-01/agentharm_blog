#!/usr/bin/env python3
"""AgentHarm results per harm category (test_public, all 8 categories, neutral prompt).

Qwen3-14B rows: Fraud + Cybercrime from the Hedgerow evals in plot_recent_runs_eval.RUNS (refusal-judge
v3), the other six categories from the 2026-10-24 '-othercat' runs in agentharm-inspect-eval/ (same
checkpoints, refusal-judge v5, no judge or sample errors). Frontier rows: per-sample results of
hedgehog-quills/benchmarks/agentharm/run_analysis/agentharm_frontier_models_full_8cat.jsonl (same
neutral setup; Fraud + Cybercrime are the runs used in the other figures). 20-24 samples per category
and split. Fraud and Cybercrime are the training domains of the SFT / GRPO data; the other six are
held out.

Figures:
  eval_categories_qwen      grouped bars, base + best checkpoints (QWEN + QWEN_ONLY), 4 metrics
  eval_categories_frontier  heatmaps, QWEN + all hosted models (FRONTIER + OLD), 4 metrics
  eval_categories_stacked   harmful / benign refusal rate of every model as one bar, stacked by the
                            share of each category (segment = refusals in that category / 176)
  eval_categories_stacked_selected  same without SELECTED_DROP and SFT-v3, ranked by harmful refusal,
                            our checkpoints in bold, * = provider content filter
  eval_categories_stacked_heldout  same, six held-out categories only (segment = refusals / 132)
  eval_categories_qwen_rl    eval_categories_qwen without the SFT-v3 row
  eval_categories_{heat,stacked,stacked_heldout}_qwen  the same, Qwen rows only (base + SFT + RL, no SFT-v3)
  SFT-v5 + RL (QWEN_ONLY, 2026-10-25) is on the Qwen-only figures, not yet next to the hosted models
  eval_categories_range     harmful and benign refusal per category: range of the hosted models with /
                            without a provider content filter vs base Qwen3-14B and the best checkpoint

Usage (from /home/jovyan/Shared/kexin):
    python agentharm_blog/plot_eval_categories.py --out-dir agentharm_blog/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import plot_recent_runs_eval as ev
import style

CATS = ["Fraud", "Cybercrime", "Copyright", "Disinformation", "Drugs", "Harassment", "Hate", "Sexual"]
TRAIN_CATS = 2  # the first two are the SFT / GRPO training domains
OTHERCAT = {  # RUNS label -> (agentharm-inspect-eval dir, run substring)
    "Base Qwen3-14B": ("base-untrained-qwen3-14b-othercat", "vllm/Qwen/Qwen3-14B"),
    "SFT-v3": ("agentharm-sft-toolcalling-v2-v3-othercat", "_100-0"),
    "SFT-v2 + RL": ("agentharm-sft-grpo-v2.1-llmjudge-othercat", "_99-0"),
    "SFT-v3 + RL": ("agentharm-sft-grpo-v2-v3-lmmjudge-othercat", "_103-0"),
    "SFT-v5 + RL": ("agentharm-sft-grpo-v2-v3-v4-v5-llmjudge-othercat", "_164-0"),
}
QWEN = ["Base Qwen3-14B", "SFT-v3", "SFT-v2 + RL", "SFT-v3 + RL"]  # base + best checkpoints (all-8 refusal)
QWEN_LABEL = {"Base Qwen3-14B": "Base Qwen3-14B", "SFT-v3": "Qwen3-14B SFT-v3",
              "SFT-v2 + RL": "Qwen3-14B SFT-v2 + RL", "SFT-v3 + RL": "Qwen3-14B SFT-v3 + RL",
              "SFT-v5 + RL": "Qwen3-14B SFT-v5 + RL"}
QWEN_ONLY = ["SFT-v5 + RL"]  # 2026-10-25 checkpoint, shown on the Qwen-only figures
FRONTIER_JSONL = (ev.LOGS_ROOT / "hedgehog-quills/benchmarks/agentharm/run_analysis/"
                  "agentharm_frontier_models_full_8cat.jsonl")
# hosted models in the 8-category report (jsonl)
FRONTIER = [("gpt-6.1-sol", "GPT-6.1 Sol"), ("gpt-5.5", "GPT-5.5"), ("gemini-4-argon", "Gemini 4 Argon"),
            ("gemini-3.8-flash", "Gemini 3.8 Flash"), ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
            ("claude-sonnet-5", "Claude Sonnet 5")]
# older hosted models: Fraud + Cybercrime from the refusal-judge v3 logs (no content-filter blocks, so v3 ==
# v5 for them), the other six categories from the 2026-10-24 judge-v5 '-othercat' runs
OLD = [(m, l) for l, m in ev.OLD_HOSTED if l != "Claude Sonnet 5"]
# models whose benign othercat run is missing: only Fraud + Cybercrime, other cells blank (none at present)
NO_BENIGN_OTHERCAT = []
METRICS = [("ref", "harm", "(a) Harmful refusal rate ↑", True), ("score", "harm", "(b) Harm score ↓", False),
           ("ref", "ben", "(c) Benign refusal rate ↓", True), ("score", "ben", "(d) Benign score ↑", False)]


def qwen_samples(label):
    run = next(r for r in ev.RUNS if r[0] == label)
    d, m = OTHERCAT[label]
    out = {}
    for split, src, task in (("harm", run[2], "agentharm"), ("ben", run[3], "agentharm_benign")):
        rs = ev.results(src) + ev.results((ev.Q + d, m, task))
        out[split] = [dict(cat=r["attack_type"], ref=float(r["refused"]), score=float(r["reward"])) for r in rs]
        assert len(out[split]) == 176
    return out


def frontier_samples():
    out = {k: {"harm": [], "ben": []} for k, _ in FRONTIER}
    for line in open(FRONTIER_JSONL):
        d = json.loads(line)
        if d["model"] in out:
            split = "ben" if d["task"] == "agentharm_benign" else "harm"
            out[d["model"]][split].append(dict(cat=d["category"], ref=float(d["refusal"]), score=float(d["score"])))
    for k in out:
        assert len(out[k]["harm"]) == len(out[k]["ben"]) == 176, k
    return out


def old_samples(model):
    out = {}
    for split, task in (("harm", "agentharm"), ("ben", "agentharm_benign")):
        rs = ev.results((ev.J + "patched-judge-v3", model, task))
        if not (split == "ben" and model in NO_BENIGN_OTHERCAT):
            rs = rs + ev.results((ev.J + "patched-judge-v5-othercat-neutral", model, task))
        out[split] = [dict(cat=r["attack_type"], ref=float(r["refused"]), score=float(r["reward"])) for r in rs]
        assert len(out[split]) == (44 if split == "ben" and model in NO_BENIGN_OTHERCAT else 176), (model, split)
    return out


def table(samples):
    """{(metric, split): array over CATS + ['All']}"""
    t = {}
    for metric, split, _, _ in METRICS:
        rs = samples[split]
        per = [[r[metric] for r in rs if r["cat"] == c] for c in CATS]
        t[metric, split] = np.array([np.mean(v) if v else np.nan for v in per]
                                    + [np.mean([r[metric] for r in rs]) if all(per) else np.nan])  # 'All' only if complete
        # each category's contribution to the overall rate (sums to the 'All' column)
        t["share", metric, split] = np.array([sum(v) / len(rs) if all(per) else np.nan for v in per])
        # same over the six held-out categories only
        held = per[TRAIN_CATS:]
        n_held = sum(len(v) for v in held)
        t["held", metric, split] = sum(map(sum, held)) / n_held if all(held) else np.nan
        t["held share", metric, split] = np.array([sum(v) / n_held if all(held) else np.nan for v in held])
    return t


def qwen_figure(rows, out):
    colors = {"Base Qwen3-14B": "#555555", "SFT-v3": style.TOL["teal"],
              "SFT-v2 + RL": style.TOL["indigo"], "SFT-v3 + RL": style.TOL["teal"],
              "SFT-v5 + RL": style.TOL["purple"]}
    x = np.arange(len(CATS) + 2) + np.r_[np.zeros(len(CATS)), 0.4, 0.4]  # gap before 'Held-out 6', 'All 8'
    w = 0.8 / len(rows)
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 3.9), sharex=True)
    for ax, (metric, split, title, pct) in zip(axes.flat, METRICS):
        for i, (label, t) in enumerate(rows):
            grpo = label.endswith("+ RL")
            v = np.r_[t[metric, split][:-1], t["held", metric, split], t[metric, split][-1]]
            ax.bar(x + (i - (len(rows) - 1) / 2) * w, v, w, color=colors[label],
                   edgecolor="black" if grpo else "white", lw=0.4, hatch="////" if grpo else None, alpha=0.9)
        ax.axvline(TRAIN_CATS - 0.5, color="black", lw=0.5, ls=(0, (2, 2)))
        ax.axvline(len(CATS) - 0.3, color=style.TOL["pale"], lw=0.6)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25", "50", "75", "100%"] if pct else ["0", ".25", ".5", ".75", "1"])
        ax.grid(axis="y", color=style.TOL["pale"], lw=0.4)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left")
        ax.tick_params(axis="x", length=0)
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(CATS + ["Held-out 6", "All 8"], rotation=30, ha="right")
    axes[0, 0].text(TRAIN_CATS - 0.45, 0.97, "held out →", fontsize=6, va="top")
    handles = [Patch(facecolor=colors[l], edgecolor="black" if l.endswith("+ RL") else "white", lw=0.4,
                     hatch="////" if l.endswith("+ RL") else None, label=QWEN_LABEL[l]) for l, _ in rows]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=1.0)
    style.save(fig, out)


def heat_figure(rows, out, n_hosted=None):
    cols = CATS + ["Held-out 6", "All 8"]
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 0.22 * len(rows) + 1.05), sharey=True)
    for ax, (metric, split, title, pct) in zip(axes, METRICS):
        m = np.array([np.r_[t[metric, split][:-1], t["held", metric, split], t[metric, split][-1]] for _, t in rows])
        cmap = plt.get_cmap("YlOrBr").copy()
        cmap.set_bad("0.85")  # not run
        ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(m):
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v * 100:.0f}" if pct else f"{v:.2f}"[1:] if v < 1 else "1", ha="center", va="center",
                    fontsize=4.6, color="white" if v > 0.6 else "black")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=5.5)
        ax.axvline(TRAIN_CATS - 0.5, color="black", lw=0.8)
        ax.axvline(len(CATS) - 0.5, color="white", lw=1.5)
        ax.axvline(len(CATS) + 0.5, color="white", lw=0.8)
        if n_hosted:
            ax.axhline(n_hosted - 0.5, color="white", lw=1.5)
        ax.set_title(title + (" (%)" if pct else ""), loc="left", fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([QWEN_LABEL.get(l, l) for l, _ in rows], fontsize=6)
    fig.tight_layout(w_pad=0.5)
    style.save(fig, out)


CAT_COLORS = dict(zip(CATS, [style.TOL[c] for c in
                              ("wine", "indigo", "sand", "olive", "green", "teal", "cyan", "purple")]))


def stacked_figure(rows, n_hosted, out, held=False, bold=(), star=()):
    """One bar per model = overall refusal rate, split into each category's contribution.
    held: only the six held-out categories (bar = held-out refusal rate). n_hosted: rows after it are
    set off by a gap (None: no gap). bold: row labels in bold; star: rows marked * (content filter)."""
    cats = CATS[TRAIN_CATS:] if held else CATS
    key = "held share" if held else "share"
    n = len(rows)
    y = np.arange(n)[::-1].astype(float)
    if n_hosted is not None:
        y[n_hosted:] -= 0.6  # gap between hosted models and Qwen rows
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, max(0.19 * n + 1.0, 1.9)), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.0, 0.45]))
    for ax, split, title, top in ((axes[0], "harm", "(a) Harmful refusal rate ↑", 1.0),
                                  (axes[1], "ben", "(b) Benign refusal rate ↓", 0.2)):
        title += " (held-out 6)" if held and split == "harm" else ""
        for yi, (label, t) in zip(y, rows):
            sh = t[key, "ref", split]
            if np.isnan(sh).any():
                ax.text(0.005, yi, "not run", va="center", fontsize=5.5, color="0.4")
                continue
            ax.barh(yi, sh, left=np.r_[0, np.cumsum(sh)[:-1]], height=0.72,
                    color=[CAT_COLORS[c] for c in cats], edgecolor="white", lw=0.3)
            ax.text(sh.sum() + 0.012 * top / 0.5, yi, f"{sh.sum():.0%}", va="center", fontsize=5.5)
        ax.set_xlim(0, top)
        ticks = np.arange(0, top + 1e-9, 0.25 if top == 1 else 0.1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v * 100:.0f}" for v in ticks[:-1]] + [f"{ticks[-1]:.0%}"])
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=style.TOL["pale"], lw=0.4)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([QWEN_LABEL.get(l, l) + ("*" if l in star else "") for l, _ in rows])
    for tl, (l, _) in zip(axes[0].get_yticklabels(), rows):
        if l in bold:
            tl.set_fontweight("bold")
    axes[0].set_ylim(y.min() - 0.6, y.max() + 0.6)
    handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in cats]
    if star:
        handles.append(plt.Line2D([], [], ls="", label="* provider content filter"))
    fig.legend(handles=handles, loc="lower center", ncol=3 if held else 5 if star else 4, bbox_to_anchor=(0.5, 0.0), fontsize=6,
               handlelength=1.0, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.3 / fig.get_figheight(), 1, 1), w_pad=0.8)
    style.save(fig, out)


def range_figure(hosted, qwen, out):
    """Per category: range over hosted models with / without a provider content filter vs the Qwen base
    and best checkpoint. (a) harmful refusal rate, (b) benign refusal rate."""
    cols = CATS + ["Held-out 6", "All 8"]
    y = np.arange(len(cols))[::-1].astype(float)
    y[-2:] -= 0.4
    groups = (([t for l, t in hosted if l not in ev.CONTENT_FILTER], "0.82", 7, "Hosted, no content filter (range)"),
              ([t for l, t in hosted if l in ev.CONTENT_FILTER], style.TOL["rose"], 3.2,
               "Hosted, provider content filter (range)"))
    mk = {"Base Qwen3-14B": dict(marker="o", mfc="white"), "SFT-v3 + RL": dict(marker="D", mfc=style.TOL["teal"])}
    vals = lambda t, split: np.r_[t["ref", split][:-1], t["held", "ref", split], t["ref", split][-1]]
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 3.0), sharey=True, gridspec_kw=dict(width_ratios=[1, 0.6]))
    for ax, split, title, top in ((axes[0], "harm", "(a) Harmful refusal rate ↑", 1.0),
                                  (axes[1], "ben", "(b) Benign refusal rate ↓", 0.4)):
        for ts, col, lw, lab in groups:
            m = np.array([vals(t, split) for t in ts])
            ax.hlines(y, np.nanmin(m, 0), np.nanmax(m, 0), color=col, lw=lw, label=lab, zorder=1)
        for label, t in qwen:
            if label in mk:
                ax.plot(vals(t, split), y, ls="", ms=4, mec="black", mew=0.5, zorder=3, label=QWEN_LABEL[label],
                        **mk[label])
        ax.axhline((y[TRAIN_CATS - 1] + y[TRAIN_CATS]) / 2, color="black", lw=0.5, ls=(0, (2, 2)))
        ax.axhline(y[len(CATS) - 1] - 0.7, color=style.TOL["pale"], lw=0.6)
        ax.set_xlim(-0.02 * top, 1.02 * top)
        ticks = np.arange(0, top + 1e-9, 0.25 if top == 1 else 0.1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v * 100:.0f}" for v in ticks[:-1]] + [f"{ticks[-1]:.0%}"])
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=style.TOL["pale"], lw=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(cols)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=6, frameon=False, handlelength=1.5, columnspacing=1.0,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.11, 1, 1), w_pad=1.0)
    style.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="agentharm_blog/figures")
    args = ap.parse_args()
    style.use()
    qwen = [(l, table(qwen_samples(l))) for l in QWEN]
    fr = frontier_samples()
    front = [(name, table(fr[k])) for k, name in FRONTIER] + [(name, table(old_samples(m))) for m, name in OLD]
    for l, t in front + qwen:
        print(f"{QWEN_LABEL.get(l, l):24s} harmful refusal " + " ".join(f"{v:4.0%}" for v in t["ref", "harm"])
              + "   benign refusal " + " ".join(f"{v:4.0%}" for v in t["ref", "ben"]))
    qwen_all = qwen + [(l, table(qwen_samples(l))) for l in QWEN_ONLY]
    for l, t in qwen_all[len(qwen):]:
        print(f"{QWEN_LABEL[l]:24s} harmful refusal " + " ".join(f"{v:4.0%}" for v in t["ref", "harm"]))
    qwen_figure(qwen_all, str(Path(args.out_dir) / "eval_categories_qwen"))
    qrl = [r for r in qwen_all if r[0] != "SFT-v3"]  # base + the SFT + RL checkpoints (no SFT-only row)
    heat_figure(front + qwen, str(Path(args.out_dir) / "eval_categories_frontier"), n_hosted=len(front))
    qwen_figure(qrl, str(Path(args.out_dir) / "eval_categories_qwen_rl"))
    heat_figure(qrl, str(Path(args.out_dir) / "eval_categories_heat_qwen"))
    stacked_figure(qrl, 0, str(Path(args.out_dir) / "eval_categories_stacked_qwen"))
    stacked_figure(qrl, 0, str(Path(args.out_dir) / "eval_categories_stacked_heldout_qwen"), held=True)
    order = sorted(front, key=lambda r: -r[1]["ref", "harm"][-1])
    stacked_figure(order + qwen, len(front), str(Path(args.out_dir) / "eval_categories_stacked"))
    # selected: without SELECTED_DROP and the SFT-only row, all rows ranked by harmful refusal
    sel = sorted([r for r in front if r[0] not in ev.SELECTED_DROP] + qrl, key=lambda r: -r[1]["ref", "harm"][-1])
    stacked_figure(sel, None, str(Path(args.out_dir) / "eval_categories_stacked_selected"),
                   bold={"SFT-v2 + RL", "SFT-v3 + RL"}, star=set(ev.CONTENT_FILTER))
    order = sorted(front, key=lambda r: -r[1]["held", "ref", "harm"])
    stacked_figure(order + qwen, len(front), str(Path(args.out_dir) / "eval_categories_stacked_heldout"), held=True)
    range_figure(front, qwen, str(Path(args.out_dir) / "eval_categories_range"))


if __name__ == "__main__":
    main()
