#!/usr/bin/env python3
"""AgentHarm harmful and benign refusal rates, split into provider content-filter blocks and model refusals.

Same models, tasks (test_public Fraud + Cybercrime, 44 harmful + 44 benign) and logs as
plot_recent_runs_eval.py --eval-all / --selected, in all three prompt modes. A refusal counts as a
content-filter refusal when the provider stopped an agent generation with stop_reason ==
"content_filter" in that sample (definition of hedgehog-quills/benchmarks/agentharm/run_analysis/
agentharm_frontier_models_safety_stack_summary.md); the rest are model refusals. The Qwen runs are
served locally with vLLM and have no filter. Rows are ranked by neutral harmful refusal rate.

Usage (from /home/jovyan/Shared/kexin):
    python agentharm_blog/plot_eval_content_filter.py --out-dir agentharm_blog/figures             # -> eval_content_filter_all
    python agentharm_blog/plot_eval_content_filter.py --selected --out-dir agentharm_blog/figures  # -> eval_content_filter_selected
    python agentharm_blog/plot_eval_content_filter.py --filtered --out-dir agentharm_blog/figures  # -> eval_content_filter_filtered
    python agentharm_blog/plot_eval_content_filter.py --filtered --neutral --out-dir agentharm_blog/figures  # -> eval_content_filter_filtered_neutral
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import plot_recent_runs_eval as ev
import style

MODES = [("neutral", "Neutral"), ("refusal", "Refusal prompt"), ("jailbreak", "Jailbreak")]
QWEN_LABEL = {"Base Qwen3-14B": "Base Qwen3-14B", "SFT-v2 + RL": "Qwen3-14B SFT-v2 + RL",
              "SFT-v3 + RL": "Qwen3-14B SFT-v3 + RL"}
FILTERED = ["Gemini 4 Argon", "Claude Sonnet 5", "Claude Sonnet 5.5", "GPT-6.1 Sol"]  # --filtered
CF_COLOR = style.TOL["rose"]
MODEL_COLOR = style.TOL["grey"]


def sources(mode, selected):
    hosted = ev.HOSTED_LOCAL_NEUTRAL if mode == "neutral" else ev.HOSTED_MODES[mode]
    if selected:
        hosted = [m for m in hosted if m[0] not in ev.SELECTED_DROP]
    out = list(hosted)
    for l, _, h, b, _, _ in (r for r in ev.RUNS if r[0] in ev.FRONTIER_QWEN):
        out.append((QWEN_LABEL[l], *(ev.QWEN_MODES[mode][l] if mode != "neutral" else (h, b))))
    return out


def rates(rs):
    n = len(rs)
    ref = sum(float(r["refused"]) for r in rs)
    cf = sum(float(r["refused"]) for r in rs if r.get("blocked"))
    return ref / n, cf / n, int(cf), int(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="agentharm_blog/figures")
    ap.add_argument("--selected", action="store_true", help="drop SELECTED_DROP (-> eval_content_filter_selected)")
    ap.add_argument("--filtered", action="store_true",
                    help="only the hosted models in FILTERED, no Qwen rows (-> eval_content_filter_filtered)")
    ap.add_argument("--neutral", action="store_true", help="neutral prompt mode only (adds _neutral to the name)")
    args = ap.parse_args()
    style.use()
    modes = MODES[:1] if args.neutral else MODES

    data = {}  # (mode, label) -> {"harm": (rate, cf_rate, n_cf, n_ref), "ben": ...}
    for mode, _ in modes:
        for label, hid, bid in sources(mode, args.selected):
            if args.filtered and label not in FILTERED:
                continue
            H, B = ev.results(hid), ev.results(bid)
            assert len(H) == len(B) == 44
            data[mode, label] = {"harm": rates(H), "ben": rates(B)}
    labels = [l for l, _, _ in sources("neutral", args.selected) if not args.filtered or l in FILTERED]
    labels.sort(key=lambda l: (-data["neutral", l]["harm"][0], -data["neutral", l]["ben"][0]))
    for l in labels:
        print(f"{l:24s}", "  ".join(f"{m[:4]} harm {data[m, l]['harm'][0]:.0%} (cf {data[m, l]['harm'][2]}/{data[m, l]['harm'][3]})"
                                    f" ben {data[m, l]['ben'][0]:.0%} (cf {data[m, l]['ben'][2]}/{data[m, l]['ben'][3]})"
                                    for m, _ in modes))

    y = np.arange(len(labels))[::-1]
    if args.neutral:
        fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH, 0.3 * len(labels) + 0.95), sharey=True)
        panels = [("harm", "neutral", "(a) Harmful refusal rate ↑\nharmful tasks, n = 44"),
                  ("ben", "neutral", "(b) Benign refusal rate ↓\nbenign tasks, n = 44")]
    else:
        fig, axes = plt.subplots(1, 6, figsize=(6.75, 0.24 * len(labels) + (1.3 if args.filtered else 0.95)), sharey=True)
        panels = [(k, m, t) for k in ("harm", "ben") for m, t in MODES]
    for ax, (key, mode, title) in zip(axes, panels):
        top = 1.0 if key == "harm" or args.neutral else 0.5  # 6-panel benign refusals stay below 50%: zoom in
        for yi, l in zip(y, labels):
            tot, cf, _, _ = data[mode, l][key]
            ax.barh(yi, cf, height=0.7, color=CF_COLOR, edgecolor="black", lw=0.4)
            ax.barh(yi, tot - cf, left=cf, height=0.7, color=MODEL_COLOR, edgecolor="black", lw=0.4)
            ax.text(tot + 0.03 * top, yi, f"{tot:.0%}", va="center", fontsize=5.5)
        ax.set_xlim(0, top)
        ax.set_xticks([0, top / 2, top] if not args.neutral else [0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"] if args.neutral
                           else ["0", "50", "100%"] if key == "harm" else ["0", "25", "50%"])
        ax.set_title(title, loc="left", fontsize=8 if args.neutral else 7)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=style.TOL["pale"], lw=0.4)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_ylim(-0.6, len(labels) - 0.4)
    handles = [Patch(facecolor=CF_COLOR, edgecolor="black", lw=0.4, label="Content-filter block"),
               Patch(facecolor=MODEL_COLOR, edgecolor="black", lw=0.4, label="Model refusal")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    if args.neutral:
        fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
    else:
        fig.tight_layout(rect=(0, 0.05, 1, 0.95), w_pad=0.4)
        # group headers over the harmful / benign panel triples
        for i, txt in ((0, "(a) Harmful refusal rate ↑  (harmful tasks, n = 44)"), (3, "(b) Benign refusal rate ↓  (benign tasks, n = 44; axis to 50%)")):
            x0 = axes[i].get_position().x0
            fig.text(x0, 0.975, txt, ha="left", va="top", fontsize=8)
    style.save(fig, str(Path(args.out_dir) / f"eval_content_filter_{'filtered' if args.filtered else 'selected' if args.selected else 'all'}"
                                                f"{'_neutral' if args.neutral else ''}"))


if __name__ == "__main__":
    main()
