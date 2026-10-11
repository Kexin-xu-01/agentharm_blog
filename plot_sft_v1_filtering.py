#!/usr/bin/env python3
"""Rows removed per generator model while building the AgentHarm SFT v1 dataset,
as a share of each model's generated rows, stacked by reason.

Counts were recovered on 2026-10-10 from row-level files in git history (the per-batch
files were deleted in PR #53, commit 08d7ea5): raw / _clean / _dropped_* files at commit
0dd4647 under benchmarks/agentharm/datasets/sft-toolcalling/{,claude-batch/,claude-batch-2/,
gpt4o/,gpt4omini/}, diffed stage by stage through the combined files down to v1. Generated
rows minus all removals equals the v1 totals exactly. Categories:

  unsafe      LLM-judge step, harmful-labelled row that did not refuse: 9 had a tool call
              (dropped by rule, no judge call) + 95 the judge did not confirm benign
              (includes 3-vote ties / failed calls, which were also dropped)
  overrefuse  LLM-judge step, benign-labelled row refused and the judge confirmed benign
  duplicate   93 exact-duplicate prompts + 191 near-duplicates (TF-IDF cosine >= 0.9)
  multiturn   rows with a tool round-trip, removed to keep v1 single-turn
  other       4 quality defects + 9 over-refusals found by regex and checked by hand
              + 13 miscategorised rows (2026-09-05 review)

Usage (from /home/jovyan/Shared/kexin):
    python agentharm_blog/plot_sft_v1_filtering.py --out agentharm_blog/figures/sft_v1_filtering
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import style

GENERATED = np.array([1341, 1079, 782, 136])  # Claude, Gemini, GPT-4o, GPT-4o-mini
FINAL = np.array([1090, 780, 575, 86])
REMOVED = {  # label: (colour, rows per model)
    "Unsafe compliance": (style.TOL["wine"], [15, 50, 35, 4]),
    "Over-refusal": (style.TOL["rose"], [1, 20, 2, 1]),
    "Duplicate": (style.TOL["teal"], [56, 121, 77, 30]),
    "Multi-turn": (style.TOL["sand"], [178, 86, 90, 15]),
    "Other": (style.TOL["grey"], [1, 22, 3, 0]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="agentharm_blog/figures/sft_v1_filtering")
    p.add_argument("--dpi", type=int, default=600, help="PNG resolution (the PDF is vector)")
    args = p.parse_args()
    assert (GENERATED - np.array([c for _, c in REMOVED.values()]).sum(0) == FINAL).all()

    style.use()
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    x = np.arange(len(GENERATED))
    bottom = np.zeros(len(GENERATED))
    for label, (color, counts) in REMOVED.items():
        pct = 100 * np.array(counts) / GENERATED
        ax.bar(x, pct, 0.6, bottom=bottom, color=color, label=label)
        bottom += pct
    ax.set_xticks(x, style.MODEL_TICKS)
    ax.tick_params(axis="x", length=0, labelsize=6.5)
    ax.set_ylabel("Removed (% of generated)")
    ax.set_ylim(0, 50)
    ax.legend(loc="upper left", ncol=2, columnspacing=0.8)
    fig.tight_layout()
    style.save(fig, args.out, args.dpi)


if __name__ == "__main__":
    main()
