#!/usr/bin/env python3
"""Rows removed per generator model while building the AgentHarm SFT v1 dataset.

  (a) all removals as a share of each model's generated rows, stacked.
  (b) LLM-judge rates: unsafe compliance as a share of harmful prompts, over-refusal
      as a share of benign prompts.

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
import shutil
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["Claude\nSonnet 5", "Gemini 2.5\nFlash-Lite", "GPT-4o", "GPT-4o\nmini"]
GENERATED = np.array([1341, 1079, 782, 136])
FINAL = np.array([1090, 780, 575, 86])
HARMFUL_GENERATED = np.array([538, 540, 402, 75])  # rows from the malicious configs
BENIGN_GENERATED = np.array([803, 539, 380, 61])

REMOVED = {  # label: (colour, rows per model)
    "Unsafe compliance": ("#b2182b", [15, 50, 35, 4]),
    "Over-refusal": ("#ef8a62", [1, 20, 2, 1]),
    "Duplicate": ("#4d7ea8", [56, 121, 77, 30]),
    "Multi-turn": ("#bababa", [178, 86, 90, 15]),
    "Other": ("#6b6b6b", [1, 22, 3, 0]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("agentharm_blog/figures/sft_v1_filtering"))
    p.add_argument("--width", type=float, default=5.5, help="inches; 5.5 = NeurIPS/ICLR text width")
    p.add_argument("--dpi", type=int, default=1200, help="PNG resolution (the PDF is vector)")
    args = p.parse_args()

    removed = np.array([c for _, c in REMOVED.values()])
    assert (GENERATED - removed.sum(0) == FINAL).all()
    assert (HARMFUL_GENERATED + BENIGN_GENERATED == GENERATED).all()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(args.width, 2.2), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(MODELS))

    bottom = np.zeros(len(MODELS))
    for label, (color, counts) in REMOVED.items():
        pct = 100 * np.array(counts) / GENERATED
        ax1.bar(x, pct, 0.6, bottom=bottom, color=color, label=label)
        bottom += pct
    ax1.set_xticks(x, MODELS)
    ax1.set_ylabel("Removed (% of generated)")
    ax1.set_ylim(0, 40)
    ax1.set_title("(a) All removals", fontsize=8)
    ax1.legend(loc="upper left", fontsize=6.5, handlelength=1, labelspacing=0.3, ncol=2, columnspacing=0.8)

    unsafe = 100 * np.array(REMOVED["Unsafe compliance"][1]) / HARMFUL_GENERATED
    overref = 100 * np.array(REMOVED["Over-refusal"][1]) / BENIGN_GENERATED
    w = 0.36
    ax2.bar(x - w / 2, unsafe, w, color=REMOVED["Unsafe compliance"][0], label="Unsafe compliance")
    ax2.bar(x + w / 2, overref, w, color=REMOVED["Over-refusal"][0], label="Over-refusal")
    ax2.set_xticks(x, MODELS)
    ax2.set_ylabel("Rate (%)")
    ax2.set_ylim(0, 12.5)
    ax2.set_title("(b) LLM-judge filter", fontsize=8)
    ax2.legend(loc="upper right", fontsize=6.5, handlelength=1, labelspacing=0.3)

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", length=0, labelsize=6.5)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(w_pad=1.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = args.out.with_suffix(".pdf"), args.out.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    if shutil.which("gs"):  # low-memory rasterisation (this box has a 4 GB limit)
        subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", f"-r{args.dpi}",
                        "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", f"-sOutputFile={png}", str(pdf)],
                       check=True)
    else:
        fig.savefig(png, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {pdf} and {png}")


if __name__ == "__main__":
    main()
