"""Shared matplotlib style and colours for the agentharm_blog figures.

Colours are Paul Tol's "muted" scheme (https://personal.sron.nl/~pault/), which stays
distinguishable under colour-vision deficiency and prints well in greyscale order.
"""

import shutil
import subprocess

import matplotlib.pyplot as plt

TOL = {
    "indigo": "#332288", "cyan": "#88CCEE", "teal": "#44AA99", "green": "#117733",
    "olive": "#999933", "sand": "#DDCC77", "rose": "#CC6677", "wine": "#882255",
    "purple": "#AA4499", "grey": "#BBBBBB", "pale": "#DDDDDD",
}

# Categories: one hue per domain, dark = harmful, light = benign.
CATEGORIES = [  # key, label, colour
    ("cybercrime", "Cybercrime (harmful)", TOL["indigo"]),
    ("benign_cybercrime", "Cybercrime (benign)", TOL["cyan"]),
    ("fraud", "Fraud (harmful)", TOL["wine"]),
    ("benign_fraud", "Fraud (benign)", TOL["rose"]),
    ("benign_general", "General (benign)", TOL["green"]),
]
MODELS = [  # key, label, colour (disjoint from the category colours)
    ("claude-sonnet-5", "Claude Sonnet 5", TOL["sand"]),
    ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", TOL["teal"]),
    ("gpt-4o", "GPT-4o", TOL["purple"]),
    ("gpt-4o-mini", "GPT-4o-mini", TOL["olive"]),
]
MODEL_TICKS = ["Claude\nSonnet 5", "Gemini 2.5\nFlash-Lite", "GPT-4o", "GPT-4o\nmini"]

WIDTH = 5.5  # NeurIPS / ICLR text width in inches (ICML: 6.75)


def use():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "legend.handlelength": 1.0,
        "legend.labelspacing": 0.3,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, stem, dpi=600):
    """Write <stem>.pdf, then rasterise it to <stem>.png with Ghostscript if available
    (much lower peak memory than matplotlib at high dpi; this box has a 4 GB limit)."""
    pdf, png = f"{stem}.pdf", f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    if shutil.which("gs"):
        subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", f"-r{dpi}",
                        "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", f"-sOutputFile={png}", pdf],
                       check=True)
    else:
        fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {pdf} and {png}")
