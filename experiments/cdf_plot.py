from pathlib import Path
from fractions import Fraction

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, norm

SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (6.5, 3.5)

mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
    "lines.linewidth": 1,
    "figure.figsize": SINGLE_COL,
    "font.size": 13,
    "savefig.dpi": 300,
})


# --- j1: Normal CDF on Pr(detection) ---
x1 = np.linspace(0, 1, 500)
j1 = norm.cdf(x1, loc=0.5, scale=0.2)

# --- j2: Lognormal CCDF on false-alarm rate ---
x2 = np.linspace(0.01, 1, 500)
j2 = 1 - lognorm.cdf(x2, s=0.2, scale=0.1)

# --- j3: Lognormal CCDF on anomaly magnitude ---
x3 = np.linspace(0, 1, 500)
j3 = 1 - lognorm.cdf(x3, s=0.6, scale=0.3)


# === PLOT ===
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5))

axes[0].plot(x1, j1, color="tab:blue")
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1.05)
axes[0].set_xlabel(r"$\mathtt{PD}$")
axes[0].set_ylabel(r"$j_1$")
axes[0].set_xticks([0, 0.5, 1])
axes[0].set_yticks([0, 1])

axes[1].plot(x2, j2, color="tab:blue")
axes[1].set_xlim(0.01, 1 / 5)
axes[1].set_ylim(0, 1.05)
axes[1].set_xlabel(r"$\mathtt{FA}(year^{-1})$")
axes[1].set_ylabel(r"$j_2$")
x_ticks_frac = [1 / 20, 1 / 10, 1 / 5]
axes[1].set_xticks(x_ticks_frac)
axes[1].set_xticklabels([f"${Fraction(x).limit_denominator()}$" for x in x_ticks_frac])
axes[1].set_yticks([0, 1])

axes[2].plot(x3, j3, color="tab:blue")
axes[2].set_xlim(0, 1)
axes[2].set_ylim(0, 1.05)
axes[2].set_xlabel(r"$\mathtt{AM}(\mathrm{u}/\mathrm{year})$")
axes[2].set_ylabel(r"$j_3$")
axes[2].set_xticks([0, 0.5, 1])
axes[2].set_yticks([0, 1])

for ax in axes:
    ax.yaxis.labelpad = 0.5

fig.tight_layout(w_pad=0.1)
fig.subplots_adjust(wspace=0.25)

out_dir = Path(__file__).resolve().parent / "out"
out_dir.mkdir(parents=True, exist_ok=True)
stem = out_dir / "cdf_plot"
fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(stem.with_suffix(".pgf"), bbox_inches="tight")
