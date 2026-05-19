from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
    "font.size": 9,
    "savefig.dpi": 300,
})


def false_alarm_adjusted_score(prob_det, far, h=1.0):
    """Score = P_DET * 2^(-FAR / h)."""
    return prob_det * 2 ** (-far / h)


def false_alarm_adjusted_score_hpfree(prob_det, far):
    """Hyperparameter-free score: P_DET / (1 + FAR)^3."""
    return prob_det / (1 + far) ** 3


far = np.linspace(0, 5, 500)
prob_det = 1.0

fig, ax = plt.subplots(figsize=SINGLE_COL)

h = 0.2
score_h = false_alarm_adjusted_score(prob_det, far, h=h)
line_h, = ax.plot(far, score_h, label=rf"$h = {h}$")
ax.scatter([h], [0.5], s=12, color=line_h.get_color(), zorder=3)
ax.vlines(h, 0, 0.5, linestyle=":", linewidth=0.7, color=line_h.get_color())

score_hpfree = false_alarm_adjusted_score_hpfree(prob_det, far)
line_hpfree, = ax.plot(
    far,
    score_hpfree,
    label=r"$(1+\#_{\mathrm{ALM}}(y^{-1}))^{-3}$",
)
far_half = 2 ** (1 / 3) - 1
ax.scatter([far_half], [0.5], s=12, color=line_hpfree.get_color(), zorder=3)
ax.vlines(far_half, 0, 0.5, linestyle=":", linewidth=0.7, color=line_hpfree.get_color())

ax.axhline(0.5, linestyle="--", linewidth=0.7, color="0.4")
ax.text(4.95, 0.52, "half-score", ha="right", va="bottom", color="0.4")

ax.set_xlabel(r"False-alarm rate $\#_{\mathrm{ALM}}(y^{-1})$")
ax.set_ylabel(r"Penalty")
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.05)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=2,
    frameon=False,
    handlelength=1.5,
    columnspacing=1.2,
)

fig.tight_layout()

out_dir = Path(__file__).resolve().parent / "out"
out_dir.mkdir(parents=True, exist_ok=True)
stem = out_dir / "fad_metric_plot"
fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(stem.with_suffix(".pgf"), bbox_inches="tight")
fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
