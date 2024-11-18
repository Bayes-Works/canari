import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_with_uncertainty(
    time,
    mu,
    std,
    color,
    label: Optional[list[str]] = ["", ""],
    index: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
):
    if index is not None:
        _mu = mu[1:, index]
        _std = std[1:, index, index].flatten()
    else:
        _mu = mu
        _std = std.flatten()
    ax.plot(time, _mu, color=color, label=label[0])
    ax.fill_between(
        time,
        _mu - _std,
        _mu + _std,
        color=color,
        alpha=0.2,
        label=label[1],
    )
