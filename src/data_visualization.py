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
        mu_plot = mu[:, index]
        std_plot = std[:, index, index].flatten()
    else:
        mu_plot = mu
        std_plot = std.flatten()

    if ax is None:
        ax = plt.gca()

    ax.plot(time, mu_plot, color=color, label=label[0])
    ax.fill_between(
        time,
        mu_plot - std_plot,
        mu_plot + std_plot,
        color=color,
        alpha=0.2,
        label=label[1],
    )
