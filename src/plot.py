import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def PlotStates(time, mu, var, color, index: Optional[int] = None):
    if index is not None:
        _mu = mu[:, index]
        _std = var[:, index, index].flatten() ** 0.5
    else:
        _mu = mu
        _std = var.flatten() ** 0.5
    plt.plot(time, _mu, color=color)
    plt.fill_between(
        time,
        _mu - _std,
        _mu + _std,
        color=color,
        alpha=0.2,
    )
