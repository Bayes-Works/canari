import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from examples import DataProcess
from src.data_struct import StatesHistory


def plot_data(
    data_processor: DataProcess,
    normalization: Optional[bool] = False,
    plot_training_data: Optional[bool] = True,
    plot_validation_data: Optional[bool] = True,
    plot_test_data: Optional[bool] = True,
    plot_column: list[int] = None,
    sub_plot: Optional[plt.Axes] = None,
    color: Optional[str] = "r",
    linestyle: Optional[str] = "-",
):
    if sub_plot is None:
        ax = plt.gca()
    else:
        ax = sub_plot

    if plot_column is None:
        plot_column = data_processor.output_col

    if plot_training_data:
        if normalization:
            ax.plot(
                data_processor.train_time,
                data_processor.train_data_norm[:, plot_column],
                color=color,
                linestyle=linestyle,
            )
        else:
            ax.plot(
                data_processor.train_time,
                data_processor.train_data[:, plot_column],
                color=color,
                linestyle=linestyle,
            )

    if plot_validation_data:
        if normalization:
            ax.plot(
                data_processor.validation_time,
                data_processor.validation_data_norm[:, plot_column],
                color=color,
                linestyle=linestyle,
            )
        else:
            ax.plot(
                data_processor.validation_time,
                data_processor.validation_data[:, plot_column],
                color=color,
                linestyle=linestyle,
            )

    if plot_test_data:
        if normalization:
            ax.plot(
                data_processor.test_time,
                data_processor.test_data_norm[:, plot_column],
                color=color,
                linestyle=linestyle,
            )
        else:
            ax.plot(
                data_processor.test_time,
                data_processor.test_data[:, plot_column],
                color=color,
                linestyle=linestyle,
            )


def plot_prediction(
    data_processor: DataProcess,
    mean_train_pred: Optional[np.ndarray] = None,
    std_train_pred: Optional[np.ndarray] = None,
    mean_validation_pred: Optional[np.ndarray] = None,
    std_validation_pred: Optional[np.ndarray] = None,
    mean_test_pred: Optional[np.ndarray] = None,
    std_test_pred: Optional[np.ndarray] = None,
    num_std: Optional[int] = 1,
    sub_plot: Optional[plt.Axes] = None,
    color: Optional[str] = "blue",
    linestyle: Optional[str] = "-",
):
    if sub_plot is None:
        ax = plt.gca()
    else:
        ax = sub_plot

    if mean_train_pred is not None:
        plot_with_uncertainty(
            time=data_processor.train_time,
            mu=mean_train_pred,
            std=std_train_pred,
            num_std=num_std,
            color=color,
            linestyle=linestyle,
            ax=ax,
        )

    if mean_validation_pred is not None:
        plot_with_uncertainty(
            time=data_processor.validation_time,
            mu=mean_validation_pred,
            std=std_validation_pred,
            num_std=num_std,
            color=color,
            linestyle=linestyle,
            ax=ax,
        )

    if mean_test_pred is not None:
        plot_with_uncertainty(
            time=data_processor.test_time,
            mu=mean_test_pred,
            std=std_test_pred,
            num_std=num_std,
            color=color,
            linestyle=linestyle,
            ax=ax,
        )


def plot_states(
    data_processor: DataProcess,
    states: StatesHistory,
    plot_states: str,
    states_type: Optional[str] = "posterior",
    num_std: Optional[int] = 1,
    sub_plot: Optional[plt.Axes] = None,
    color: Optional[str] = "k",
    linestyle: Optional[str] = "-",
):
    # Subplot
    if sub_plot is None:
        ax = plt.gca()
    else:
        ax = sub_plot

    # Time
    len_states = len(states.mu_prior)
    if len_states == len(data_processor.time):
        time = data_processor.time
    elif len_states == len(data_processor.train_time):
        time = data_processor.train_time
    else:
        time = np.concatenate(
            [data_processor.train_time, data_processor.validation_time], axis=0
        )

    # Mean and var for plotted hidden states
    if states_type == "posterior":
        mu_plot = np.array(states.mu_posterior)
        var_plot = np.array(states.var_posterior)
    elif states_type == "prior":
        mu_plot = np.array(states.mu_prior)
        var_plot = np.array(states.var_prior)
    elif states_type == "smooth":
        mu_plot = np.array(states.mu_smooth)
        var_plot = np.array(states.var_smooth)

    # Index
    plot_index = states.states_name.index(plot_states)

    plot_with_uncertainty(
        time=time,
        mu=mu_plot[:, plot_index],
        std=var_plot[:, plot_index, plot_index] ** 0.5,
        num_std=num_std,
        color=color,
        linestyle=linestyle,
        ax=ax,
    )


def plot_with_uncertainty(
    time,
    mu,
    std,
    color: Optional[str] = "k",
    linestyle: Optional[str] = "-",
    label: Optional[list[str]] = ["", ""],
    index: Optional[int] = None,
    num_std: Optional[int] = 1,
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

    ax.plot(time, mu_plot, color=color, label=label[0], linestyle=linestyle)
    ax.fill_between(
        time,
        mu_plot - num_std * std_plot,
        mu_plot + num_std * std_plot,
        color=color,
        alpha=0.2,
        label=label[1],
    )
