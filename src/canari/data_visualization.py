"""
Visualization for Canari's results.

This module provides functions to plot:

- Raw or standardized data
- Prediction results with uncertainty (mean ± standard deviation)
- Hidden state estimates
- Probability of regime changes (anomalies) from Switching Kalman filter (SKF)

"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from canari.data_process import DataProcess
from canari.data_struct import StatesHistory
import matplotlib.dates as mdates
from matplotlib.dates import (
    YearLocator,
    MonthLocator,
    WeekdayLocator,
    DayLocator,
    HourLocator,
    DateFormatter,
)
from canari import common


def _add_dynamic_grids(ax, time):
    """
    Add time-aware x-axis grids depending on data range.

    Args:
        ax (plt.Axes): Axis to configure.
        time (np.ndarray): Time array (datetime-indexed).
    """

    # Calculate the time range
    start_date = time[0]
    end_date = time[-1]
    time_range = end_date - start_date

    if time_range > np.timedelta64(15 * 365, "D"):  # More than 15 years
        major_locator = YearLocator(3)
        minor_locator = YearLocator(1)
        major_formatter = DateFormatter("%Y")
    elif time_range > np.timedelta64(365, "D"):  # More than a year
        major_locator = YearLocator(1)
        minor_locator = MonthLocator(bymonth=[1, 4, 7, 10])
        major_formatter = DateFormatter("%Y")
    elif time_range > np.timedelta64(30, "D"):  # More than a month
        major_locator = MonthLocator()
        minor_locator = WeekdayLocator(byweekday=mdates.MONDAY)  # Weekly on Mondays
        major_formatter = DateFormatter("%b %Y")
    else:  # Less than a month
        major_locator = DayLocator()
        minor_locator = HourLocator(byhour=[0, 6, 12, 18])  # 4 times a day
        major_formatter = DateFormatter("%m-%d")

    # Apply locators and formatters
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_major_formatter(major_formatter)

    # Enable vertical grid only
    ax.grid(
        visible=True, which="major", axis="x", linestyle="-", linewidth=0.8, alpha=0.8
    )
    ax.grid(
        visible=True, which="minor", axis="x", linestyle="--", linewidth=0.5, alpha=0.5
    )

    # Disable horizontal grid
    ax.grid(visible=False, which="major", axis="y")
    ax.grid(visible=False, which="minor", axis="y")


def _determine_time(data_processor: DataProcess, len_states: int) -> np.ndarray:
    """
    Infer which segment of the time index to use for state plots.

    Args:
        data_processor (DataProcess): Provides split index mapping.
        len_states (int): Number of states in time.

    Returns:
        np.ndarray: Time values corresponding to the state sequence.
    """

    train_index, val_index, _ = data_processor.get_split_indices()
    if len_states == len(data_processor.data):
        return data_processor.data.index.to_numpy()
    elif len_states == len(train_index):
        return data_processor.data.index[train_index].to_numpy()
    else:
        return np.concatenate(
            [
                data_processor.data.index[train_index].to_numpy(),
                data_processor.data.index[val_index].to_numpy(),
            ],
            axis=0,
        )


def plot_data(
    data_processor: DataProcess,
    standardization: Optional[bool] = False,
    plot_train_data: Optional[bool] = True,
    plot_validation_data: Optional[bool] = True,
    plot_test_data: Optional[bool] = True,
    plot_column: list[int] = None,
    sub_plot: Optional[plt.Axes] = None,
    color: Optional[str] = "r",
    linestyle: Optional[str] = "-",
    train_label: Optional[str] = None,
    validation_label: Optional[str] = None,
    test_label: Optional[str] = None,
    plot_nan: Optional[bool] = True,
):
    """
    Plot train, validation, and test data with optional standardization and NaN filtering.

    Args:
        data_processor (DataProcess): Data processing object.
        standardization (bool, optional): Plot data in standardized or original space
        plot_train_data (bool, optional): If True, plot training data.
        plot_validation_data (bool, optional): If True, plot validation data.
        plot_test_data (bool, optional): If True, plot test data.
        plot_column (list[int], optional): List of column indices to plot.
        sub_plot (plt.Axes, optional): Matplotlib subplot axis to plot on.
        color (str, optional): Line color for plot lines.
        linestyle (str, optional): Line style (e.g., '-', '--').
        train_label (str, optional): Legend label for training data.
        validation_label (str, optional): Legend label for validation data.
        test_label (str, optional): Legend label for test data.
        plot_nan (bool, optional): Whether to include NaNs in the plot for plotting missing values.

    Examples:
        >>> from canari import DataProcess, plot_data
        >>> # Create data
        >>> dt_index = pd.date_range(start="2025-01-01", periods=11, freq="H")
        >>> data = pd.DataFrame({'value': np.linspace(0.1, 1.0, 11)},
                        index=dt_index)
        >>> dp = DataProcess(data,
                    train_split=0.7,
                    validation_split=0.2,
                    test_split=0.1,
                    time_covariates = ["hour_of_day"],
                    standardization=True)
        >>> fig, ax = plt.subplots(figsize=(12, 4))
        >>> plot_data(
                data_processor=dp,
                standardization=True,
                plot_validation_data=True,
                plot_test_data=True,
            )
    """

    if sub_plot is None:
        ax = plt.gca()
    else:
        ax = sub_plot

    if plot_column is None:
        plot_column = data_processor.output_col

    if standardization:
        data = data_processor.standardize_data()
    else:
        data = data_processor.data.values

    labels = [train_label, validation_label, test_label]
    plot_flags = [plot_train_data, plot_validation_data, plot_test_data]
    split_indices = list(data_processor.get_split_indices())
    total_time = []

    for plot_flag, index, label in zip(plot_flags, split_indices, labels):
        if plot_flag:
            data_plot = data[index, plot_column]
            time = data_processor.data.index[index]
            total_time.extend(list(time))
            if plot_nan:
                ax.plot(time, data_plot, label=label, color=color, linestyle=linestyle)
            else:
                mask = ~np.isnan(data_plot).flatten()
                ax.plot(
                    time[mask],
                    data_plot[mask],
                    label=label,
                    color=color,
                    linestyle=linestyle,
                )

    if (
        plot_validation_data
        and data_processor.validation_start != data_processor.validation_end
    ):
        ax.axvspan(
            data_processor.data.index[data_processor.validation_start],
            data_processor.data.index[data_processor.validation_end - 1],
            color="green",
            alpha=0.1,
            edgecolor=None,
        )
    _add_dynamic_grids(ax, total_time)


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
    train_label: Optional[List[str]] = ["", ""],
    validation_label: Optional[List[str]] = ["", ""],
    test_label: Optional[List[str]] = ["", ""],
):
    """
    Plot predicted mean and uncertainty for each data split. The uncertainty is represented by
    confidence regions based on a multiple of the associated variable's standard deviation.

    Args:
        data_processor (DataProcess): Data processing object.
        mean_train_pred (np.ndarray, optional): Predicted means for training.
        std_train_pred (np.ndarray, optional): Standard deviations for training predictions.
        mean_validation_pred (np.ndarray, optional): Predicted means for validation.
        std_validation_pred (np.ndarray, optional): Standard deviations for validation predictions.
        mean_test_pred (np.ndarray, optional): Predicted means for test predictions.
        std_test_pred (np.ndarray, optional): Standard deviations for test predictions.
        num_std (int, optional): Number of std deviations for confidence region.
        sub_plot (plt.Axes, optional): Matplotlib subplot axis to plot on.
        color (str, optional): Line color.
        linestyle (str, optional): Line style.
        train_label (List[str], optional): [mean_label, std_label] for train.
        validation_label (List[str], optional): [mean_label, std_label] for validation.
        test_label (List[str], optional): [mean_label, std_label] for test.

    Examples:
        >>> from canari import plot_prediction
        >>> mu_preds_val, std_preds_val, states = model.lstm_train(train_data=train_set,validation_data=val_set)
        >>> plot_prediction(
                data_processor=dp,
                mean_validation_pred=mu_preds_val,
                std_validation_pred=std_preds_val,
            )
    """

    if sub_plot is None:
        ax = plt.gca()
    else:
        ax = sub_plot

    split_indices = list(data_processor.get_split_indices())
    mean_values = [mean_train_pred, mean_validation_pred, mean_test_pred]
    std_values = [std_train_pred, std_validation_pred, std_test_pred]
    labels = [train_label, validation_label, test_label]
    total_time = []

    for index, mean_value, std_value, label in zip(
        split_indices, mean_values, std_values, labels
    ):
        if mean_value is not None:
            time = data_processor.data.index[index]
            total_time.extend(list(time))
            plot_with_uncertainty(
                time=time,
                mu=mean_value,
                std=std_value,
                num_std=num_std,
                color=color,
                linestyle=linestyle,
                ax=ax,
                label=label,
            )
    # _add_dynamic_grids(ax, total_time)


def plot_states(
    data_processor: DataProcess,
    states: StatesHistory,
    states_to_plot: Optional[list[str]] = "all",
    states_type: Optional[str] = "posterior",
    standardization: Optional[bool] = False,
    num_std: Optional[int] = 1,
    sub_plot: Optional[plt.Axes] = None,
    color: Optional[str] = "b",
    linestyle: Optional[str] = "-",
    legend_location: Optional[str] = None,
):
    """
    Plot hidden states with mean and confidence regions.

    Args:
        data_processor (DataProcess): Data processing object.
        states (StatesHistory): Object containing hidden states history over time.
        states_to_plot (list[str] or "all", optional): Names of states to plot.
        states_type (str, optional): Type of state ('posterior' or 'prior' or 'smooth').
        standardization (bool, optional): Whether to plot hidden states in standardized or original space.
        num_std (int, optional): Number of standard deviations for confidence region.
        sub_plot (plt.Axes, optional): Matplotlib subplot axis to plot on.
        color (str, optional): Color for mean line and confidence region fill.
        linestyle (str, optional): Line style.
        legend_location (str, optional): Legend placement for first state subplot.

    Examples:
        >>> from canari import plot_states
        >>> mu_preds_val, std_preds_val, states = model.lstm_train(train_data=train_set,validation_data=val_set)
        >>> fig, ax = plot_states(
                data_processor=dp,
                states=states,
                states_type="posterior"
            )
    """

    if states_to_plot == "all":
        states_to_plot = states.states_name

    # Create new figure or use predefined subplot
    if sub_plot is None:
        fig, axes = plt.subplots(len(states_to_plot), 1, figsize=(10, 8))
        if len(states_to_plot) == 1:
            axes = [axes]
    else:
        fig = None
        axes = [sub_plot]

    # Time determination
    len_states = len(states.mu_prior)
    time = _determine_time(data_processor, len_states)

    for idx, plot_state in enumerate(states_to_plot):

        ax = axes[idx] if sub_plot is None else sub_plot

        # Get mean and std to plot
        if standardization:
            mu_states = states.get_mean(
                states_type=states_type, states_name=plot_state, standardization=True
            )
            std_states = states.get_std(
                states_type=states_type, states_name=plot_state, standardization=True
            )
        else:
            mu_states = states.get_mean(
                states_type=states_type,
                states_name=plot_state,
                standardization=False,
                scale_const_mean=data_processor.scale_const_mean[
                    data_processor.output_col
                ],
                scale_const_std=data_processor.scale_const_std[
                    data_processor.output_col
                ],
            )
            std_states = states.get_std(
                states_type=states_type,
                states_name=plot_state,
                standardization=False,
                scale_const_std=data_processor.scale_const_std[
                    data_processor.output_col
                ],
            )

        # Plot with uncertainty
        plot_with_uncertainty(
            time=time,
            mu=mu_states,
            std=std_states,
            num_std=num_std,
            color=color,
            linestyle=linestyle,
            ax=ax,
        )

        # Add legends for the first subplot
        if legend_location:
            if idx == 0:
                ax.legend(
                    [r"$\mu$", f"$\pm{num_std}\sigma$"], loc=legend_location, ncol=2
                )

        # Plot horizontal line at y=0.0 for specific states
        if plot_state in ["trend", "acceleration"]:
            ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8)

        # Set ylabel to the name of the current state
        ax.set_ylabel(plot_state)
        _add_dynamic_grids(ax, time)

        if sub_plot is not None:
            break

    if sub_plot is None:
        plt.tight_layout()

    return fig, axes


def plot_skf_states(
    data_processor: DataProcess,
    states: StatesHistory,
    model_prob: np.ndarray,
    states_to_plot: Optional[list[str]] = "all",
    states_type: Optional[str] = "posterior",
    standardization: Optional[bool] = False,
    num_std: Optional[int] = 1,
    plot_observation: Optional[bool] = True,
    color: Optional[str] = "b",
    linestyle: Optional[str] = "-",
    legend_location: Optional[str] = None,
    plot_nan: Optional[bool] = True,
):
    """
    Plot hidden states along with probabilities of regime changes.

    Args:
        data_processor (DataProcess): Data processing object.
        states (StatesHistory): Object containing hidden states history over time.
        model_prob (np.ndarray): Probabilities of the abnormal regime.
        states_to_plot (list[str] or "all", optional): Names of states to plot.
        states_type (str, optional): Type of state ('posterior' or 'prior' or 'smooth').
        standardization (bool, optional):  Whether to plot hidden states in standardized or original space.
        num_std (int, optional): Number of standard deviations for confidence regions.
        plot_observation (bool, optional): Whether to include observed data into "level" plot.
        color (str, optional): Line color for states.
        linestyle (str, optional): Line style.
        legend_location (str, optional): Location of legend in top subplot.
        plot_nan (bool, optional): Whether to include NaNs in the plot for plotting missing values.

    Examples:
        >>> from canari import plot_skf_states
        >>> filter_marginal_abnorm_prob, states = skf.filter(data=all_data)
        >>> fig, ax = plot_skf_states(
                data_processor=dp,
                states=states,
                states_type="posterior",
                model_prob=filter_marginal_abnorm_prob,
            )

    """

    if states_to_plot == "all":
        states_to_plot = states.states_name

    fig, axes = plt.subplots(len(states_to_plot) + 1, 1, figsize=(10, 8))
    if len(states_to_plot) == 1:
        axes = [axes]

    # Time determination
    len_states = len(states.mu_prior)
    time = _determine_time(data_processor, len_states)

    for idx, plot_state in enumerate(states_to_plot):

        ax = axes[idx]

        # Get mean and std to plot
        if standardization:
            mu_states = states.get_mean(
                states_type=states_type, states_name=plot_state, standardization=True
            )
            std_states = states.get_std(
                states_type=states_type, states_name=plot_state, standardization=True
            )
        else:
            mu_states = states.get_mean(
                states_type=states_type,
                states_name=plot_state,
                standardization=False,
                scale_const_mean=data_processor.scale_const_mean[
                    data_processor.output_col
                ],
                scale_const_std=data_processor.scale_const_std[
                    data_processor.output_col
                ],
            )
            std_states = states.get_std(
                states_type=states_type,
                states_name=plot_state,
                standardization=False,
                scale_const_std=data_processor.scale_const_std[
                    data_processor.output_col
                ],
            )

        # Plot with uncertainty
        plot_with_uncertainty(
            time=time,
            mu=mu_states,
            std=std_states,
            num_std=num_std,
            color=color,
            linestyle=linestyle,
            ax=ax,
        )

        # Plot horizontal line at y=0.0 for specific states
        if plot_state in ["trend", "acceleration"]:
            ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8)

        # Set ylabel to the name of the current state
        ax.set_ylabel(plot_state)
        _add_dynamic_grids(ax, time)

    if plot_observation:
        plot_data(
            data_processor=data_processor,
            plot_column=data_processor.output_col,
            standardization=standardization,
            sub_plot=axes[0],
            plot_nan=plot_nan,
        )
    _add_dynamic_grids(axes[0], time)

    # Add legends for the first subplot
    if legend_location:
        axes[0].legend(
            [r"$\mu$", f"$\pm{num_std}\sigma$", r"$y$"],
            loc=legend_location,
            ncol=3,
        )

    # Plot abnormal model probability
    axes[len(states_to_plot)].plot(time, model_prob, color="b")
    axes[len(states_to_plot)].set_ylabel("Pr(Abnormal)")
    _add_dynamic_grids(axes[len(states_to_plot)], time)
    axes[len(states_to_plot)].set_ylim(-0.02, 1)
    axes[len(states_to_plot)].set_xlabel("Time")
    if legend_location:
        axes[len(states_to_plot)].legend(["Pr(Abnormal)"], loc="upper left")

    plt.tight_layout()

    return fig, axes


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
    """
    Plot mean and confidence region for a variable.

    Args:
        time (np.ndarray): Time vector.
        mu (np.ndarray): Mean values to plot.
        std (np.ndarray): Standard deviation values.
        color (str, optional): Line/fill color.
        linestyle (str, optional): Line style.
        label (List[str], optional): Labels for mean and confidence region.
        index (int, optional): Index if plotting from a multivariate tensor.
        num_std (int, optional): Number of standard deviations for confidence regions.
        ax (plt.Axes, optional): Axis to use for plotting.
    """

    if index is not None:
        mu_states = mu[:, index]
        std_states = std[:, index, index].flatten()
    else:
        mu_states = mu
        std_states = std.flatten()

    if ax is None:
        ax = plt.gca()

    ax.plot(time, mu_states, color=color, label=label[0], linestyle=linestyle)
    ax.fill_between(
        time,
        mu_states - num_std * std_states,
        mu_states + num_std * std_states,
        color=color,
        alpha=0.2,
        label=label[1],
    )
