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


def plot_data(
    data_processor: DataProcess,
    normalization: Optional[bool] = False,
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
    """Plot train, validation, and test data with normalization and NaN filtering"""

    if sub_plot is None:
        ax = plt.gca()
    else:
        ax = sub_plot

    if plot_column is None:
        plot_column = data_processor.output_col

    if normalization:
        data = data_processor.normalize_data()
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
    add_dynamic_grids(ax, total_time)


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
    """Plot mean and standard deviation for predictions"""

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
    # add_dynamic_grids(ax, total_time)


def plot_states(
    data_processor: DataProcess,
    states: StatesHistory,
    states_to_plot: Optional[list[str]] = "all",
    states_type: Optional[str] = "posterior",
    normalization: Optional[bool] = False,
    num_std: Optional[int] = 1,
    sub_plot: Optional[plt.Axes] = None,
    color: Optional[str] = "b",
    linestyle: Optional[str] = "-",
    legend_location: Optional[str] = None,
):
    """Plot hidden states"""

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
    time = determine_time(data_processor, len_states)

    # Mean and std to plot
    mu_states = states.get_mean(states_type=states_type, states_name=states_to_plot)
    std_states = states.get_std(states_type=states_type, states_name=states_to_plot)
    if not normalization:
        mu_states, std_states = common.unstandardize_states(
            mu_states,
            std_states,
            data_processor.norm_const_mean[data_processor.output_col],
            data_processor.norm_const_std[data_processor.output_col],
        )

    for idx, plot_state in enumerate(states_to_plot):
        ax = axes[idx] if sub_plot is None else sub_plot

        # Plot with uncertainty
        plot_with_uncertainty(
            time=time,
            mu=mu_states[plot_state].flatten(),
            std=std_states[plot_state].flatten(),
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
        if plot_state in ["local trend", "local acceleration"]:
            ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8)

        # Set ylabel to the name of the current state
        ax.set_ylabel(plot_state)
        add_dynamic_grids(ax, time)

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
    normalization: Optional[bool] = False,
    num_std: Optional[int] = 1,
    plot_observation: Optional[bool] = True,
    color: Optional[str] = "b",
    linestyle: Optional[str] = "-",
    legend_location: Optional[str] = None,
    plot_nan: Optional[bool] = True,
):
    """Plot hidden states along with probability of regime switch"""

    if states_to_plot == "all":
        states_to_plot = states.states_name

    fig, axes = plt.subplots(len(states_to_plot) + 1, 1, figsize=(10, 8))
    if len(states_to_plot) == 1:
        axes = [axes]

    # Time determination
    len_states = len(states.mu_prior)
    time = determine_time(data_processor, len_states)

    # Mean and std to plot
    mu_states = states.get_mean(states_type=states_type, states_name=states_to_plot)
    std_states = states.get_std(states_type=states_type, states_name=states_to_plot)
    if not normalization:
        mu_states, std_states = common.unstandardize_states(
            mu_states,
            std_states,
            data_processor.norm_const_mean[data_processor.output_col],
            data_processor.norm_const_std[data_processor.output_col],
        )

    for idx, plot_state in enumerate(states_to_plot):
        ax = axes[idx]

        # Plot with uncertainty
        plot_with_uncertainty(
            time=time,
            mu=mu_states[plot_state].flatten(),
            std=std_states[plot_state].flatten(),
            num_std=num_std,
            color=color,
            linestyle=linestyle,
            ax=ax,
        )

        # Plot horizontal line at y=0.0 for specific states
        if plot_state in ["local trend", "local acceleration"]:
            ax.axhline(0.0, color="red", linestyle="--", linewidth=0.8)

        # Set ylabel to the name of the current state
        ax.set_ylabel(plot_state)
        add_dynamic_grids(ax, time)

    if plot_observation:
        plot_data(
            data_processor=data_processor,
            plot_column=data_processor.output_col,
            normalization=normalization,
            sub_plot=axes[0],
            plot_nan=plot_nan,
        )
    add_dynamic_grids(axes[0], time)

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
    add_dynamic_grids(axes[len(states_to_plot)], time)
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
    """Plot mean and std"""

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


def add_dynamic_grids(ax, time):
    """
    Add dynamic major and minor grids to the x-axis of the plot based on the time range.

    Parameters:
        ax (plt.Axes): The axis to add the grid to.
        time (np.ndarray): Array of time values.
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


def determine_time(data_processor: DataProcess, len_states: int) -> np.ndarray:
    """
    Determine the appropriate time array based on the length of states.
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
