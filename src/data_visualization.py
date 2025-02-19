import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from examples import DataProcess
from src.data_struct import StatesHistory
import matplotlib.dates as mdates
from matplotlib.dates import (
    YearLocator,
    MonthLocator,
    WeekdayLocator,
    DayLocator,
    HourLocator,
    DateFormatter,
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

    if time_range > np.timedelta64(365, "D"):  # More than a year
        major_locator = YearLocator()
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
    if len_states == len(data_processor.time):
        return data_processor.time
    elif len_states == len(data_processor.train_time):
        return data_processor.train_time
    else:
        return np.concatenate(
            [data_processor.train_time, data_processor.validation_time], axis=0
        )


def get_mu_and_variance(
    states: StatesHistory, states_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the mean and variance arrays based on the states type.
    """
    if states_type == "posterior":
        return np.array(states.mu_posterior), np.array(states.var_posterior)
    elif states_type == "prior":
        return np.array(states.mu_prior), np.array(states.var_prior)
    elif states_type == "smooth":
        return np.array(states.mu_smooth), np.array(states.var_smooth)
    else:
        raise ValueError(f"Invalid states_type: {states_type}")


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
    if sub_plot is None:
        ax = plt.gca()
    else:
        ax = sub_plot

    if plot_column is None:
        plot_column = data_processor.output_col

    if plot_train_data and len(data_processor.train_time) > 0:
        if normalization:
            if plot_nan:
                ax.plot(
                    data_processor.train_time,
                    data_processor.train_data_norm[:, plot_column],
                    color=color,
                    linestyle=linestyle,
                    label=train_label,
                )
            else:
                mask = ~np.isnan(
                    data_processor.train_data_norm[:, plot_column]
                ).flatten()
                ax.plot(
                    data_processor.train_time[mask],
                    data_processor.train_data_norm[:, plot_column][mask],
                    color=color,
                    linestyle=linestyle,
                    label=train_label,
                )
            add_dynamic_grids(ax, data_processor.train_time)
        else:
            if plot_nan:
                ax.plot(
                    data_processor.train_time,
                    data_processor.train_data[:, plot_column],
                    color=color,
                    linestyle=linestyle,
                    label=train_label,
                )
            else:
                mask = ~np.isnan(data_processor.train_data[:, plot_column]).flatten()
                ax.plot(
                    data_processor.train_time[mask],
                    data_processor.train_data[:, plot_column][mask],
                    color=color,
                    linestyle=linestyle,
                    label=train_label,
                )
            add_dynamic_grids(ax, data_processor.train_time)

    if plot_validation_data and len(data_processor.validation_time) > 0:
        if normalization:
            if plot_nan:
                ax.plot(
                    data_processor.validation_time,
                    data_processor.validation_data_norm[:, plot_column],
                    color=color,
                    linestyle=linestyle,
                    label=validation_label,
                )
            else:
                mask = ~np.isnan(
                    data_processor.validation_data_norm[:, plot_column]
                ).flatten()
                ax.plot(
                    data_processor.validation_time[mask],
                    data_processor.validation_data_norm[:, plot_column][mask],
                    color=color,
                    linestyle=linestyle,
                    label=validation_label,
                )
            add_dynamic_grids(ax, data_processor.train_time)
        else:
            if plot_nan:
                ax.plot(
                    data_processor.validation_time,
                    data_processor.validation_data[:, plot_column],
                    color=color,
                    linestyle=linestyle,
                    label=validation_label,
                )
            else:
                mask = ~np.isnan(
                    data_processor.validation_data[:, plot_column]
                ).flatten()
                ax.plot(
                    data_processor.validation_time[mask],
                    data_processor.validation_data[:, plot_column][mask],
                    color=color,
                    linestyle=linestyle,
                    label=validation_label,
                )
            add_dynamic_grids(ax, data_processor.train_time)
        ax.axvspan(
            data_processor.validation_time[0],
            data_processor.validation_time[-1],
            color="green",
            alpha=0.1,
            edgecolor=None,
        )

    if plot_test_data and len(data_processor.test_time) > 0:
        if normalization:
            if plot_nan:
                ax.plot(
                    data_processor.test_time,
                    data_processor.test_data_norm[:, plot_column],
                    color=color,
                    linestyle=linestyle,
                    label=test_label,
                )
            else:
                mask = ~np.isnan(
                    data_processor.test_data_norm[:, plot_column]
                ).flatten()
                ax.plot(
                    data_processor.test_time[mask],
                    data_processor.test_data_norm[:, plot_column][mask],
                    color=color,
                    linestyle=linestyle,
                    label=test_label,
                )
            add_dynamic_grids(ax, data_processor.train_time)
        else:
            if plot_nan:
                ax.plot(
                    data_processor.test_time,
                    data_processor.test_data[:, plot_column],
                    color=color,
                    linestyle=linestyle,
                    label=test_label,
                )
            else:
                mask = ~np.isnan(data_processor.test_data[:, plot_column]).flatten()
                ax.plot(
                    data_processor.test_time[mask],
                    data_processor.test_data[:, plot_column][mask],
                    color=color,
                    linestyle=linestyle,
                    label=test_label,
                )
            add_dynamic_grids(ax, data_processor.train_time)
        ax.axvspan(
            data_processor.test_time[0],
            data_processor.test_time[-1],
            color="k",
            alpha=0.1,
            edgecolor=None,
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
    train_label: Optional[List[str]] = ["", ""],
    validation_label: Optional[List[str]] = ["", ""],
    test_label: Optional[List[str]] = ["", ""],
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
            label=train_label,
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
            label=validation_label,
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
            label=test_label,
        )


def plot_states(
    data_processor: DataProcess,
    states: StatesHistory,
    states_to_plot: Optional[list[str]] = ["all"],
    states_type: Optional[str] = "posterior",
    num_std: Optional[int] = 1,
    sub_plot: Optional[plt.Axes] = None,
    color: Optional[str] = "b",
    linestyle: Optional[str] = "-",
    legend_location: Optional[str] = None,
):
    # Time determination
    len_states = len(states.mu_prior)
    time = determine_time(data_processor, len_states)

    # Mean and variance selection based on states_type
    mu_plot, var_plot = get_mu_and_variance(states, states_type)

    if states_to_plot == ["all"]:
        states_to_plot = states.states_name

    # Create new figure or use predefined subplot
    if sub_plot is None:
        fig, axes = plt.subplots(len(states_to_plot), 1, figsize=(10, 8))
        if len(states_to_plot) == 1:
            axes = [axes]
    else:
        fig = None
        axes = [sub_plot]

    for idx, plot_state in enumerate(states_to_plot):
        ax = axes[idx] if sub_plot is None else sub_plot
        plot_index = states.states_name.index(plot_state)

        # Plot with uncertainty
        plot_with_uncertainty(
            time=time,
            mu=mu_plot[:, plot_index].flatten(),
            std=var_plot[:, plot_index, plot_index] ** 0.5,
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
    states_to_plot: Optional[list[str]] = ["all"],
    states_type: Optional[str] = "posterior",
    num_std: Optional[int] = 1,
    color: Optional[str] = "k",
    linestyle: Optional[str] = "-",
    legend_location: Optional[str] = None,
    plot_nan: Optional[bool] = True,
):
    # Time determination
    len_states = len(states.mu_prior)
    time = determine_time(data_processor, len_states)

    # Mean and variance selection based on states_type
    mu_plot, var_plot = get_mu_and_variance(states, states_type)

    if states_to_plot == ["all"]:
        states_to_plot = states.states_name

    fig, axes = plt.subplots(len(states_to_plot) + 1, 1, figsize=(10, 8))
    if len(states_to_plot) == 1:
        axes = [axes]

    for idx, plot_state in enumerate(states_to_plot):
        ax = axes[idx]
        plot_index = states.states_name.index(plot_state)

        # Plot with uncertainty
        plot_with_uncertainty(
            time=time,
            mu=mu_plot[:, plot_index].flatten(),
            std=var_plot[:, plot_index, plot_index] ** 0.5,
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

    plot_data(
        data_processor=data_processor,
        plot_column=data_processor.output_col,
        normalization=True,
        sub_plot=axes[0],
        plot_nan=plot_nan,
    )

    # Add legends for the first subplot
    if legend_location:
        axes[0].legend(
            [r"$\mu$", f"$\pm{num_std}\sigma$", r"$y$"],
            loc=legend_location,
            ncol=3,
        )

    # Plot abnormal model probability
    axes[len(states_to_plot)].plot(data_processor.time, model_prob, color="b")
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
