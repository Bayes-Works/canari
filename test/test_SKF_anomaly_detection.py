import os
import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pytagi import Normalizer as normalizer
import pytagi.metric as metric

from src import (
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    WhiteNoise,
    Model,
    SKF,
    plot_with_uncertainty,
)
from examples import DataProcess

# Components
local_trend = LocalTrend(var_states=[1e-4, 1e-4])
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=12,
    num_features=1,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
    manual_seed=1,
)
noise = WhiteNoise(std_error=1e-3)

# Switching Kalman filter
skf = SKF(
    norm_model=Model(local_trend, lstm_network, noise),
    abnorm_model=Model(local_acceleration, lstm_network, noise),
    std_transition_error=1e-3,
    norm_to_abnorm_prob=1e-4,
    abnorm_to_norm_prob=1e-1,
    norm_model_prior_prob=0.99,
)


def SKF_anomaly_detection_runner(
    test_model: SKF,
    time_step_anomaly: int,
    slope_anomaly: float,
    anomaly_threshold: float,
    plot: bool,
) -> bool:
    """Run anomaly detection"""

    output_col = [0]

    # Read data
    data_file = "./data/toy_time_series/sine.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    data_file_time = "./data/toy_time_series/sine_datetime.csv"
    time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(time_series[0])
    df_raw.index = time_series

    # Add synthetic anomaly
    trend = np.linspace(0, 2, num=len(df_raw))
    anomaly_trend = np.linspace(0, slope_anomaly, num=len(df_raw) - time_step_anomaly)
    trend[time_step_anomaly:] = trend[time_step_anomaly:] + anomaly_trend
    df_raw = df_raw.add(trend, axis=0)

    # Data processing
    data_processor = DataProcess(
        data=df_raw,
        train_start="2000-01-01 00:00:00",
        train_end="2000-01-09 23:00:00",
        validation_start="2000-01-10 00:00:00",
        validation_end="2000-01-10 23:00:00",
        test_start="2000-01-11 00:00:00",
        output_col=output_col,
    )
    train_data, validation_data, _, all_data = data_processor.get_splits()

    # Training
    test_model.auto_initialize_baseline_states(train_data["y"][0:23])
    for _ in range(2):
        (mu_validation_preds, std_validation_preds, _) = test_model.lstm_train(
            train_data=train_data, validation_data=validation_data
        )
        # Unstandardize
        mu_validation_preds = normalizer.unstandardize(
            mu_validation_preds,
            data_processor.data_mean[output_col],
            data_processor.data_std[output_col],
        )
        std_validation_preds = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor.data_std[output_col],
        )

    # Anomaly detection
    filter_marginal_abnorm_prob, _ = skf.filter(data=all_data)
    smooth_marginal_abnorm_prob, states = skf.smoother(data=all_data)

    # Check anomalies
    condition1 = np.any(
        smooth_marginal_abnorm_prob[time_step_anomaly:] > anomaly_threshold
    )
    condition2 = np.any(
        smooth_marginal_abnorm_prob[:time_step_anomaly] < anomaly_threshold
    )
    detection = condition1 and condition2

    # Plot if requested
    if plot:
        mu_plot = np.array(states.mu_posterior)
        var_plot = np.array(states.var_posterior)
        marginal_abnorm_prob_plot = filter_marginal_abnorm_prob
        local_level_index = skf.states_name.index("local level")
        local_trend_index = skf.states_name.index("local trend")
        lstm_index = skf.states_name.index("lstm")
        white_noise_index = skf.states_name.index("white noise")

        fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=False)
        axs[0].plot(data_processor.time, all_data["y"], color="r", label="obs")
        axs[0].set_title("Observed Data")
        axs[0].set_ylabel("y")
        plot_with_uncertainty(
            time=data_processor.time,
            mu=mu_plot[:, local_level_index],
            std=var_plot[:, local_level_index, local_level_index] ** 0.5,
            color="b",
            label=["mu_val_pred", "±1σ"],
            ax=axs[0],
        )
        axs[0].axvline(
            x=data_processor.time[time_step_anomaly],
            color="b",
            linestyle="--",
            label="anomaly",
        )
        axs[0].xaxis.set_major_locator(mdates.DayLocator())
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axs[0].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        axs[0].grid(True, which="major", axis="x", linewidth=1.0, linestyle="-")
        axs[0].grid(True, which="minor", axis="x", linewidth=0.5, linestyle="--")
        axs[0].set_ylabel("Local level")

        plot_with_uncertainty(
            time=data_processor.time,
            mu=mu_plot[:, local_trend_index],
            std=var_plot[:, local_trend_index, local_trend_index] ** 0.5,
            color="b",
            label=["mu_val_pred", "±1σ"],
            ax=axs[1],
        )
        axs[1].axvline(
            x=data_processor.time[time_step_anomaly],
            color="b",
            linestyle="--",
            label="anomaly",
        )
        axs[1].axhline(y=0, color="black", linestyle="--", label="zero speed")
        axs[1].xaxis.set_major_locator(mdates.DayLocator())
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axs[1].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        axs[1].grid(True, which="major", axis="x", linewidth=1.0, linestyle="-")
        axs[1].grid(True, which="minor", axis="x", linewidth=0.5, linestyle="--")
        axs[1].legend()
        axs[1].set_ylabel("Local trend")

        plot_with_uncertainty(
            time=data_processor.time,
            mu=mu_plot[:, lstm_index],
            std=var_plot[:, lstm_index, lstm_index] ** 0.5,
            color="b",
            label=["mu_val_pred", "±1σ"],
            ax=axs[2],
        )
        axs[2].axvline(
            x=data_processor.time[time_step_anomaly],
            color="b",
            linestyle="--",
            label="anomaly",
        )
        axs[2].xaxis.set_major_locator(mdates.DayLocator())
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axs[2].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        axs[2].grid(True, which="major", axis="x", linewidth=1.0, linestyle="-")
        axs[2].grid(True, which="minor", axis="x", linewidth=0.5, linestyle="--")
        axs[2].set_ylabel("LSTM")

        plot_with_uncertainty(
            time=data_processor.time,
            mu=mu_plot[:, white_noise_index],
            std=var_plot[:, white_noise_index, white_noise_index] ** 0.5,
            color="b",
            label=["mu_val_pred", "±1σ"],
            ax=axs[3],
        )
        axs[3].axvline(
            x=data_processor.time[time_step_anomaly],
            color="b",
            linestyle="--",
            label="anomaly",
        )
        axs[3].xaxis.set_major_locator(mdates.DayLocator())
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axs[3].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        axs[3].grid(True, which="major", axis="x", linewidth=1.0, linestyle="-")
        axs[3].grid(True, which="minor", axis="x", linewidth=0.5, linestyle="--")
        axs[3].set_ylabel("White noise residual")

        axs[4].plot(data_processor.time, marginal_abnorm_prob_plot, color="b")
        axs[4].axvline(
            x=data_processor.time[time_step_anomaly],
            color="b",
            linestyle="--",
            label="anomaly",
        )
        axs[4].xaxis.set_major_locator(mdates.DayLocator())
        axs[4].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axs[4].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
        axs[4].grid(True, which="major", axis="x", linewidth=1.0, linestyle="-")
        axs[4].grid(True, which="minor", axis="x", linewidth=0.5, linestyle="--")
        axs[4].set_xlabel("Time")
        axs[4].set_ylabel("Pr(Abnormal)")

        plt.tight_layout()
        plt.show()

    return detection


def test_anomaly_detection(plot_mode):
    detection = SKF_anomaly_detection_runner(
        skf,
        time_step_anomaly=200,
        slope_anomaly=2,
        anomaly_threshold=0.5,
        plot=plot_mode,
    )
    assert detection == True, "Anomaly detection failed to detect anomaly."
