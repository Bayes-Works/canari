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
    plot_data,
    plot_prediction,
    plot_skf_states,
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
            data_processor.norm_const_mean[output_col],
            data_processor.norm_const_std[output_col],
        )
        std_validation_preds = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor.norm_const_std[output_col],
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
        fig, ax = plot_skf_states(
            data_processor=data_processor,
            states=states,
            model_prob=filter_marginal_abnorm_prob,
        )
        fig.suptitle("SKF hidden states", fontsize=10, y=1)
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
