"""Offline LSTM pretraining followed by online SKF anomaly detection.

Run from the repository root:

    python examples/online_lstm_anomaly_detection.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytagi import metric

from canari import DataProcess, Model, SKF, plot_skf_states
from canari.component import LocalAcceleration, LocalTrend, LstmNetwork, WhiteNoise

ROOT = Path(__file__).resolve().parents[1]
NUM_EPOCHS = 30
ONLINE_WINDOW_LEN = 24
ANOMALY_THRESHOLD = 0.1


def load_data():
    values = pd.read_csv(ROOT / "data/toy_time_series/sine.csv")
    datetimes = pd.to_datetime(
        pd.read_csv(ROOT / "data/toy_time_series/sine_datetime.csv")["date_time"]
    )

    # Add a gradual drift entirely inside the test split.
    anomaly_start = 180
    values.loc[anomaly_start:, "sin"] += np.linspace(
        0.0, 2.0, len(values) - anomaly_start
    )
    values.index = datetimes
    values.index.name = "date_time"
    data_processor = DataProcess(
        data=values,
        time_covariates=["hour_of_day"],
        train_split=0.6,
        validation_split=0.15,
        output_col=[0],
    )
    return data_processor, anomaly_start


def main():
    data_processor, anomaly_start = load_data()
    train_data, validation_data, _, all_data = data_processor.get_splits()

    normal_model = Model(
        LocalTrend(),
        LstmNetwork(
            look_back_len=12,
            num_features=2,
            infer_len=24,
            num_hidden_unit=40,
            manual_seed=1,
        ),
        WhiteNoise(std_error=0.05),
    )
    abnormal_model = Model(
        LocalAcceleration(),
        LstmNetwork(),
        WhiteNoise(std_error=0.05),
    )
    skf = SKF(
        norm_model=normal_model,
        abnorm_model=abnormal_model,
        std_transition_error=1e-3,
        norm_to_abnorm_prob=1e-3,
        abnorm_to_norm_prob=0.1,
    )
    skf.auto_initialize_baseline_states(train_data["y"][:24])

    # Pretrain only the normal model's LSTM, selecting the epoch on validation data.
    for epoch in range(NUM_EPOCHS):
        validation_mean, validation_std, states = skf.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
            data_processor=data_processor,
        )
        validation_log_likelihood = float(
            metric.log_likelihood(
                validation_mean,
                validation_data["y"].flatten(),
                validation_std,
            )
        )
        skf.early_stopping(
            evaluate_metric=validation_log_likelihood,
            current_epoch=epoch,
            max_epoch=NUM_EPOCHS,
            mode="max",
            patience=5,
            skip_epoch=0,
        )
        skf.model["norm_norm"].set_memory(states=states, time_step=0)
        if skf.stop_training:
            break

    # Restart at the beginning and return the complete online SKF history.
    anomaly_probability, states = skf.online_lstm_filter(
        data=all_data,
        window_len=ONLINE_WINDOW_LEN,
    )
    detections = np.flatnonzero(
        anomaly_probability[data_processor.test_start :] > ANOMALY_THRESHOLD
    )
    if len(detections):
        first_detection = data_processor.test_start + int(detections[0])
        print(f"First anomaly detected at time step {first_detection}")
    else:
        print("No anomaly crossed the detection threshold")
    print(f"Injected anomaly starts at time step {anomaly_start}")
    print(f"Maximum anomaly probability: {anomaly_probability.max():.3f}")

    fig, axes = plot_skf_states(
        data_processor=data_processor,
        states=states,
        model_prob=anomaly_probability,
        states_to_plot=["level", "trend", "lstm"],
        states_type="posterior",
        standardization=True,
        legend_location="upper left",
    )
    anomaly_time = data_processor.data.index[anomaly_start]
    for ax in axes:
        ax.axvline(anomaly_time, color="red", linestyle="--")
    axes[-1].axhline(ANOMALY_THRESHOLD, color="black", linestyle=":")
    fig.suptitle("Offline-pretrained LSTM with online SKF anomaly detection")
    plt.show()


if __name__ == "__main__":
    main()
