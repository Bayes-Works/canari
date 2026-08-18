"""Offline LSTM pretraining followed by online filtering on the test set.

Run from the repository root:

    python examples/online_lstm.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pytagi import metric

from canari import DataProcess, Model, plot_data, plot_prediction
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

ROOT = Path(__file__).resolve().parents[1]
NUM_EPOCHS = 30
ONLINE_WINDOW_LEN = 24


def load_data():
    values = pd.read_csv(ROOT / "data/toy_time_series/sine.csv")
    datetimes = pd.to_datetime(
        pd.read_csv(ROOT / "data/toy_time_series/sine_datetime.csv")["date_time"]
    )
    values.index = datetimes
    values.index.name = "date_time"
    return DataProcess(
        data=values,
        time_covariates=["hour_of_day"],
        train_split=0.6,
        validation_split=0.15,
        output_col=[0],
    )


def main():
    data_processor = load_data()
    train_data, validation_data, test_data, all_data = data_processor.get_splits()

    model = Model(
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
    model.auto_initialize_baseline_states(train_data["y"][:24])

    # Offline pretraining: train split for parameter updates, validation split
    # for epoch selection and early stopping.
    for epoch in range(NUM_EPOCHS):
        validation_mean, validation_std, states = model.lstm_train(
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
        model.early_stopping(
            evaluate_metric=validation_log_likelihood,
            current_epoch=epoch,
            max_epoch=NUM_EPOCHS,
            mode="max",
            patience=5,
            skip_epoch=0,
        )
        model.set_memory(states=states, time_step=0)
        if model.stop_training:
            break

    # Online phase: each returned prediction is made before its test observation
    # updates the LSTM and the fixed-lag window is smoothed and advanced.
    test_mean, test_std, _ = model.online_lstm_filter(
        data=all_data,
        start=data_processor.test_start,
        window_len=ONLINE_WINDOW_LEN,
    )
    test_log_likelihood = metric.log_likelihood(
        test_mean,
        test_data["y"].flatten(),
        test_std,
    )
    print(f"Optimal epoch: {model.optimal_epoch}")
    print(f"Online test log-likelihood: {test_log_likelihood:.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data(
        data_processor=data_processor,
        standardization=True,
        sub_plot=ax,
        train_label="train",
        validation_label="validation",
        test_label="test",
    )
    plot_prediction(
        data_processor=data_processor,
        mean_test_pred=test_mean,
        std_test_pred=test_std,
        sub_plot=ax,
        test_label=["online prediction", r"$\pm\sigma$"],
    )
    ax.set_title("Offline-pretrained LSTM with online test filtering")
    ax.set_xlabel("Time")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
