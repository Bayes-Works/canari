"""Online LSTM learning from the first observation, then forecasting.

No offline pretraining: the LSTM is trained online over the train split and the
resulting model forecasts the validation split.

Run from the repository root:

    python examples/online_lstm.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pytagi import metric

from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_with_uncertainty,
)
from canari.component import LstmNetwork, WhiteNoise

ROOT = Path(__file__).resolve().parents[1]
ONLINE_WINDOW_LEN = 48
STATES_TO_PLOT = ["lstm"]


def load_data():
    values = pd.read_csv(ROOT / "data/toy_time_series/ar_periodic.csv")
    datetimes = pd.to_datetime(
        pd.read_csv(ROOT / "data/toy_time_series/ar_periodic_datetime.csv")["date_time"]
    )
    values.index = datetimes
    values.index.name = "date_time"
    return DataProcess(
        data=values,
        time_covariates=["hour_of_day"],
        train_split=0.8,
        validation_split=0.2,
        output_col=[0],
    )


def main():
    data_processor = load_data()
    _, validation_data, _, all_data = data_processor.get_splits()

    model = Model(
        LstmNetwork(
            look_back_len=24,
            num_features=2,
            infer_len=24,
            num_hidden_unit=40,
            num_layer=2,
            manual_seed=1,
        ),
        WhiteNoise(std_error=0.2),
    )

    # Online training: the LSTM starts untrained and learns while filtering the train
    # split. The first ONLINE_WINDOW_LEN steps fill the first fixed-lag window, so the
    # returned predictions and states start at that index.
    online_start = ONLINE_WINDOW_LEN
    train_mean, train_std, states = model.online_lstm_filter(
        data=all_data,
        start=online_start,
        window_len=ONLINE_WINDOW_LEN,
        end=data_processor.validation_start,
    )
    train_log_likelihood = metric.log_likelihood(
        train_mean,
        all_data["y"][online_start : data_processor.validation_start].flatten(),
        train_std,
    )

    # Forecast: the online phase leaves the model at the last training step, so the
    # validation split is predicted recursively with frozen LSTM parameters.
    model.lstm_net.num_samples = len(validation_data["y"])
    model.lstm_net.eval()
    validation_mean, validation_std, _ = model.forecast(data=validation_data)
    validation_log_likelihood = metric.log_likelihood(
        validation_mean,
        validation_data["y"].flatten(),
        validation_std,
    )

    print(f"Online train log-likelihood     : {train_log_likelihood: 0.3f}")
    print(f"Validation forecast log-likelihood: {validation_log_likelihood: 0.3f}")

    # Observations, online predictions, and validation forecast
    online_time = data_processor.data.index[
        online_start : data_processor.validation_start
    ]
    validation_time = data_processor.data.index[
        data_processor.validation_start : data_processor.validation_end
    ]
    fig, axes = plt.subplots(2 + len(STATES_TO_PLOT), 1, figsize=(10, 8), sharex=True)
    plot_data(
        data_processor=data_processor,
        standardization=True,
        sub_plot=axes[0],
        train_label="train",
        validation_label="validation",
    )
    plot_with_uncertainty(
        time=online_time,
        mu=train_mean,
        std=train_std,
        color="blue",
        label=["online prediction", r"$\pm\sigma$"],
        ax=axes[0],
    )
    plot_prediction(
        data_processor=data_processor,
        mean_validation_pred=validation_mean,
        std_validation_pred=validation_std,
        sub_plot=axes[0],
        color="purple",
        validation_label=["forecast", r"$\pm\sigma$"],
    )
    axes[0].set_ylabel("y")
    axes[0].legend(loc=(0.0, 1.01), ncol=5)

    # Online and forecast residuals
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].plot(
        online_time,
        all_data["y"][online_start : data_processor.validation_start].flatten()
        - train_mean,
        color="blue",
    )
    axes[1].plot(
        validation_time,
        validation_data["y"].flatten() - validation_mean,
        color="purple",
    )
    axes[1].set_ylabel("residual")

    # Hidden states estimated during online training
    for ax, state in zip(axes[2:], STATES_TO_PLOT):
        plot_with_uncertainty(
            time=online_time,
            mu=states.get_mean(state, "posterior"),
            std=states.get_std(state, "posterior"),
            color="blue",
            ax=ax,
        )
        ax.set_ylabel(state)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
