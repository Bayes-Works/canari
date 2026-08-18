"""Offline LSTM training, then forecasting the validation split.

The LSTM is trained over several epochs on the train split, the epoch is selected on
the validation forecast, and the optimal model then forecasts the validation split.

Run from the repository root:

    python examples/offline_lstm.py
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
NUM_EPOCHS = 50
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
    train_data, validation_data, _, _ = data_processor.get_splits()

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

    # Offline training: each epoch filters the whole train split and forecasts the
    # validation split, which is used for epoch selection and early stopping.
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
            patience=10,
            skip_epoch=5,
        )
        model.set_memory(states=states, time_step=0)
        if epoch == model.optimal_epoch:
            # Snapshot after set_memory: the model is positioned at the first time step.
            optimal_dict = model.get_dict()
        if model.stop_training:
            break

    # Restore the optimal epoch, replay the train split with frozen parameters to get
    # one-step-ahead predictions, then forecast the validation split recursively.
    optimal_epoch = model.optimal_epoch
    model = Model.load_dict(optimal_dict)
    model.lstm_net.num_samples = len(train_data["y"]) + len(validation_data["y"])
    model.lstm_net.eval()
    train_mean, train_std, _ = model.filter(train_data, train_lstm=False)
    validation_mean, validation_std, states = model.forecast(data=validation_data)
    train_log_likelihood = metric.log_likelihood(
        train_mean,
        train_data["y"].flatten(),
        train_std,
    )
    validation_log_likelihood = metric.log_likelihood(
        validation_mean,
        validation_data["y"].flatten(),
        validation_std,
    )

    print(f"Optimal epoch                     : {optimal_epoch}")
    print(f"Train log-likelihood              : {train_log_likelihood: 0.3f}")
    print(f"Validation forecast log-likelihood: {validation_log_likelihood: 0.3f}")

    # Observations, train predictions, and validation forecast
    train_time = data_processor.data.index[: data_processor.validation_start]
    validation_time = data_processor.data.index[
        data_processor.validation_start : data_processor.validation_end
    ]
    all_time = data_processor.data.index[: data_processor.validation_end]
    fig, axes = plt.subplots(2 + len(STATES_TO_PLOT), 1, figsize=(10, 8), sharex=True)
    plot_data(
        data_processor=data_processor,
        standardization=True,
        sub_plot=axes[0],
        train_label="train",
        validation_label="validation",
    )
    plot_prediction(
        data_processor=data_processor,
        mean_train_pred=train_mean,
        std_train_pred=train_std,
        sub_plot=axes[0],
        color="blue",
        train_label=["prediction", r"$\pm\sigma$"],
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

    # Residuals over the train and validation splits
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].plot(
        train_time, train_data["y"].flatten() - train_mean, color="blue"
    )
    axes[1].plot(
        validation_time,
        validation_data["y"].flatten() - validation_mean,
        color="purple",
    )
    axes[1].set_ylabel("residual")

    # Hidden states from the train filtering and the validation forecast
    for ax, state in zip(axes[2:], STATES_TO_PLOT):
        plot_with_uncertainty(
            time=all_time,
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
