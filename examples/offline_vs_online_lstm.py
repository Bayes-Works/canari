"""Compare offline and online LSTM learning on a weekly benchmark series.

The same model is trained twice: offline over several epochs on the train split, and
online in a single pass with fixed-lag smoothing. Both then forecast the validation
split, and metrics are reported separately for the train and validation splits.

Run from the repository root:

    python examples/offline_vs_online_lstm.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytagi import metric

from canari import DataProcess, Model
from canari.component import LstmNetwork, WhiteNoise

ROOT = Path(__file__).resolve().parents[1]
SERIES = "MAT001PIAP-F510_x_cleaned"
NUM_EPOCHS = 100
ONLINE_WINDOW_LEN = 52
SIGMA_V = 0.1
COLORS = {"offline": "tab:blue", "online": "tab:green"}


def load_data():
    values = pd.read_csv(ROOT / "data/exp01_data/ts_weekly_values.csv")[SERIES]
    datetimes = pd.read_csv(ROOT / "data/exp01_data/ts_weekly_datetimes.csv")[SERIES]
    # Missing observations are kept as NaN so that the weekly time steps stay regular.
    series = pd.DataFrame({"date_time": pd.to_datetime(datetimes), SERIES: values})
    series = series.dropna(subset=["date_time"]).set_index("date_time")
    return DataProcess(
        data=series,
        time_covariates=["week_of_year"],
        train_split=0.8,
        validation_split=0.2,
        output_col=[0],
    )


def build_model():
    model = Model(
        LstmNetwork(
            look_back_len=52,
            num_features=2,
            infer_len=52 * 3,
            num_hidden_unit=256,
            num_layer=1,
            manual_seed=1,
        ),
        WhiteNoise(std_error=SIGMA_V),
    )

    # load global model
    # model.load_lstm_parameter_means("saved_params/global_BM_52_256.bin")

    return model


def run_offline(data_processor, train_data, validation_data):
    """Train over epochs on the train split, then forecast the validation split."""

    model = build_model()
    for epoch in range(NUM_EPOCHS):
        validation_mean, validation_std, states = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
            data_processor=data_processor,
        )
        model.early_stopping(
            evaluate_metric=float(
                metric.log_likelihood(
                    validation_mean, validation_data["y"].flatten(), validation_std
                )
            ),
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
    print(f"Offline optimal epoch: {model.optimal_epoch}")

    # Replay the train split with the optimal parameters frozen, then forecast.
    model = Model.load_dict(optimal_dict)
    model.lstm_net.num_samples = len(train_data["y"]) + len(validation_data["y"])
    model.lstm_net.eval()
    train_mean, train_std, _ = model.filter(train_data, train_lstm=False)
    validation_mean, validation_std, _ = model.forecast(data=validation_data)
    return train_mean, train_std, validation_mean, validation_std


def run_online(data_processor, all_data, validation_data):
    """Learn in a single pass over the train split, then forecast the validation split."""

    model = build_model()
    train_mean, train_std, _ = model.online_lstm_filter(
        data=all_data,
        start=ONLINE_WINDOW_LEN,
        window_len=ONLINE_WINDOW_LEN,
        end=data_processor.validation_start,
    )
    model.lstm_net.num_samples = len(validation_data["y"])
    model.lstm_net.eval()
    validation_mean, validation_std, _ = model.forecast(data=validation_data)
    return train_mean, train_std, validation_mean, validation_std


def score(mean, std, obs):
    """Log-likelihood and MSE over the time steps where an observation exists."""

    observed = ~np.isnan(obs)
    return {
        "log-likelihood": float(
            metric.log_likelihood(mean[observed], obs[observed], std[observed])
        ),
        "MSE": float(metric.mse(mean[observed], obs[observed])),
    }


def main():
    data_processor = load_data()
    train_data, validation_data, _, all_data = data_processor.get_splits()
    validation_obs = validation_data["y"].flatten()

    offline_train_mean, offline_train_std, offline_val_mean, offline_val_std = (
        run_offline(data_processor, train_data, validation_data)
    )
    online_train_mean, online_train_std, online_val_mean, online_val_std = run_online(
        data_processor, all_data, validation_data
    )

    # The online pass has no prediction before its first window is filled, so both
    # methods are compared from that time step onwards.
    train_obs = train_data["y"].flatten()[ONLINE_WINDOW_LEN:]
    offline_train_mean = offline_train_mean[ONLINE_WINDOW_LEN:]
    offline_train_std = offline_train_std[ONLINE_WINDOW_LEN:]

    scores = {
        "offline": {
            "train": score(offline_train_mean, offline_train_std, train_obs),
            "validation": score(offline_val_mean, offline_val_std, validation_obs),
        },
        "online": {
            "train": score(online_train_mean, online_train_std, train_obs),
            "validation": score(online_val_mean, online_val_std, validation_obs),
        },
    }
    predictions = {
        "offline": (
            np.concatenate([offline_train_mean, offline_val_mean]),
            np.concatenate([offline_train_std, offline_val_std]),
        ),
        "online": (
            np.concatenate([online_train_mean, online_val_mean]),
            np.concatenate([online_train_std, online_val_std]),
        ),
    }

    print(
        f"\nSeries {SERIES}: {len(train_obs)} train, {len(validation_obs)} validation"
    )
    print(f"{'':9}{'train LL':>10}{'train MSE':>11}{'val LL':>10}{'val MSE':>10}")
    for method, splits in scores.items():
        print(
            f"{method:9}"
            f"{splits['train']['log-likelihood']:>10.3f}"
            f"{splits['train']['MSE']:>11.3f}"
            f"{splits['validation']['log-likelihood']:>10.3f}"
            f"{splits['validation']['MSE']:>10.3f}"
        )

    # Predictions and residuals of both methods, over train and validation
    observations = np.concatenate([train_obs, validation_obs])
    time = data_processor.data.index[
        ONLINE_WINDOW_LEN : data_processor.validation_end
    ].to_numpy()
    split_time = data_processor.data.index[data_processor.validation_start]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(time, observations, color="tab:red", label="observation")
    for method, (mean, std) in predictions.items():
        axes[0].plot(time, mean, color=COLORS[method], label=method)
        axes[0].fill_between(
            time, mean - std, mean + std, color=COLORS[method], alpha=0.2
        )
        axes[1].plot(time, observations - mean, color=COLORS[method], label=method)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    for ax in axes:
        ax.axvline(split_time, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("y")
    axes[0].legend(loc=(0.0, 1.01), ncol=3)
    axes[1].set_ylabel("residual")
    axes[1].set_xlabel(f"validation starts at the dashed line ({split_time.date()})")
    fig.tight_layout()

    # Metrics per split
    fig_metrics, metric_axes = plt.subplots(1, 2, figsize=(7, 3))
    positions = np.arange(2)
    for ax, name in zip(metric_axes, ("log-likelihood", "MSE")):
        for offset, method in zip((-0.2, 0.2), COLORS):
            ax.bar(
                positions + offset,
                [scores[method][split][name] for split in ("train", "validation")],
                width=0.4,
                color=COLORS[method],
                label=method,
            )
        ax.set_xticks(positions, ["train", "validation"])
        ax.set_ylabel(name)
    metric_axes[0].legend(loc=(0.0, 1.01), ncol=2)
    fig_metrics.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
