"""
Sweep look-back and smoothing-window lengths for the online LSTM experiment.

The data, model, and online update scheme match ``toy_online_lstm.py``, splits included:
the online walk takes everything up to the test set and the test span is the only held-out
one. For every ``(look_back_len, D)`` pair this script records one number per span:

- online one-step MSE over the rolling training predictions;
- multi-step forecast MSE over the test span;
- elapsed runtime and the number of online windows.

Results are written incrementally to
``experiments/out/toy_online_lstm_grid/toy_online_lstm_grid.csv`` so a long sweep can be
resumed with ``--resume``. Two annotated heatmaps are saved as PNG, PDF, and PGF files with
the same stem.

Examples
--------
Run the default 4 x 4 grid::

    python -m experiments.toy_online_lstm_grid

Run a custom grid::

    python -m experiments.toy_online_lstm_grid \
        --lookbacks 1 12 24 48 --smoothing-windows 12 24 48 96

Run from the repository root: the script imports `experiments.utils`.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import pytagi.metric as metric

# Import utils before canari so it owns the matplotlib configuration.
from experiments.utils import (
    LstmLookBackBuffer,
    generate_periodic_signal,
    make_window,
    output_dir,
    plot_performance_grid,
    rewind_to_step,
    set_output_subdir,
    smooth_window,
)

from canari import DataProcess, Model
from canari.component import LstmNetwork, WhiteNoise

# Everything this sweep writes - the heatmaps and the results CSV - goes under
# `experiments/out/toy_online_lstm_grid/`.
set_output_subdir("toy_online_lstm_grid")

DEFAULT_LOOK_BACKS = (1, 6, 12, 24)
DEFAULT_SMOOTHING_WINDOWS = (12, 24, 48, 72)
DEFAULT_STEM = "toy_online_lstm_grid"
NUM_TIME_STEPS = 24 * 30
NOISE_STD = 0.2
# Stationary, like the base experiment: the grid measures how look-back and window length
# affect the scheme itself, not how they affect adapting to a change.
REGIMES = ((0, 1.0, 24),)


def build_data() -> tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], float
]:
    """Build the same standardized splits as the base experiment."""

    y = generate_periodic_signal(
        num_time_steps=NUM_TIME_STEPS,
        regimes=REGIMES,
        noise_std=NOISE_STD,
        seed=0,
    )
    data = pd.DataFrame(
        {"values": y},
        index=pd.date_range(
            start="2023-01-01",
            periods=NUM_TIME_STEPS,
            freq="h",
        ),
    )
    data.index.name = "date_time"

    # Same splits as the base experiment: no validation span, because there is no epoch loop
    # to early-stop. Everything up to the test set is walked online.
    processor = DataProcess(
        data=data,
        time_covariates=["hour_of_day"],
        train_split=0.9,
        validation_split=0.0,
        output_col=[0],
    )
    train_data, _, test_data, all_data = processor.get_splits()
    observation_noise_std = float(NOISE_STD / processor.scale_const_std[0])
    return train_data, test_data, all_data, observation_noise_std


def run_configuration(
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    all_data: Dict[str, np.ndarray],
    look_back_len: int,
    smoothing_window: int,
    observation_noise_std: float,
    manual_seed: int = 1,
) -> dict:
    """Run one configuration of the rolling online-LSTM experiment."""

    num_train = len(train_data["y"])
    num_all = len(all_data["y"])
    if look_back_len < 1:
        raise ValueError("look_back_len must be at least 1.")
    if not 1 <= smoothing_window < num_train:
        raise ValueError(
            f"smoothing_window must be between 1 and {num_train - 1}; "
            f"got {smoothing_window}."
        )

    model = Model(
        LstmNetwork(
            look_back_len=look_back_len,
            # The layer input is look_back_len outputs plus one time covariate.
            num_features=2,
            infer_len=24,
            num_layer=1,
            num_hidden_unit=50,
            device="cpu",
            manual_seed=manual_seed,
        ),
        WhiteNoise(std_error=observation_noise_std),
    )
    if not model.lstm_net.smooth:
        raise ValueError("The online loop needs the LSTM smoother to be enabled.")

    # Each window filters D observations plus the extra one-step-ahead target.
    model.lstm_net.num_samples = smoothing_window + 1
    look_back_buffer = LstmLookBackBuffer(
        look_back_len=look_back_len,
        num_time_steps=num_train,
    )

    num_windows = num_train - smoothing_window
    mu_predictions = []
    prediction_indices = []
    started_at = time.perf_counter()

    for window_index in range(num_windows):
        window_start = window_index
        window_end = window_start + smoothing_window + 1
        training_window = make_window(train_data, window_start, window_end)

        model.lstm_net.train()
        mu_filter, _, _ = model.filter(training_window, train_lstm=True)
        mu_predictions.append(mu_filter[-1])
        prediction_indices.append(window_end - 1)

        mu_smooth_lstm, var_smooth_lstm = smooth_window(model)
        look_back_buffer.store(window_start, mu_smooth_lstm, var_smooth_lstm)

        if window_index < num_windows - 1:
            rewind_to_step(model, look_back_buffer, window_start, time_step=1)
        else:
            # Smoothing clears the current recurrent state. The last buffer entry is
            # the filtered state at the end of the training data.
            model.lstm_net.set_lstm_states(
                model.lstm_net.get_lstm_states(smoothing_window)
            )

    # One open-loop forecast over the test span, from the end of the training data to the end
    # of the series.
    model.lstm_net.eval()
    model.lstm_net.num_samples = num_all - num_train
    model.initialize_states_history()
    mu_forecast, _, _ = model.forecast(make_window(all_data, num_train, num_all))

    mu_predictions = np.asarray(mu_predictions).flatten()
    mu_forecast = np.asarray(mu_forecast).flatten()
    prediction_indices = np.asarray(prediction_indices, dtype=int)
    train_observations = train_data["y"].flatten()
    test_observations = test_data["y"].flatten()

    return {
        "look_back_len": look_back_len,
        "smoothing_window": smoothing_window,
        "num_windows": num_windows,
        "observation_noise_std": observation_noise_std,
        "online_mse": float(
            metric.mse(mu_predictions, train_observations[prediction_indices])
        ),
        "test_mse": float(metric.mse(mu_forecast, test_observations)),
        "runtime_seconds": time.perf_counter() - started_at,
    }


def _positive_unique(values: Sequence[int], argument_name: str) -> list[int]:
    normalized = list(dict.fromkeys(values))
    if not normalized or any(value < 1 for value in normalized):
        raise ValueError(f"{argument_name} must contain positive integers.")
    return normalized


def run_sweep(
    look_back_lengths: Sequence[int],
    smoothing_windows: Sequence[int],
    results_path: Path,
    resume: bool = False,
) -> pd.DataFrame:
    """Run every grid cell, persisting one row after each completed configuration."""

    look_back_lengths = _positive_unique(look_back_lengths, "look_back_lengths")
    smoothing_windows = _positive_unique(smoothing_windows, "smoothing_windows")
    train_data, test_data, all_data, observation_noise_std = build_data()

    if resume and results_path.exists():
        results = pd.read_csv(results_path).to_dict("records")
    else:
        results = []
    completed = {
        (int(row["look_back_len"]), int(row["smoothing_window"])) for row in results
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(look_back_lengths) * len(smoothing_windows)
    for smoothing_window in smoothing_windows:
        for look_back_len in look_back_lengths:
            key = (look_back_len, smoothing_window)
            if key in completed:
                print(
                    f"Skipping look_back={look_back_len}, D={smoothing_window} "
                    "(already in results)."
                )
                continue

            print(
                f"Running look_back={look_back_len}, D={smoothing_window} "
                f"({len(completed) + 1}/{total})...",
                flush=True,
            )
            row = run_configuration(
                train_data=train_data,
                test_data=test_data,
                all_data=all_data,
                look_back_len=look_back_len,
                smoothing_window=smoothing_window,
                observation_noise_std=observation_noise_std,
            )
            results.append(row)
            completed.add(key)
            pd.DataFrame(results).to_csv(results_path, index=False)
            print(
                f"  online MSE={row['online_mse']:.6f}, "
                f"test MSE={row['test_mse']:.6f}, "
                f"runtime={row['runtime_seconds']:.1f}s",
                flush=True,
            )

    return pd.DataFrame(results)


def plot_results(
    results: pd.DataFrame,
    look_back_lengths: Sequence[int],
    smoothing_windows: Sequence[int],
    stem: str,
) -> None:
    """Reshape the long-form results and render the two performance heatmaps."""

    look_back_lengths = list(look_back_lengths)
    smoothing_windows = list(smoothing_windows)

    def metric_grid(metric_name: str) -> np.ndarray:
        pivot = results.pivot_table(
            index="smoothing_window",
            columns="look_back_len",
            values=metric_name,
            aggfunc="last",
        )
        return pivot.reindex(
            index=smoothing_windows,
            columns=look_back_lengths,
        ).to_numpy(dtype=float)

    plot_performance_grid(
        look_back_lengths=look_back_lengths,
        smoothing_windows=smoothing_windows,
        online_mse=metric_grid("online_mse"),
        test_mse=metric_grid("test_mse"),
        stem=stem,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lookbacks",
        nargs="+",
        type=int,
        default=list(DEFAULT_LOOK_BACKS),
        help="LSTM look-back lengths (default: %(default)s).",
    )
    parser.add_argument(
        "--smoothing-windows",
        nargs="+",
        type=int,
        default=list(DEFAULT_SMOOTHING_WINDOWS),
        help="Smoothing window lengths D (default: %(default)s).",
    )
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help="Output filename stem under experiments/out/ (default: %(default)s).",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help="Optional CSV path; defaults to experiments/out/<stem>.csv.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed grid cells already present in the results CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    look_back_lengths = _positive_unique(args.lookbacks, "--lookbacks")
    smoothing_windows = _positive_unique(
        args.smoothing_windows, "--smoothing-windows"
    )
    results_path = args.results_csv or output_dir() / f"{args.stem}.csv"

    results = run_sweep(
        look_back_lengths=look_back_lengths,
        smoothing_windows=smoothing_windows,
        results_path=results_path,
        resume=args.resume,
    )
    plot_results(
        results=results,
        look_back_lengths=look_back_lengths,
        smoothing_windows=smoothing_windows,
        stem=args.stem,
    )
    print(f"Results CSV             : {results_path}")
    print(f"Performance grid        : {output_dir() / f'{args.stem}.png'}")


if __name__ == "__main__":
    main()
