"""LSTM encoder-decoder anomaly detector using Merlion.

Reference: ``experiments/zhan/lstmed.py``. Builds an augmented training series
by concatenating the clean train split with ``augmentation_realizations`` copies
containing randomly injected trend anomalies, trains Merlion's ``LSTMED`` on
the augmented series with matching anomaly labels, and calibrates a fixed score
threshold at ``max(threshold_multiplier * max_score_on_clean, min_score_threshold)``.

If ``dataset["calibration"]`` is provided by the orchestrator, the threshold
is instead chosen as the smallest value whose flagger produces
``FA_rate_per_y <= fa_target_per_year`` on the clean train/val region (see
``calibrate.py``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from merlion.models.anomaly.lstm_ed import LSTMED, LSTMEDConfig
from merlion.utils import TimeSeries

from .calibrate import calibrate_threshold


def _sanitize(y: np.ndarray) -> np.ndarray:
    y = pd.Series(np.asarray(y, dtype=float).flatten())
    return y.interpolate("linear", limit_direction="both").to_numpy()


def _build_augmented_training(
    train_y: np.ndarray,
    *,
    seed: int,
    augmentation_realizations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate clean train with anomaly-augmented copies (zhan/lstmed.py)."""

    rng = np.random.default_rng(seed)
    train = _sanitize(train_y).astype(float)
    train_std = float(np.nanstd(train, ddof=1))
    if not np.isfinite(train_std) or train_std <= 0:
        train_std = 1.0

    n = train.shape[0]
    lower = max(n // 4, 1)
    upper = max(n * 3 // 8, lower + 1)

    values = [train]
    labels = [np.zeros(n, dtype=np.int64)]
    for _ in range(int(augmentation_realizations)):
        slope = rng.uniform(-train_std / 52.0, train_std / 52.0)
        if rng.random() <= 0.5:
            slope = 0.0
        anomaly_start = int(rng.integers(lower, upper))
        baseline = np.arange(n, dtype=float) * slope
        baseline[anomaly_start:] -= baseline[anomaly_start]
        baseline[:anomaly_start] = 0.0
        values.append(train + baseline)
        lbl = np.zeros(n, dtype=np.int64)
        if slope != 0.0:
            lbl[anomaly_start:] = 1
        labels.append(lbl)

    all_values = np.concatenate(values)
    all_labels = np.concatenate(labels)

    new_index = pd.date_range(
        start="1970-01-01", periods=len(all_values), freq="D"
    )
    value_df = pd.DataFrame({"value": all_values}, index=new_index)
    label_df = pd.DataFrame({"anomaly": all_labels}, index=new_index)
    return value_df, label_df


def _raw_scores(model: LSTMED, y: np.ndarray, time: pd.DatetimeIndex) -> np.ndarray:
    df = pd.DataFrame({"value": _sanitize(y)}, index=pd.to_datetime(time))
    scores_ts = model.get_anomaly_label(time_series=TimeSeries.from_pd(df))
    return scores_ts.univariates[scores_ts.names[0]].np_values


def _aligned_scores(model: LSTMED, y: np.ndarray, time) -> np.ndarray:
    """Return per-timestep scores aligned to ``y`` (NaN where undefined)."""
    y = np.asarray(y).flatten()
    raw = _raw_scores(model, y, time)
    scores = np.full(len(y), np.nan)
    if raw.shape[0] == len(y):
        scores[:] = raw
    elif raw.shape[0] < len(y):
        scores[-raw.shape[0]:] = raw
    else:
        scores[:] = raw[: len(y)]
    return scores


# Opt-in trailing rolling-mean smoother for anomaly scores. No-op when window<=1.
# To remove: delete this helper, the `score_smoothing_window` option read, the
# smoothing of `train_scores`, the scorer wrap, and the info-dict entry.
def _smooth_scores(scores: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return scores
    smoothed = pd.Series(scores).rolling(window, min_periods=1).mean().to_numpy()
    smoothed[~np.isfinite(scores)] = np.nan
    return smoothed


def build_detector(dataset: dict, options: dict) -> tuple[Callable, dict]:
    seed = int(options.get("seed", 0))
    window_size = int(options.get("window_size", 52))
    hidden = int(options.get("hidden", 32))
    num_epochs = int(options.get("num_epochs", 100))
    batch_size = int(options.get("batch_size", 64))
    learning_rate = float(options.get("learning_rate", 1e-3))
    augmentation_realizations = int(options.get("augmentation_realizations", 50))
    threshold_multiplier = float(options.get("threshold_multiplier", 1.1))
    min_score_threshold = float(options.get("min_score_threshold", 0.1))
    score_smoothing_window = int(options.get("score_smoothing_window", 1))

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    train_y = np.asarray(dataset["train_data"]["y"]).flatten()
    value_df, label_df = _build_augmented_training(
        train_y,
        seed=seed,
        augmentation_realizations=augmentation_realizations,
    )

    model = LSTMED(
        LSTMEDConfig(
            num_epochs=num_epochs,
            sequence_len=window_size,
            hidden_size=hidden,
            batch_size=batch_size,
            lr=learning_rate,
        )
    )
    model.train(
        train_data=TimeSeries.from_pd(value_df),
        anomaly_labels=TimeSeries.from_pd(label_df),
    )

    # Threshold anchor: max score on the clean train prefix of the augmented series.
    train_scores = _raw_scores(
        model,
        value_df["value"].to_numpy()[: train_y.shape[0]],
        value_df.index[: train_y.shape[0]],
    )
    train_scores = _smooth_scores(train_scores, score_smoothing_window)
    valid = train_scores[np.isfinite(train_scores)]
    if valid.size == 0:
        raise ValueError("LSTM-ED produced no finite scores on train prefix.")
    train_score_max = float(np.max(valid))

    def scorer(eval_input: dict) -> np.ndarray:
        return _smooth_scores(
            _aligned_scores(model, eval_input["y"], eval_input["time"]),
            score_smoothing_window,
        )

    def flagger(scores: np.ndarray, threshold: float) -> np.ndarray:
        flags = np.zeros(len(scores), dtype=bool)
        finite = np.isfinite(scores)
        flags[finite] = scores[finite] > threshold
        return flags

    calibration_cfg = dataset.get("calibration")
    calibration_result = None
    if calibration_cfg:
        calibration_result = calibrate_threshold(
            scorer=scorer, flagger=flagger, **calibration_cfg
        )
        score_threshold = float(calibration_result["threshold"])
    else:
        score_threshold = float(
            max(threshold_multiplier * train_score_max, min_score_threshold)
        )

    def detector(eval_input: dict) -> np.ndarray:
        return flagger(scorer(eval_input), score_threshold)

    info = {
        "window_size": window_size,
        "hidden": hidden,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "augmentation_realizations": augmentation_realizations,
        "threshold_multiplier": threshold_multiplier,
        "min_score_threshold": min_score_threshold,
        "score_smoothing_window": score_smoothing_window,
        "train_score_max": train_score_max,
        "score_threshold": score_threshold,
        "threshold_rule": (
            "score > max(threshold_multiplier * max_score_on_train, min_score_threshold)"
            if calibration_result is None
            else "score > smallest threshold with FA_rate <= target on clean train/val"
        ),
    }
    if calibration_result is not None:
        info["calibration"] = calibration_result
    return detector, info
