"""Run baseline anomaly detectors against the same synthetic anomalies
evaluated by anomaly_detection_lstm.py.

Currently implements:
    - prophet  -> Prophet forecast residuals.
    - damp     -> Discord Aware Matrix Profile (Lu et al., KDD 2022);
                  approximate left matrix profile over [train || eval] with
                  sp_index = len(train).
    - lstm_ed  -> LSTM encoder-decoder autoencoder (Malhotra 2016);
                  reconstruction MSE.
    - tranad   -> TranAD (Tuli et al., VLDB 2022, minimal univariate
                  reimplementation); reconstruction MSE.

Each method defines its own threshold rule (matching zhan's conventions
where applicable) but exposes a unified "raw > coefficient" interface: the
method's raw_scorer returns a per-step score already normalised so that a
single scalar coefficient decides the flag. Rules:
    - Prophet : raw = abs(y - yhat) / (yhat_upper - yhat)
                -> detect when raw > c (c scales the Prophet CI half-width)
    - LSTM-ED : raw = score / max(train_scores)
                -> detect when raw > c (zhan's max-ratio rule)
    - TranAD  : raw = loss  / max(train_loss)
                -> detect when raw > c (one-sided max-ratio rule)
    - DAMP    : raw = (score - mean(train_mp)) / std(train_mp)
                -> detect when raw > c (sigma-multiplier, zhan has no DAMP)

The coefficient ``c`` is tuned per method on train+val (mirroring
anomaly_detection_lstm.py's tuning-on-train_val / eval-on-test split); each
method carries its own ``coefficient_grid``. The winner maximises P(detect)
over synthetic anomalies injected into train_val subject to zero false
alarms on clean train_val.

Synthetic anomalies are regenerated via DataProcess.add_synthetic_anomaly,
which seeds numpy internally (seed=5), so each magnitude yields the exact
same realizations used by anomaly_detection_lstm.py for a given config.

Output schema matches anomaly_detection_lstm.py so downstream aggregators
can consume either source interchangeably:
    multi_realization_evaluation[mag_key] = {
        probability_of_detection, false_alarm_rate_per_y,
        time_to_detection_years_mean, time_to_detection_years_std,
        num_realizations, ...
    }

Usage:
    python -m experiments.baselines_anomaly_detection \\
        --experiment_config_path experiments/config/OOD_timeseries/LGA002EFAPRG910.yaml

    # single method:
    python -m experiments.baselines_anomaly_detection \\
        --experiment_config_path ... --methods "[prophet]"
"""

import json
import logging
import os
from pathlib import Path
from typing import Callable

import fire
import numpy as np
import pandas as pd
import yaml

from canari.data_process import DataProcess

try:
    from experiments.utils import prepare_dataset
except ModuleNotFoundError:
    from utils import prepare_dataset

try:
    from experiments.baselines import damp as _damp_mod
    from experiments.baselines import lstm_ed as _lstm_ed_mod
    from experiments.baselines import tranad as _tranad_mod
except ModuleNotFoundError:
    from baselines import damp as _damp_mod
    from baselines import lstm_ed as _lstm_ed_mod
    from baselines import tranad as _tranad_mod


STEPS_PER_YEAR = 52.0  # weekly data, matches anomaly_detection_lstm.py


def _detection_in_window(
    detections: np.ndarray,
    anomaly_timestep: int,
    max_timestep_to_detect: int,
) -> tuple[bool, int | None]:
    """Return (detected, offset_in_steps_from_anomaly_start) for one realization."""
    det = np.asarray(detections, dtype=bool)
    end = min(anomaly_timestep + int(max_timestep_to_detect), len(det))
    if end <= anomaly_timestep:
        return False, None
    window = det[anomaly_timestep:end]
    if window.any():
        return True, int(np.argmax(window))
    return False, None


def _build_prophet_fit(
    train_times: np.ndarray,
    train_y: np.ndarray,
    eval_times: np.ndarray,
) -> dict:
    """Fit Prophet once and return normalised per-step residuals so that
    detection reduces to ``raw > coefficient``.

    raw(t) = |y(t) - yhat(t)| / (yhat_upper(t) - yhat(t))

    where the denominator is Prophet's own predictive-interval half-width
    (the default 80% CI). A coefficient of 1.0 reproduces zhan's
    interval-based flag; larger values widen the band.
    """
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    from prophet import Prophet

    train_df = pd.DataFrame(
        {"ds": pd.to_datetime(train_times), "y": np.asarray(train_y).flatten()}
    ).dropna()
    model = Prophet()
    model.fit(train_df)

    future = pd.DataFrame({"ds": pd.to_datetime(eval_times)})
    forecast = model.predict(future)
    yhat = forecast["yhat"].to_numpy()
    half_width = forecast["yhat_upper"].to_numpy() - yhat
    half_width = np.where(half_width > 0, half_width, np.nan)

    def raw_scorer(eval_y: np.ndarray) -> np.ndarray:
        y = np.asarray(eval_y).flatten()
        n = min(len(y), len(yhat))
        raw = np.full(len(y), np.nan)
        raw[:n] = np.abs(y[:n] - yhat[:n]) / half_width[:n]
        raw[np.isnan(y)] = np.nan
        return raw

    return {
        "raw_scorer": raw_scorer,
        "coefficient_grid": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "coefficient_name": "interval_width_coefficient",
        "threshold_rule": "abs(y - yhat) > c * (yhat_upper - yhat)",
        "window_size": 1,
    }


def _make_detector(
    fit: dict, coefficient: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Turn a fit dict + coefficient into a per-step bool detector.

    All methods expose raw scores pre-normalised so that detection is a
    simple ``raw > coefficient`` comparison.
    """
    raw_scorer = fit["raw_scorer"]
    c = float(coefficient)

    def detector(eval_y: np.ndarray) -> np.ndarray:
        raw = raw_scorer(eval_y)
        flags = np.zeros(len(eval_y), dtype=bool)
        valid = np.isfinite(raw)
        flags[valid] = raw[valid] > c
        return flags

    return detector


def _tune_threshold_coefficient(
    method_name: str,
    fit: dict,
    train_val: dict,
    slope_val: float,
    cdf_num_anomaly: int,
    cdf_anomaly_start: float,
    cdf_anomaly_end: float,
    max_timestep_to_detect: int,
) -> dict:
    """Grid-search the method's detection coefficient on the train+val region.

    For each candidate ``c`` in ``fit["coefficient_grid"]``:
        - count false alarms on clean train_val,
        - inject synthetic anomalies (slope +/- slope_val/52) and compute
          detection rate in a window of length ``max_timestep_to_detect``
          after each anomaly.

    Selection: max P_DET over candidates with zero false alarms; if none
    achieve zero FAs, fall back to (min FA, max P_DET).
    """
    grid = list(fit["coefficient_grid"])
    name = fit["coefficient_name"]

    tv_y = np.asarray(train_val["y"]).flatten()
    raw_clean = fit["raw_scorer"](tv_y)

    slope_anomaly = slope_val / 52.0
    synthetic = DataProcess.add_synthetic_anomaly(
        train_val,
        num_samples=cdf_num_anomaly,
        slope=[slope_anomaly, -slope_anomaly],
        anomaly_start=cdf_anomaly_start,
        anomaly_end=cdf_anomaly_end,
    )
    anomaly_ts = [int(r["anomaly_timestep"]) for r in synthetic]
    synth_raw = [fit["raw_scorer"](r["y"]) for r in synthetic]

    grid_rows = []
    for c in grid:
        c_f = float(c)
        valid_clean = np.isfinite(raw_clean)
        fa = int(np.sum(raw_clean[valid_clean] > c_f))

        n_detected = 0
        for raw, ts in zip(synth_raw, anomaly_ts):
            det_flags = np.zeros(len(raw), dtype=bool)
            valid = np.isfinite(raw)
            det_flags[valid] = raw[valid] > c_f
            detected, _ = _detection_in_window(
                det_flags, ts, max_timestep_to_detect
            )
            if detected:
                n_detected += 1
        p_det = n_detected / max(1, len(synthetic))

        grid_rows.append(
            {
                name: c_f,
                "false_alarms": fa,
                "probability_of_detection": float(p_det),
            }
        )
        print(
            f"  [{method_name}] {name}={c_f:.3f}: "
            f"FA={fa}  P(detect)={p_det:.3f}"
        )

    zero_fa = [r for r in grid_rows if r["false_alarms"] == 0]
    if zero_fa:
        best = max(
            zero_fa,
            key=lambda r: (r["probability_of_detection"], r[name]),
        )
        selection = "max_pdet_zero_fa"
    else:
        best = max(
            grid_rows,
            key=lambda r: (
                -r["false_alarms"],
                r["probability_of_detection"],
            ),
        )
        selection = "min_fa_fallback"

    print(
        f"  [{method_name}] selected {name}={best[name]:.3f} "
        f"(rule={selection}, FA={best['false_alarms']}, "
        f"P(detect)={best['probability_of_detection']:.3f})"
    )

    return {
        "coefficient_name": name,
        "threshold_rule": fit.get("threshold_rule"),
        "optimal_coefficient": float(best[name]),
        "selection_rule": selection,
        "tuning_slope": float(slope_val),
        "num_realizations": int(len(synthetic)),
        "grid": grid_rows,
    }


def _evaluate_method(
    method_name: str,
    scorer: Callable[[np.ndarray], np.ndarray],
    clean_y: np.ndarray,
    data_len_years: float,
    anomaly_magnitudes: list[float],
    num_realizations: int,
    max_timestep_to_detect: int,
    test_start_ratio: float,
    anomaly_end_ratio: float,
    all_data: dict,
) -> dict:
    clean_detections = scorer(clean_y)
    num_false_alarm_total = int(np.sum(clean_detections))
    false_rate_yearly = (
        num_false_alarm_total / data_len_years if data_len_years > 0 else None
    )

    multi_eval: dict[str, dict] = {}
    for mag in anomaly_magnitudes:
        mag_key = f"mag_{mag:.3f}"
        mag_anomaly = mag / 52.0
        synthetic = DataProcess.add_synthetic_anomaly(
            all_data,
            num_samples=num_realizations,
            slope=[mag_anomaly, -mag_anomaly],
            anomaly_start=test_start_ratio,
            anomaly_end=anomaly_end_ratio,
        )

        ttd_steps: list[int] = []
        n_detected = 0
        for realization in synthetic:
            detections = scorer(realization["y"])
            anomaly_timestep = int(realization["anomaly_timestep"])
            detected, offset = _detection_in_window(
                detections, anomaly_timestep, max_timestep_to_detect
            )
            if detected:
                n_detected += 1
                ttd_steps.append(offset)

        total = len(synthetic)
        detection_rate = n_detected / total if total else 0.0
        if ttd_steps:
            ttd_years = np.asarray(ttd_steps, dtype=float) / STEPS_PER_YEAR
            ttd_mean = float(np.nanmean(ttd_years))
            ttd_std = float(np.nanstd(ttd_years))
        else:
            ttd_mean = float("nan")
            ttd_std = float("nan")

        multi_eval[mag_key] = {
            "probability_of_detection": detection_rate,
            "false_alarm_rate_per_y": false_rate_yearly,
            "time_to_detection_years_mean": ttd_mean,
            "time_to_detection_years_std": ttd_std,
            "num_realizations": total,
            "plot_directory": None,
        }
        print(
            f"  [{method_name}] mag={mag:.4f}: "
            f"P(detect)={detection_rate:.2f}  "
            f"FA/yr={false_rate_yearly:.2f}  "
            f"TTD(yr)={ttd_mean:.3f}±{ttd_std:.3f}"
        )

    return {
        "method": method_name,
        "num_false_alarm_clean": num_false_alarm_total,
        "false_alarm_rate_per_y_clean": false_rate_yearly,
        "multi_realization_evaluation": multi_eval,
    }


def main(
    experiment_config_path: str = "./experiments/config/ID_timeseries/LGA008EFAPRG910.yaml",
    methods: list[str] = ("prophet", "damp", "lstm_ed", "tranad"),
    damp_window: int = 16,
    lstm_ed_window: int = 30,
    tranad_window: int = 10,
):
    """Run baseline anomaly detectors on the synthetic anomalies defined by
    the YAML config.

    Args:
        experiment_config_path: Path to the experiment config (same format
            as anomaly_detection_lstm.py).
        methods: Which baselines to run. Subset of {"prophet",
            "damp", "lstm_ed", "tranad"}.
        damp_window: Subsequence length m for DAMP. Default 16
            (~4 months of weekly data).
        lstm_ed_window: Window length for LSTM-ED. Default 30.
        tranad_window: Window length for TranAD. Default 10 (paper value).
    """
    config_path = Path(experiment_config_path)
    with config_path.open("r") as f:
        experiment_config = yaml.safe_load(f)

    experiment_name = experiment_config["experiment_name"]
    output_root = Path(experiment_config.get("output_root", "experiments/out"))
    baselines_root = output_root / f"{experiment_name}_baselines"
    baselines_root.mkdir(parents=True, exist_ok=True)

    dataset = prepare_dataset(
        train_split=float(experiment_config["train_split"]),
        anomaly_slope=0.0,
        experiment_config=experiment_config,
    )
    data_processor = dataset["data_processor"]
    train_data = dataset["train_data"]
    validation_data = dataset["validation_data"]
    all_data = dataset["all_data"]
    train_val = dataset["train_val"]

    clean_y = np.asarray(all_data["y"]).flatten()
    train_y = np.asarray(train_data["y"]).flatten()
    val_y = np.asarray(validation_data["y"]).flatten()
    train_times = np.asarray(train_data["time"])
    all_times = np.asarray(all_data["time"])[: len(clean_y)]

    tuning_slope = float(experiment_config.get("slope", 0.075))
    cdf_num_anomaly = int(experiment_config.get("cdf_num_anomaly", 50))
    cdf_anomaly_start = float(experiment_config.get("cdf_anomaly_start", 0.25))
    cdf_anomaly_end = float(experiment_config.get("cdf_anomaly_end", 0.75))

    total_eval_steps = len(clean_y)
    max_timestep_to_detect = int(
        experiment_config.get("max_timestep_to_detect ", 156)
    )
    test_start_ratio = data_processor.test_start / total_eval_steps
    anomaly_end_ratio = (
        data_processor.test_end - max_timestep_to_detect
    ) / total_eval_steps

    num_realizations = int(experiment_config.get("num_anomaly_realizations", 25))
    anomaly_magnitudes = list(
        experiment_config.get(
            "slope_search_space",
            [0.025, 0.05, 0.075, 0.225, 0.5, 0.75, 1.0],
        )
    )

    def _timestamp_at_exclusive_end(end_idx: int):
        last_idx = min(max(end_idx - 1, 0), len(data_processor.data.index) - 1)
        return data_processor.data.index[last_idx]

    data_len_years = (
        _timestamp_at_exclusive_end(data_processor.test_end)
        - data_processor.data.index[data_processor.train_start]
    ).days / 365.25

    used_config_path = baselines_root / "experiment_config_used.yaml"
    with used_config_path.open("w") as f:
        yaml.safe_dump(experiment_config, f, sort_keys=False)

    method_summaries: dict[str, dict] = {}
    for method in methods:
        method_dir = baselines_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}\nRunning method: {method}\n{'=' * 60}")

        seed = int(experiment_config.get("lstm_manual_seed", 0))
        if method == "prophet":
            fit = _build_prophet_fit(
                train_times=train_times,
                train_y=train_y,
                eval_times=all_times,
            )
        elif method == "damp":
            fit = _damp_mod.build_scorer(
                train_y=train_y,
                val_y=val_y,
                window_size=int(damp_window),
            )
        elif method == "lstm_ed":
            fit = _lstm_ed_mod.build_scorer(
                train_y=train_y,
                window_size=int(lstm_ed_window),
                seed=seed,
            )
        elif method == "tranad":
            fit = _tranad_mod.build_scorer(
                train_y=train_y,
                window_size=int(tranad_window),
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")

        print(f"-- Tuning threshold coefficient on train_val ({method}) --")
        threshold_tuning = _tune_threshold_coefficient(
            method_name=method,
            fit=fit,
            train_val=train_val,
            slope_val=tuning_slope,
            cdf_num_anomaly=cdf_num_anomaly,
            cdf_anomaly_start=cdf_anomaly_start,
            cdf_anomaly_end=cdf_anomaly_end,
            max_timestep_to_detect=max_timestep_to_detect,
        )
        scorer = _make_detector(fit, threshold_tuning["optimal_coefficient"])

        summary = _evaluate_method(
            method_name=method,
            scorer=scorer,
            clean_y=clean_y,
            data_len_years=data_len_years,
            anomaly_magnitudes=anomaly_magnitudes,
            num_realizations=num_realizations,
            max_timestep_to_detect=max_timestep_to_detect,
            test_start_ratio=test_start_ratio,
            anomaly_end_ratio=anomaly_end_ratio,
            all_data=all_data,
        )

        method_summary = {
            "experiment_name": experiment_name,
            "method": method,
            "config_path": str(config_path),
            **summary,
            "threshold_coefficient_tuning": threshold_tuning,
        }
        if method == "damp":
            method_summary["damp_window"] = int(damp_window)
        elif method == "lstm_ed":
            method_summary["lstm_ed_window"] = int(lstm_ed_window)
        elif method == "tranad":
            method_summary["tranad_window"] = int(tranad_window)

        summary_path = method_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(method_summary, f, indent=2, default=str)
        method_summaries[method] = method_summary
        print(f"  Saved: {summary_path}")

    combined_path = baselines_root / "summary.json"
    with combined_path.open("w") as f:
        json.dump(
            {
                "experiment_name": experiment_name,
                "config_path": str(config_path),
                "methods": list(methods),
                "per_method": method_summaries,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nAll baselines saved to: {baselines_root}")
    print(f"Combined summary: {combined_path}")


if __name__ == "__main__":
    fire.Fire(main)
