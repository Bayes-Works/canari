"""Run baseline anomaly detectors with the same multi-realization evaluation as
``experiments/anomaly_detection_lstm.py``.

Each baseline builds a detector (following the setup of the corresponding
script in ``experiments/zhan/``) and is then evaluated on synthetic anomalies
injected into the full evaluation series. For every magnitude in
``slope_search_space`` we generate ``num_anomaly_realizations`` realizations
and report probability of detection, false-alarm rate, and time to detection
exactly as the SKF experiment does.

Seed-dependent methods (``lstm_ed``, ``tranad``) are fit once per requested
seed; deterministic methods (``prophet``, ``damp``) are fit once. The combined
``summary.json`` exposes ``magnitude_results`` and ``aggregate_by_magnitude``
in the same shape as ``benchmark_anomaly_detection.py`` so the same notebook
aggregation paths work on baselines (``condition`` is the method name).

Usage:
    python -m experiments.baselines_anomaly_detection \
        --experiment_config_path experiments/config/ID_timeseries/LGA008EFAPRG910.yaml
    python -m experiments.baselines_anomaly_detection \
        --experiment_config_path ... --methods "[prophet]" --seeds "[1,2,3]"
"""

from __future__ import annotations

import json
import multiprocessing as mp
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Callable

import fire
import numpy as np
import yaml

from canari.data_process import DataProcess

from experiments.baselines import METHODS
from experiments.utils import prepare_dataset

STEPS_PER_YEAR = 52.0

# Methods whose fits depend on a random seed. For these, the orchestrator runs
# one fit per requested seed; other methods run a single deterministic fit.
SEED_DEPENDENT_METHODS = {"lstm_ed", "tranad"}

# Module-level state used by fork-based worker processes. The parent fills this
# in before spawning the pool, and each forked worker inherits a copy-on-write
# view of the trained detector and pre-generated realizations — no pickling.
_worker_state: dict = {}


METHOD_OPTION_KEYS = {
    "prophet": (
        "online_begin_ratio",
        "changepoint_range",
        "changepoint_delta_threshold",
    ),
    "lstm_ed": (
        "seed",
        "window_size",
        "hidden",
        "num_epochs",
        "batch_size",
        "learning_rate",
        "augmentation_realizations",
        "threshold_multiplier",
        "min_score_threshold",
        "score_smoothing_window",
    ),
    "damp": (
        "window_size",
        "lookahead",
        "threshold_multiplier",
    ),
    "tranad": (
        "seed",
        "window_size",
        "num_epochs",
        "learning_rate",
        "threshold_multiplier",
        "lower_threshold_multiplier",
        "min_score_threshold",
    ),
}


def _collect_options(config: dict, method_name: str) -> dict:
    """Pick method-specific options out of the experiment YAML (``<method>_<key>``)."""
    prefix = f"{method_name}_"
    options: dict = {}
    for key in METHOD_OPTION_KEYS.get(method_name, ()):  # harmless if method unknown
        full = prefix + key
        if full in config:
            options[key] = config[full]
    if method_name == "lstm_ed" and "seed" not in options:
        if "lstm_manual_seed" in config:
            options["seed"] = int(config["lstm_manual_seed"])
    return options


def _worker_init(method_name: str) -> None:
    """Pool initializer run once per forked worker."""
    # PyTorch inference inside Merlion's LSTMED already uses OpenMP/MKL threads
    # internally; stacking process-level parallelism on top oversubscribes
    # cores. Force one thread per worker so the speedup is actually realized.
    if method_name in ("lstm_ed", "tranad"):
        import torch

        torch.set_num_threads(1)


def _detect_one(task: tuple) -> tuple:
    """Worker: run the detector on one realization and return detection stats."""
    mag_key, idx = task
    realization = _worker_state["realizations_by_mag"][mag_key][idx]
    detector = _worker_state["detector"]
    max_timestep_to_detect = _worker_state["max_timestep_to_detect"]

    flags = detector(realization)
    anomaly_t = int(realization["anomaly_timestep"])
    end = min(anomaly_t + int(max_timestep_to_detect), len(flags))
    if end <= anomaly_t:
        return mag_key, False, None
    window = flags[anomaly_t:end]
    if window.any():
        return mag_key, True, int(np.argmax(window))
    return mag_key, False, None


def _evaluate_detector(
    detector: Callable,
    method_name: str,
    all_data: dict,
    synthetic_by_magnitude: dict,
    data_len_years: float,
    max_timestep_to_detect: int,
    n_jobs: int = 1,
) -> dict:
    clean_flags = detector(all_data)
    num_false_alarms = int(np.sum(clean_flags))
    false_rate_yearly = num_false_alarms / data_len_years if data_len_years > 0 else float("nan")

    # Use stringified magnitudes as stable keys to avoid float-equality pitfalls
    # when results cross process boundaries.
    mag_keys = {f"{float(mag):.6f}": float(mag) for mag in synthetic_by_magnitude}
    realizations_by_key = {
        key: synthetic_by_magnitude[mag] for key, mag in mag_keys.items()
    }
    totals = {key: len(realizations_by_key[key]) for key in mag_keys}

    _worker_state["detector"] = detector
    _worker_state["realizations_by_mag"] = realizations_by_key
    _worker_state["max_timestep_to_detect"] = int(max_timestep_to_detect)

    tasks = [
        (key, idx)
        for key, realizations in realizations_by_key.items()
        for idx in range(len(realizations))
    ]
    n_workers = min(max(int(n_jobs), 1), len(tasks)) if tasks else 1
    print(f"  Evaluating {len(tasks)} realization task(s) with {n_workers} worker(s)")

    if n_workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(method_name,),
        ) as pool:
            raw = pool.map(_detect_one, tasks)
    else:
        raw = [_detect_one(task) for task in tasks]

    detected_counts = {key: 0 for key in mag_keys}
    ttd_steps: dict[str, list[int]] = {key: [] for key in mag_keys}
    for mag_key, detected, ttd in raw:
        if detected:
            detected_counts[mag_key] += 1
            if ttd is not None:
                ttd_steps[mag_key].append(ttd)

    results: dict = {}
    for key, mag in mag_keys.items():
        total = totals[key]
        p_det = detected_counts[key] / total if total else 0.0
        steps = ttd_steps[key]
        if steps:
            years = np.asarray(steps, dtype=float) / STEPS_PER_YEAR
            ttd_mean = float(np.nanmean(years))
            ttd_std = float(np.nanstd(years))
        else:
            ttd_mean = float("nan")
            ttd_std = float("nan")

        results[f"mag_{mag:.3f}"] = {
            "probability_of_detection": float(p_det),
            "false_alarm_rate_per_y": float(false_rate_yearly),
            "time_to_detection_years_mean": ttd_mean,
            "time_to_detection_years_std": ttd_std,
            "num_realizations": total,
        }
        print(
            f"  mag={mag:.4f}: P(detect)={p_det:.2f} "
            f"FA/yr={false_rate_yearly:.2f} "
            f"TTD(yr)={ttd_mean:.3f}±{ttd_std:.3f}"
        )

    return {
        "num_false_alarm_clean": num_false_alarms,
        "false_alarm_rate_per_y_clean": float(false_rate_yearly),
        "multi_realization_evaluation": results,
    }


def _collect_magnitude_results(runs: list[dict]) -> list[dict]:
    """Flatten per-run multi-realization results into one row per (method, seed, magnitude)."""
    rows = []
    for run in runs:
        multi_eval = run.get("multi_realization_evaluation", {})
        for mag_key, mag_data in multi_eval.items():
            magnitude = float(mag_key.replace("mag_", ""))
            rows.append(
                {
                    "condition": run["method"],
                    "seed": run["seed"],
                    "anomaly_magnitude": magnitude,
                    "probability_of_detection": mag_data.get("probability_of_detection"),
                    "false_alarm_rate_per_y": mag_data.get("false_alarm_rate_per_y"),
                    "time_to_detection_years_mean": mag_data.get("time_to_detection_years_mean"),
                    "time_to_detection_years_std": mag_data.get("time_to_detection_years_std"),
                    "num_realizations": mag_data.get("num_realizations"),
                }
            )
    return rows


def _aggregate_by_condition_magnitude(magnitude_results: list[dict]) -> list[dict]:
    """Aggregate per-magnitude rows by (method, magnitude) across seeds.

    Mirrors ``experiments/benchmark_anomaly_detection.py`` so the same notebook
    aggregation paths work on baseline outputs.
    """
    groups = defaultdict(list)
    for r in magnitude_results:
        groups[(r["condition"], r["anomaly_magnitude"])].append(r)

    def _finite(values):
        arr = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return arr

    aggregates = []
    for (condition, magnitude), items in sorted(groups.items()):
        p_dets = _finite([r["probability_of_detection"] for r in items])
        fa_rates = _finite([r["false_alarm_rate_per_y"] for r in items])
        ttd_means = _finite([r["time_to_detection_years_mean"] for r in items])
        aggregates.append(
            {
                "condition": condition,
                "anomaly_magnitude": magnitude,
                "probability_of_detection": float(np.mean(p_dets)) if p_dets else None,
                "probability_of_detection_std": float(np.std(p_dets)) if p_dets else None,
                "false_alarm_rate_per_y_mean": float(np.mean(fa_rates)) if fa_rates else None,
                "false_alarm_rate_per_y_std": float(np.std(fa_rates)) if fa_rates else None,
                "time_to_detection_years_mean": float(np.mean(ttd_means)) if ttd_means else None,
                "time_to_detection_years_std": float(np.std(ttd_means)) if ttd_means else None,
                "total_realizations": sum(
                    r["num_realizations"] for r in items if r["num_realizations"] is not None
                ),
                "num_seeds": len({r["seed"] for r in items}),
            }
        )
    return aggregates


def main(
    experiment_config_path: str = "./experiments/config/ID_timeseries/LGA008EFAPRG910.yaml",
    methods: list[str] = ("prophet", "lstm_ed", "damp", "tranad"),
    seeds: list[int] = (1, 2, 3),
    n_jobs: int = 45,
):
    config_path = Path(experiment_config_path)
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    requested = [str(m) for m in methods]
    unknown = [m for m in requested if m not in METHODS]
    if unknown:
        raise ValueError(
            f"Unknown baseline method(s): {unknown}. Available: {sorted(METHODS)}"
        )

    experiment_name = config["experiment_name"]
    output_root = Path(config.get("output_root", "experiments/out"))
    baselines_root = output_root / f"{experiment_name}_baselines"
    baselines_root.mkdir(parents=True, exist_ok=True)
    with (baselines_root / "experiment_config_used.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    dataset = prepare_dataset(
        train_split=float(config["train_split"]),
        anomaly_slope=0.0,
        experiment_config=config,
    )
    data_processor = dataset["data_processor"]
    all_data = dataset["all_data"]

    # Evaluation setup — identical to anomaly_detection_lstm.py.
    total_steps = len(all_data["y"])
    max_timestep_to_detect = int(config.get("max_timestep_to_detect ", 156))
    num_realizations = int(config.get("num_anomaly_realizations", 25))
    anomaly_magnitudes = [
        float(m)
        for m in config.get("slope_search_space", [0.025, 0.05, 0.075, 0.225, 0.5, 0.75, 1.0])
    ]

    test_start_ratio = data_processor.test_start / total_steps
    anomaly_end_ratio = (data_processor.test_end - max_timestep_to_detect) / total_steps
    last_idx = min(max(int(data_processor.test_end) - 1, 0), total_steps - 1)
    data_len_years = (
        data_processor.data.index[last_idx]
        - data_processor.data.index[data_processor.train_start]
    ).days / 365.25

    # Pre-generate realizations once and share across methods.
    synthetic_by_magnitude = {
        mag: DataProcess.add_synthetic_anomaly(
            all_data,
            num_samples=num_realizations,
            slope=[mag / STEPS_PER_YEAR, -mag / STEPS_PER_YEAR],
            anomaly_start=test_start_ratio,
            anomaly_end=anomaly_end_ratio,
        )
        for mag in anomaly_magnitudes
    }

    # Optional threshold calibration: if enabled, each method picks the
    # smallest threshold whose flagger produces FA_rate <= fa_target_per_year
    # on the clean train/val region. No synthetic anomalies -- one scorer
    # pass per method.
    if bool(config.get("calibrate_threshold", True)):
        dataset["calibration"] = {
            "all_data": all_data,
            "train_val_end_idx": int(data_processor.test_start),
            "steps_per_year": STEPS_PER_YEAR,
            "fa_target_per_year": float(config.get("calibration_fa_target_per_y", 0.0)),
        }

    requested_seeds = [int(s) for s in seeds]
    all_runs: list[dict] = []
    per_method_runs: dict[str, list[dict]] = {m: [] for m in requested}
    failed_runs: list[dict] = []
    for method_name in requested:
        seed_dependent = method_name in SEED_DEPENDENT_METHODS
        method_seeds: list[int | None] = (
            list(requested_seeds) if seed_dependent else [None]
        )
        method_dir = baselines_root / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        for seed in method_seeds:
            label = f"{method_name}" + (f" (seed={seed})" if seed is not None else "")
            print(f"\n{'=' * 60}\nRunning {label}\n{'=' * 60}")

            options = _collect_options(config, method_name)
            if seed is not None:
                options["seed"] = int(seed)
            run_dir = method_dir if seed is None else method_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                print(f"-- Fitting {label} (options: {options}) --")
                detector, info = METHODS[method_name](dataset, options)

                # DAMP's backend initializes an OpenMP threadpool at fit time, which
                # makes forking unsafe — run it serially.
                method_n_jobs = 1 if method_name == "damp" else int(n_jobs)
                print(
                    f"-- Evaluating {label} on synthetic anomalies "
                    f"(n_jobs={method_n_jobs}) --"
                )
                eval_summary = _evaluate_detector(
                    detector,
                    method_name=method_name,
                    all_data=all_data,
                    synthetic_by_magnitude=synthetic_by_magnitude,
                    data_len_years=data_len_years,
                    max_timestep_to_detect=max_timestep_to_detect,
                    n_jobs=method_n_jobs,
                )

                run_summary = {
                    "experiment_name": experiment_name,
                    "method": method_name,
                    "seed": seed,
                    "options": options,
                    "fit_info": info,
                    **eval_summary,
                }
                summary_path = run_dir / "summary.json"
                with summary_path.open("w") as f:
                    json.dump(run_summary, f, indent=2, default=str)
                print(f"  Saved: {summary_path}")

                all_runs.append(run_summary)
                per_method_runs[method_name].append(run_summary)
            except Exception as exc:
                failure = {
                    "experiment_name": experiment_name,
                    "method": method_name,
                    "seed": seed,
                    "options": options,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                failure_path = run_dir / "failure.json"
                with failure_path.open("w") as f:
                    json.dump(failure, f, indent=2, default=str)
                failed_runs.append(failure)
                print(f"  FAILED: {failure['error']}")
                print(f"  Failure details saved to: {failure_path}")
                continue

    magnitude_results = _collect_magnitude_results(all_runs)
    aggregate_by_magnitude = _aggregate_by_condition_magnitude(magnitude_results)

    combined_path = baselines_root / "summary.json"
    status = "completed_with_failures" if failed_runs else "completed"
    with combined_path.open("w") as f:
        json.dump(
            {
                "experiment_name": experiment_name,
                "status": status,
                "config_path": str(config_path),
                "methods": requested,
                "seeds": requested_seeds,
                "seed_dependent_methods": sorted(
                    m for m in requested if m in SEED_DEPENDENT_METHODS
                ),
                "per_method_runs": per_method_runs,
                "failed_runs": failed_runs,
                "magnitude_results": magnitude_results,
                "aggregate_by_magnitude": aggregate_by_magnitude,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nAll baselines saved to: {baselines_root}")
    print(f"Combined summary: {combined_path}")
    return {
        "experiment_name": experiment_name,
        "status": status,
        "config_path": str(config_path),
        "baselines_root": str(baselines_root),
        "methods": requested,
        "seeds": requested_seeds,
        "failed_runs": failed_runs,
        "magnitude_results": magnitude_results,
        "aggregate_by_magnitude": aggregate_by_magnitude,
    }


if __name__ == "__main__":
    fire.Fire(main)
