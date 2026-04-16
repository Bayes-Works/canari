"""Benchmark anomaly_detection_lstm across multiple seeds and global/local weights.

Usage:
    python -m experiments.benchmark_anomaly_detection --experiment_config_path experiments/config/LTU012ESAP-E020.yaml
    python -m experiments.benchmark_anomaly_detection --experiment_config_path experiments/config/LTU012ESAP-E020.yaml --seeds "[1,2,3,4,5]"
"""

import copy
import json
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import yaml

try:
    from experiments.anomaly_detection_lstm import main as run_anomaly_detection
except ModuleNotFoundError:
    from anomaly_detection_lstm import main as run_anomaly_detection


def _read_summary(output_dir: Path) -> dict:
    summary_path = output_dir / "summary.json"
    with summary_path.open("r") as f:
        return json.load(f)


def _run_single(
    base_config: dict,
    seed: int,
    condition: str,
    global_params,
    benchmark_root: Path,
) -> dict:
    """Create a modified config, run anomaly detection, and return the summary."""
    config = copy.deepcopy(base_config)
    config["lstm_manual_seed"] = seed
    config["lstm_global_params"] = global_params
    if condition == "local":
        config["lstm_num_layer"] = 1
    config["experiment_name"] = (
        f"{base_config['experiment_name']}_benchmark_{condition}_seed{seed}"
    )
    config["output_root"] = str(benchmark_root / "runs")

    output_root = Path(config["output_root"])
    output_dir = output_root / config["experiment_name"]
    temp_config_dir = benchmark_root / "temp_configs"
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_config_dir / f"{config['experiment_name']}.yaml"
    with temp_config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"\n{'=' * 60}")
    print(f"Running: condition={condition}, seed={seed}")
    print(f"{'=' * 60}")

    run_anomaly_detection(experiment_config_path=str(temp_config_path))

    return _read_summary(output_dir)


def _collect_magnitude_results(runs: list[dict]) -> list[dict]:
    """Extract per-magnitude results from all runs."""
    rows = []
    for run in runs:
        condition = run["condition"]
        seed = run["seed"]
        multi_eval = run["summary"].get("multi_realization_evaluation", {})
        for mag_key, mag_data in multi_eval.items():
            magnitude = float(mag_key.replace("mag_", ""))
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "anomaly_magnitude": magnitude,
                    "probability_of_detection": mag_data.get(
                        "probability_of_detection"
                    ),
                    "false_alarm_rate_per_y": mag_data.get(
                        "false_alarm_rate_per_y"
                    ),
                    "time_to_detection_years_mean": mag_data.get(
                        "time_to_detection_years_mean"
                    ),
                    "time_to_detection_years_std": mag_data.get(
                        "time_to_detection_years_std"
                    ),
                    "num_realizations": mag_data.get("num_realizations"),
                }
            )
    return rows


def _aggregate_by_condition_magnitude(magnitude_results: list[dict]) -> list[dict]:
    """Aggregate per-magnitude results by (condition, anomaly_magnitude) across seeds."""
    groups = defaultdict(list)
    for r in magnitude_results:
        groups[(r["condition"], r["anomaly_magnitude"])].append(r)

    aggregates = []
    for (condition, magnitude), items in sorted(groups.items()):
        p_dets = [
            r["probability_of_detection"]
            for r in items
            if r["probability_of_detection"] is not None
        ]
        fa_rates = [
            r["false_alarm_rate_per_y"]
            for r in items
            if r["false_alarm_rate_per_y"] is not None
        ]
        ttd_means = [
            r["time_to_detection_years_mean"]
            for r in items
            if r["time_to_detection_years_mean"] is not None
        ]
        aggregates.append(
            {
                "condition": condition,
                "anomaly_magnitude": magnitude,
                "probability_of_detection": (
                    float(np.mean(p_dets)) if p_dets else None
                ),
                "probability_of_detection_std": (
                    float(np.std(p_dets)) if p_dets else None
                ),
                "false_alarm_rate_per_y_mean": (
                    float(np.mean(fa_rates)) if fa_rates else None
                ),
                "false_alarm_rate_per_y_std": (
                    float(np.std(fa_rates)) if fa_rates else None
                ),
                "time_to_detection_years_mean": (
                    float(np.mean(ttd_means)) if ttd_means else None
                ),
                "time_to_detection_years_std": (
                    float(np.std(ttd_means)) if ttd_means else None
                ),
                "total_realizations": sum(
                    r["num_realizations"]
                    for r in items
                    if r["num_realizations"] is not None
                ),
                "num_seeds": len({r["seed"] for r in items}),
            }
        )
    return aggregates


def _format_value(value, fmt=".4f"):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:{fmt}}"
    return str(value)


def _print_aggregate_table(aggregates: list[dict], condition_names: list[str]):
    """Print a formatted table of per-magnitude aggregate results."""
    columns = [
        ("Condition", "condition", None),
        ("Magnitude", "anomaly_magnitude", ".3f"),
        ("P(detect)", "probability_of_detection", ".2f"),
        ("P(det) std", "probability_of_detection_std", ".3f"),
        ("FA/yr mean", "false_alarm_rate_per_y_mean", ".3f"),
        ("FA/yr std", "false_alarm_rate_per_y_std", ".3f"),
        ("TTD(yr) mean", "time_to_detection_years_mean", ".3f"),
        ("TTD(yr) std", "time_to_detection_years_std", ".3f"),
        ("N(real)", "total_realizations", None),
        ("N(seeds)", "num_seeds", None),
    ]

    widths = []
    for header, key, fmt in columns:
        col_values = [
            _format_value(r[key], fmt) if fmt else str(r[key]) for r in aggregates
        ]
        widths.append(max(len(header), max(len(v) for v in col_values)))

    header_line = "  ".join(h.ljust(w) for (h, _, _), w in zip(columns, widths))
    print(header_line)
    print("-" * len(header_line))

    for r in aggregates:
        row_values = []
        for _, key, fmt in columns:
            val = r[key]
            row_values.append(_format_value(val, fmt) if fmt else str(val))
        print("  ".join(v.ljust(w) for v, w in zip(row_values, widths)))


def _print_condition_summary(aggregates: list[dict], condition_names: list[str]):
    """Print overall summary statistics per condition."""
    print(f"\n{'=' * 80}")
    print("CONDITION SUMMARY")
    print(f"{'=' * 80}")

    for cond in condition_names:
        cond_aggs = [a for a in aggregates if a["condition"] == cond]
        n_magnitudes = len(cond_aggs)
        total_realizations = sum(a["total_realizations"] for a in cond_aggs)
        p_dets = [
            a["probability_of_detection"]
            for a in cond_aggs
            if a["probability_of_detection"] is not None
        ]

        print(f"\n--- {cond.upper()} ({n_magnitudes} magnitudes, {total_realizations} total realizations) ---")
        if p_dets:
            print(
                f"  P(detect) range:   [{min(p_dets):.2f}, {max(p_dets):.2f}]  "
                f"mean={np.mean(p_dets):.2f}"
            )

        ttd_means = [
            a["time_to_detection_years_mean"]
            for a in cond_aggs
            if a["time_to_detection_years_mean"] is not None
        ]
        if ttd_means:
            print(
                f"  TTD(yr) range:     [{min(ttd_means):.3f}, {max(ttd_means):.3f}]  "
                f"mean={np.mean(ttd_means):.3f}"
            )

        fa_means = [
            a["false_alarm_rate_per_y_mean"]
            for a in cond_aggs
            if a["false_alarm_rate_per_y_mean"] is not None
        ]
        if fa_means:
            print(
                f"  FA/yr range:       [{min(fa_means):.3f}, {max(fa_means):.3f}]  "
                f"mean={np.mean(fa_means):.3f}"
            )


def benchmark(
    experiment_config_path: str,
    seeds: list[int] = (1, 2, 3),
):
    """Run anomaly detection for multiple seeds with and without global weights.

    Args:
        experiment_config_path: Path to the base YAML config file.
        seeds: List of random seeds to evaluate.
    """
    config_path = Path(experiment_config_path)
    with config_path.open("r") as f:
        base_config = yaml.safe_load(f)

    original_name = base_config["experiment_name"]
    global_params_path = base_config.get("lstm_global_params")
    output_root = Path(base_config.get("output_root", "experiments/out"))
    benchmark_root = output_root / f"{original_name}_benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    # Define conditions: always run local; run global only if a path is configured
    conditions = []
    if global_params_path is not None:
        conditions.append(("global", global_params_path))
    conditions.append(("local", None))

    if len(conditions) == 1:
        print(
            "NOTE: lstm_global_params is null in config — "
            "only running the local (no global weights) condition."
        )

    # Run all (seed, condition) combinations
    runs = []
    for seed in seeds:
        for cond_name, gp in conditions:
            summary = _run_single(
                base_config, seed, cond_name, gp, benchmark_root
            )
            runs.append(
                {
                    "condition": cond_name,
                    "seed": seed,
                    "summary": summary,
                }
            )

    # Collect per-magnitude data and aggregate by (condition, magnitude)
    magnitude_results = _collect_magnitude_results(runs)
    aggregates = _aggregate_by_condition_magnitude(magnitude_results)
    condition_names = [c[0] for c in conditions]

    # Print results
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK RESULTS: {original_name}")
    print(f"Seeds: {list(seeds)}  |  Conditions: {condition_names}")
    print(f"{'=' * 80}\n")
    _print_aggregate_table(aggregates, condition_names)
    _print_condition_summary(aggregates, condition_names)

    # Per-seed validation metrics
    validation_metrics_per_seed = [
        {
            "condition": run["condition"],
            "seed": run["seed"],
            "validation_log_likelihood": run["summary"]
            .get("optimal_validation_metrics", {})
            .get("validation_log_likelihood"),
            "validation_rmse": run["summary"]
            .get("optimal_validation_metrics", {})
            .get("validation_rmse"),
        }
        for run in runs
    ]

    # Save benchmark summary
    benchmark_output = {
        "experiment_name": original_name,
        "config_path": str(config_path),
        "seeds": list(seeds),
        "conditions": condition_names,
        "magnitude_results": magnitude_results,
        "aggregate_by_magnitude": aggregates,
        "validation_metrics_per_seed": validation_metrics_per_seed,
        "benchmark_root": str(benchmark_root),
    }
    benchmark_path = benchmark_root / "summary.json"
    with benchmark_path.open("w") as f:
        json.dump(benchmark_output, f, indent=2, default=str)
    print(f"\nBenchmark summary saved to: {benchmark_path}")


if __name__ == "__main__":
    fire.Fire(benchmark)
