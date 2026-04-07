"""Benchmark anomaly_detection_lstm across multiple seeds and global/local weights.

Usage:
    python -m experiments.benchmark_anomaly_detection --experiment_config_path experiments/config/LTU012ESAP-E020.yaml
    python -m experiments.benchmark_anomaly_detection --experiment_config_path experiments/config/LTU012ESAP-E020.yaml --seeds "[1,2,3,4,5]"
"""

import copy
import json
import sys
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


def _run_single(base_config: dict, seed: int, condition: str, global_params):
    """Create a modified config and run anomaly detection."""
    config = copy.deepcopy(base_config)
    config["lstm_manual_seed"] = seed
    config["lstm_global_params"] = global_params
    config["experiment_name"] = (
        f"{base_config['experiment_name']}_benchmark_{condition}_seed{seed}"
    )

    output_root = Path(config.get("output_root", "experiments/out"))
    output_dir = output_root / config["experiment_name"]

    # Write temporary config for this run
    temp_config_path = output_dir / "experiment_config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    with temp_config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"\n{'=' * 60}")
    print(f"Running: condition={condition}, seed={seed}")
    print(f"{'=' * 60}")

    run_anomaly_detection(experiment_config_path=str(temp_config_path))

    summary = _read_summary(output_dir)
    detection = summary["final_skf_detection"]

    return {
        "condition": condition,
        "seed": seed,
        "false_alarm_rate": detection["false_alarm_rate_before_anomaly"],
        "false_alarm_count": detection["false_alarm_count_before_anomaly"],
        "num_pre_anomaly": detection["num_points_before_anomaly"],
        "detected": detection["detected"],
        "delay_steps": detection.get("delay_steps"),
        "delay": detection.get("delay"),
        "threshold": detection["threshold"],
        "val_ll": summary["optimal_validation_metrics"].get(
            "validation_log_likelihood"
        ),
        "val_rmse": summary["optimal_validation_metrics"].get("validation_rmse"),
    }


def _format_value(value, fmt=".4f"):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:{fmt}}"
    return str(value)


def _print_results_table(results: list[dict]):
    """Print a formatted table of individual run results."""
    columns = [
        ("Condition", "condition", None),
        ("Seed", "seed", None),
        ("FAR", "false_alarm_rate", ".6f"),
        ("FA Count", "false_alarm_count", None),
        ("Detected", "detected", None),
        ("Delay Steps", "delay_steps", None),
        ("Delay", "delay", None),
        ("Val LL", "val_ll", ".4f"),
        ("Val RMSE", "val_rmse", ".6f"),
    ]

    # Compute column widths
    widths = []
    for header, key, fmt in columns:
        col_values = [_format_value(r[key], fmt) if fmt else str(r[key]) for r in results]
        widths.append(max(len(header), max(len(v) for v in col_values)))

    # Print header
    header_line = "  ".join(h.ljust(w) for (h, _, _), w in zip(columns, widths))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for r in results:
        row_values = []
        for _, key, fmt in columns:
            val = r[key]
            row_values.append(_format_value(val, fmt) if fmt else str(val))
        print("  ".join(v.ljust(w) for v, w in zip(row_values, widths)))


def _print_aggregate(results: list[dict], condition_names: list[str]):
    """Print aggregate statistics grouped by condition."""
    print(f"\n{'=' * 80}")
    print("AGGREGATE STATISTICS")
    print(f"{'=' * 80}")

    for cond in condition_names:
        cond_results = [r for r in results if r["condition"] == cond]
        n_runs = len(cond_results)

        fars = [r["false_alarm_rate"] for r in cond_results]
        detected_flags = [r["detected"] for r in cond_results]
        delay_steps = [
            r["delay_steps"] for r in cond_results if r["delay_steps"] is not None
        ]
        val_lls = [r["val_ll"] for r in cond_results if r["val_ll"] is not None]
        val_rmses = [r["val_rmse"] for r in cond_results if r["val_rmse"] is not None]

        print(f"\n--- {cond.upper()} ({n_runs} runs) ---")
        print(
            f"  False alarm rate:   mean={np.mean(fars):.6f}  "
            f"std={np.std(fars):.6f}  "
            f"min={np.min(fars):.6f}  max={np.max(fars):.6f}"
        )
        print(
            f"  Detection rate:     {sum(detected_flags)}/{n_runs} "
            f"({sum(detected_flags)/n_runs*100:.0f}%)"
        )
        if delay_steps:
            print(
                f"  Delay (steps):      mean={np.mean(delay_steps):.1f}  "
                f"std={np.std(delay_steps):.1f}  "
                f"min={np.min(delay_steps)}  max={np.max(delay_steps)}"
            )
        else:
            print("  Delay (steps):      no detections")
        if val_lls:
            print(
                f"  Validation LL:      mean={np.mean(val_lls):.4f}  "
                f"std={np.std(val_lls):.4f}"
            )
        if val_rmses:
            print(
                f"  Validation RMSE:    mean={np.mean(val_rmses):.6f}  "
                f"std={np.std(val_rmses):.6f}"
            )


def benchmark(
    experiment_config_path: str,
    seeds: list[int] = (1, 2, 3,4,5),
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

    results = []
    for seed in seeds:
        for cond_name, gp in conditions:
            result = _run_single(base_config, seed, cond_name, gp)
            results.append(result)

    # Print individual results
    condition_names = [c[0] for c in conditions]
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK RESULTS: {original_name}")
    print(f"Seeds: {list(seeds)}  |  Conditions: {condition_names}")
    print(f"{'=' * 80}\n")
    _print_results_table(results)

    # Print aggregates
    _print_aggregate(results, condition_names)

    # Save full results to JSON
    output_root = Path(base_config.get("output_root", "experiments/out"))
    benchmark_output = {
        "experiment_name": original_name,
        "config_path": str(config_path),
        "seeds": list(seeds),
        "conditions": condition_names,
        "runs": results,
    }
    benchmark_path = output_root / f"{original_name}_benchmark_summary.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_path.open("w") as f:
        json.dump(benchmark_output, f, indent=2, default=str)
    print(f"\nBenchmark summary saved to: {benchmark_path}")


if __name__ == "__main__":
    fire.Fire(benchmark)
