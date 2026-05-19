"""Benchmark anomaly detection under data scarcity by varying ``train_split``.

For each train ratio in ``train_ratios`` the script:
  1. Builds a modified config: ``train_split`` is overridden, ``slope_search_space``
     is collapsed to a single value (``slope``, default 0.075) so the per-run
     pipeline neither searches over slopes nor evaluates multiple magnitudes.
  2. Calls :func:`experiments.benchmark_anomaly_detection.benchmark` on that
     modified config, which itself parallelizes (seed x condition) runs.
  3. Reads the per-ratio ``summary.json`` and folds it into a combined summary.

Train ratios are processed sequentially; the inner benchmark handles seed and
condition parallelism.

Usage:
    python -m experiments.benchmark_anomaly_detection_train_ratio \
        --experiment_config_path experiments/config/OOD_timeseries/LGA002EFAPRG910.yaml \
        --seeds "[1,2,3]" --train_ratios "[1.0,0.8,0.6,0.4]"
"""

from __future__ import annotations

import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import yaml

try:
    from experiments.benchmark_anomaly_detection import benchmark as run_seed_benchmark
except ModuleNotFoundError:
    from benchmark_anomaly_detection import benchmark as run_seed_benchmark  # type: ignore


def _ratio_tag(ratio: float) -> str:
    return f"train{int(round(ratio * 100)):03d}"


def _aggregate_across_ratios(per_ratio: list[dict]) -> list[dict]:
    """Average per-(ratio, condition) aggregates across the magnitudes present.

    With ``slope_search_space=[slope]`` each per-ratio summary contains a single
    magnitude row per condition; this helper just flattens those into the
    combined view, keyed by (train_ratio, condition).
    """
    rows: list[dict] = []
    for entry in per_ratio:
        ratio = entry["train_ratio"]
        for r in entry["summary"].get("aggregate_by_magnitude", []):
            rows.append({"train_ratio": ratio, **r})
    return sorted(rows, key=lambda r: (r["train_ratio"], r["condition"]))


def _print_combined_table(rows: list[dict]) -> None:
    if not rows:
        print("(no successful runs)")
        return

    columns = [
        ("TrainRatio", "train_ratio", ".2f"),
        ("Condition", "condition", None),
        ("Magnitude", "anomaly_magnitude", ".3f"),
        ("P(detect)", "probability_of_detection", ".2f"),
        ("P(det) std", "probability_of_detection_std", ".3f"),
        ("FA/yr mean", "false_alarm_rate_per_y_mean", ".3f"),
        ("FA/yr std", "false_alarm_rate_per_y_std", ".3f"),
        ("TTD(yr) mean", "time_to_detection_years_mean", ".3f"),
        ("TTD(yr) std", "time_to_detection_years_std", ".3f"),
        ("N(seeds)", "num_seeds", None),
    ]

    def _fmt(value, fmt):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:{fmt}}" if fmt else f"{value}"
        return str(value)

    widths = []
    for header, key, fmt in columns:
        col = [_fmt(r.get(key), fmt) for r in rows]
        widths.append(max(len(header), max(len(c) for c in col)))

    header_line = "  ".join(h.ljust(w) for (h, _, _), w in zip(columns, widths))
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print(
            "  ".join(
                _fmt(r.get(key), fmt).ljust(w)
                for (_, key, fmt), w in zip(columns, widths)
            )
        )


def benchmark(
    experiment_config_path: str,
    seeds: list[int] = (1, 2, 3),
    train_ratios: list[float] = (1.0, 0.8, 0.6, 0.4),
    slope: float = 0.075,
    max_concurrent: int = 6,
):
    """Sweep ``train_split`` while fixing the anomaly slope.

    Args:
        experiment_config_path: Path to a base YAML config.
        seeds: Seeds passed to the inner (seed x condition) benchmark.
        train_ratios: ``train_split`` values to evaluate. Each is run as an
            independent inner benchmark.
        slope: Anomaly slope to fix (single value). Both the optimizer search
            space and the multi-realization evaluation magnitudes collapse to
            this value.
        max_concurrent: Forwarded to the inner benchmark; bounds the number of
            (seed x condition) processes running in parallel for each ratio.
    """
    config_path = Path(experiment_config_path)
    with config_path.open("r") as f:
        base_config = yaml.safe_load(f)

    original_name = base_config["experiment_name"]
    output_root = Path(base_config.get("output_root", "experiments/out"))
    sweep_root = output_root / f"{original_name}_train_scarcity"
    sweep_root.mkdir(parents=True, exist_ok=True)
    temp_config_dir = sweep_root / "temp_configs"
    temp_config_dir.mkdir(parents=True, exist_ok=True)

    ratios = [float(r) for r in train_ratios]
    print(
        f"Train-ratio sweep for '{original_name}': "
        f"ratios={ratios}  seeds={list(seeds)}  slope={slope}"
    )
    print(f"Outputs root: {sweep_root}")

    per_ratio: list[dict] = []
    total_start = time.perf_counter()

    for ratio in ratios:
        tag = _ratio_tag(ratio)
        ratio_config = copy.deepcopy(base_config)
        ratio_config["train_split"] = ratio
        ratio_config["slope_search_space"] = [float(slope)]
        ratio_config["slope"] = float(slope)
        ratio_config["experiment_name"] = f"{original_name}_{tag}"

        temp_config_path = temp_config_dir / f"{ratio_config['experiment_name']}.yaml"
        with temp_config_path.open("w") as f:
            yaml.safe_dump(ratio_config, f, sort_keys=False)

        print(f"\n{'#' * 70}")
        print(f"# train_ratio={ratio:.2f}  (config: {temp_config_path})")
        print(f"{'#' * 70}")

        ratio_start = time.perf_counter()
        try:
            run_seed_benchmark(
                experiment_config_path=str(temp_config_path),
                seeds=list(seeds),
                max_concurrent=int(max_concurrent),
            )
            status = "completed"
            error = None
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            print(f"!! train_ratio={ratio:.2f} FAILED: {error}")
        ratio_elapsed = time.perf_counter() - ratio_start

        inner_summary_path = (
            output_root / f"{ratio_config['experiment_name']}_benchmark" / "summary.json"
        )
        inner_summary: dict = {}
        if inner_summary_path.exists():
            with inner_summary_path.open("r") as f:
                inner_summary = json.load(f)

        per_ratio.append(
            {
                "train_ratio": ratio,
                "experiment_name": ratio_config["experiment_name"],
                "config_path": str(temp_config_path),
                "inner_benchmark_root": str(inner_summary_path.parent),
                "inner_summary_path": str(inner_summary_path),
                "status": status,
                "error": error,
                "elapsed_seconds": ratio_elapsed,
                "summary": inner_summary,
            }
        )

    total_elapsed = time.perf_counter() - total_start

    combined_rows = _aggregate_across_ratios(per_ratio)
    print(f"\n{'=' * 80}")
    print(f"TRAIN-RATIO SWEEP RESULTS: {original_name}  (slope={slope})")
    print(
        f"Total elapsed: {total_elapsed:.1f}s "
        f"({total_elapsed / 60:.1f} min) across {len(ratios)} ratios"
    )
    print(f"{'=' * 80}\n")
    _print_combined_table(combined_rows)

    combined = {
        "experiment_name": original_name,
        "config_path": str(config_path),
        "seeds": list(seeds),
        "train_ratios": ratios,
        "slope": float(slope),
        "per_ratio": [
            {
                k: v
                for k, v in entry.items()
                if k != "summary"
            }
            | {
                "magnitude_results": entry["summary"].get("magnitude_results", []),
                "aggregate_by_magnitude": entry["summary"].get(
                    "aggregate_by_magnitude", []
                ),
                "validation_metrics_per_seed": entry["summary"].get(
                    "validation_metrics_per_seed", []
                ),
                "run_status": entry["summary"].get("run_status", []),
            }
            for entry in per_ratio
        ],
        "aggregate_by_ratio_condition": combined_rows,
        "total_elapsed_seconds": total_elapsed,
        "max_concurrent": int(max_concurrent),
        "sweep_root": str(sweep_root),
    }
    out_path = sweep_root / "summary.json"
    with out_path.open("w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nCombined summary saved to: {out_path}")


if __name__ == "__main__":
    fire.Fire(benchmark)
