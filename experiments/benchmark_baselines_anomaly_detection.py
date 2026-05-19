"""Run baselines_anomaly_detection.py across every config in a directory.

For each ``*.yaml`` in ``configs_dir`` (default ``experiments/config/ID_timeseries``),
fit and evaluate the requested baselines. Seed-dependent methods (``lstm_ed``,
``tranad``) run once per seed in ``seeds``; the others run once. Each per-config
run writes its own ``<experiment_name>_baselines/summary.json`` (already in
benchmark-aggregable shape with ``magnitude_results`` and
``aggregate_by_magnitude``); this driver additionally writes a combined summary
spanning all configs at ``<output_root>/baselines_<dir_name>/summary.json``.

Usage:
    python -m experiments.benchmark_baselines_anomaly_detection
    python -m experiments.benchmark_baselines_anomaly_detection \
        --configs_dir experiments/config/ID_timeseries \
        --methods "[prophet,lstm_ed,damp,tranad]" --seeds "[1,2,3]"
"""

from __future__ import annotations

import contextlib
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import yaml

try:
    from experiments.baselines_anomaly_detection import (
        SEED_DEPENDENT_METHODS,
        main as run_baselines,
    )
except ModuleNotFoundError:
    from baselines_anomaly_detection import (  # type: ignore
        SEED_DEPENDENT_METHODS,
        main as run_baselines,
    )


def _aggregate_across_configs(per_config: list[dict]) -> list[dict]:
    """Mean / std of per-config aggregate_by_magnitude rows, grouped by (method, magnitude)."""
    groups = defaultdict(list)
    for entry in per_config:
        for row in entry.get("aggregate_by_magnitude", []):
            groups[(row["condition"], row["anomaly_magnitude"])].append(
                {"experiment_name": entry["experiment_name"], **row}
            )

    def _finite(values):
        return [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]

    out = []
    for (condition, magnitude), items in sorted(groups.items()):
        p_dets = _finite([r["probability_of_detection"] for r in items])
        fa_means = _finite([r["false_alarm_rate_per_y_mean"] for r in items])
        ttd_means = _finite([r["time_to_detection_years_mean"] for r in items])
        out.append(
            {
                "condition": condition,
                "anomaly_magnitude": magnitude,
                "probability_of_detection_mean": float(np.mean(p_dets)) if p_dets else None,
                "probability_of_detection_std": float(np.std(p_dets)) if p_dets else None,
                "false_alarm_rate_per_y_mean": float(np.mean(fa_means)) if fa_means else None,
                "false_alarm_rate_per_y_std": float(np.std(fa_means)) if fa_means else None,
                "time_to_detection_years_mean": float(np.mean(ttd_means)) if ttd_means else None,
                "time_to_detection_years_std": float(np.std(ttd_means)) if ttd_means else None,
                "num_configs": len(items),
            }
        )
    return out


def benchmark(
    configs_dir: str = "experiments/config/ID_timeseries",
    methods: list[str] = ("prophet", "lstm_ed", "damp", "tranad"),
    seeds: list[int] = (1, 2, 3),
    n_jobs: int = 45,
    output_root: str = "experiments/out",
    pattern: str = "*.yaml",
):
    """Run baselines_anomaly_detection on every config in ``configs_dir``.

    Args:
        configs_dir: Directory of YAML experiment configs.
        methods: Baseline methods to run for each config.
        seeds: Seeds to use for seed-dependent methods. Deterministic methods
            run once regardless.
        n_jobs: Realization-level parallelism passed through to each per-config
            run. Configs themselves are processed sequentially.
        output_root: Combined summary lands at
            ``<output_root>/baselines_<dir_name>/summary.json``.
        pattern: Glob to select configs (defaults to ``*.yaml``).
    """
    configs_root = Path(configs_dir)
    config_paths = sorted(p for p in configs_root.glob(pattern) if p.is_file())
    if not config_paths:
        raise FileNotFoundError(f"No configs matching '{pattern}' under {configs_root}")

    requested = [str(m) for m in methods]
    requested_seeds = [int(s) for s in seeds]
    seed_dep = sorted(m for m in requested if m in SEED_DEPENDENT_METHODS)

    out_root = Path(output_root) / f"baselines_{configs_root.name}"
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configs ({len(config_paths)}): {[p.name for p in config_paths]}")
    print(f"Methods: {requested}  |  seed-dep: {seed_dep}  |  seeds: {requested_seeds}")
    print(f"Per-config logs: {log_dir}")

    per_config: list[dict] = []
    run_status: list[dict] = []
    total_start = time.perf_counter()

    for cfg_path in config_paths:
        log_path = log_dir / f"{cfg_path.stem}.log"
        print(f"\n>>> {cfg_path.name}  (log: {log_path})")
        start = time.perf_counter()
        try:
            with log_path.open("w") as log_file:
                with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                    summary = run_baselines(
                        experiment_config_path=str(cfg_path),
                        methods=requested,
                        seeds=requested_seeds,
                        n_jobs=int(n_jobs),
                    )
            elapsed = time.perf_counter() - start
            per_config.append(summary)
            status = (
                "completed_with_failures"
                if summary.get("failed_runs")
                else "completed"
            )
            run_status.append(
                {
                    "config_path": str(cfg_path),
                    "experiment_name": summary["experiment_name"],
                    "status": status,
                    "elapsed_seconds": elapsed,
                    "failed_runs": summary.get("failed_runs", []),
                }
            )
            if status == "completed_with_failures":
                print(
                    f"    completed with {len(summary.get('failed_runs', []))} "
                    f"failed run(s) in {elapsed:.1f}s -> {summary['baselines_root']}"
                )
            else:
                print(f"    done in {elapsed:.1f}s -> {summary['baselines_root']}")
        except Exception as exc:
            elapsed = time.perf_counter() - start
            run_status.append(
                {
                    "config_path": str(cfg_path),
                    "experiment_name": None,
                    "status": "failed",
                    "elapsed_seconds": elapsed,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"    FAILED ({type(exc).__name__}: {exc}) — see {log_path}")
            with log_path.open("a") as f:
                f.write("\n\n--- traceback ---\n")
                f.write(traceback.format_exc())

    total_elapsed = time.perf_counter() - total_start

    cross_config_aggregate = _aggregate_across_configs(per_config)
    combined = {
        "configs_dir": str(configs_root),
        "config_paths": [str(p) for p in config_paths],
        "methods": requested,
        "seeds": requested_seeds,
        "seed_dependent_methods": seed_dep,
        "per_config": [
            {
                "experiment_name": entry["experiment_name"],
                "config_path": entry["config_path"],
                "baselines_root": entry["baselines_root"],
                "status": entry.get("status", "completed"),
                "failed_runs": entry.get("failed_runs", []),
                "magnitude_results": entry["magnitude_results"],
                "aggregate_by_magnitude": entry["aggregate_by_magnitude"],
            }
            for entry in per_config
        ],
        "aggregate_across_configs": cross_config_aggregate,
        "run_status": run_status,
        "total_elapsed_seconds": total_elapsed,
    }
    out_path = out_root / "summary.json"
    with out_path.open("w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nCombined summary: {out_path}")
    completed = sum(1 for r in run_status if r["status"] == "completed")
    partial = sum(1 for r in run_status if r["status"] == "completed_with_failures")
    print(
        f"Completed {completed}/{len(run_status)} configs"
        f" (+{partial} partial) in {total_elapsed:.1f}s"
        f" ({total_elapsed / 60:.1f} min)"
    )


if __name__ == "__main__":
    fire.Fire(benchmark)
