"""Benchmark filter_global_lstm across multiple seeds and global/local weights,
with Chronos-2 one-step-ahead filtering as an additional baseline.

Usage:
    python -m experiments.benchmark_filter_global_lstm \
        --experiment_config_path experiments/config/OOD_timeseries/test_10.yaml
    python -m experiments.benchmark_filter_global_lstm \
        --experiment_config_path experiments/config/OOD_timeseries/test_10.yaml \
        --seeds "[1,2,3,4,5]"
    python -m experiments.benchmark_filter_global_lstm \
        --experiment_config_path experiments/config/OOD_timeseries/test_10.yaml \
        --skip_chronos
"""

import copy
import json
from pathlib import Path

import fire
import numpy as np
import yaml

try:
    from experiments.filter_global_lstm import main as run_filter
    from experiments.utils import prepare_dataset
except ModuleNotFoundError:
    from filter_global_lstm import main as run_filter
    from utils import prepare_dataset


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
):
    """Create a modified config and run filter_global_lstm."""
    config = copy.deepcopy(base_config)
    config["lstm_manual_seed"] = seed
    config["lstm_global_params"] = global_params
    config["experiment_name"] = (
        f"{base_config['experiment_name']}_benchmark_{condition}_seed{seed}"
    )
    config["output_root"] = str(benchmark_root / "runs")

    output_root = Path(config["output_root"])
    temp_config_dir = benchmark_root / "temp_configs"
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_config_dir / f"{config['experiment_name']}.yaml"
    with temp_config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"\n{'=' * 60}")
    print(f"Running: condition={condition}, seed={seed}")
    print(f"{'=' * 60}")

    run_filter(experiment_config_path=str(temp_config_path))

    # filter_global_lstm appends "_filter" to the experiment name internally
    actual_output_dir = output_root / f"{config['experiment_name']}_filter"
    summary = _read_summary(actual_output_dir)
    test_metrics = summary["test_metrics"]

    return {
        "condition": condition,
        "seed": seed,
        "test_ll": test_metrics["log_likelihood"],
        "test_rmse": test_metrics["rmse"],
        "val_ll": summary["validation_metrics_best"].get(
            "validation_log_likelihood"
        ),
        "val_rmse": summary["validation_metrics_best"].get("validation_rmse"),
    }


def _run_chronos(
    base_config: dict,
    chronos_model: str,
    benchmark_root: Path,
    seed: int = 42,
    chronos_device: str = "gpu",
):
    """Run Chronos-2 one-step-ahead filtering and return test-set metrics."""

    import pandas as pd
    from pytagi import metric as tagi_metric
    from tqdm.auto import tqdm

    try:
        import torch
        from chronos import Chronos2Pipeline
    except ImportError as e:
        raise ImportError(
            f"Failed to import torch/chronos: {e}\n"
            "This is often caused by a PyTorch/NCCL version mismatch. Try:\n"
            "  pip install torch --force-reinstall\n"
            "Or run with --skip_chronos to skip the Chronos baseline."
        ) from e

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print("Running: condition=chronos")
    print(f"{'=' * 60}")

    # Use prepare_dataset for consistent data splits (anomaly_slope=0 for filtering)
    config = copy.deepcopy(base_config)
    config["anomaly_slope"] = 0.0
    config.setdefault("anomaly_start_time", config["validation_start"])

    dataset = prepare_dataset(
        train_split=float(config["train_split"]),
        anomaly_slope=0.0,
        experiment_config=config,
    )

    data_processor = dataset["data_processor"]
    _, _, test_idx = data_processor.get_split_indices()

    # Get raw (unstandardized) full series and timestamps
    all_obs = data_processor.get_data("all").flatten()
    all_times = data_processor.get_time("all")
    n = len(all_obs)

    # Load Chronos pipeline
    requested_device = chronos_device.lower()
    if requested_device not in {"cpu", "cuda", "auto"}:
        raise ValueError(
            f"Invalid chronos_device={chronos_device!r}. Use 'cpu', 'cuda', or 'auto'."
        )
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    pipeline = Chronos2Pipeline.from_pretrained(chronos_model, device_map=device)
    print(f"Loaded {chronos_model} on {device}")

    # One-step-ahead filtering
    min_context = int(config.get("lstm_look_back_len", 52))
    pred_means = np.full(n, np.nan, dtype=np.float32)
    pred_q10 = np.full(n, np.nan, dtype=np.float32)
    pred_q90 = np.full(n, np.nan, dtype=np.float32)

    Z90 = 1.2815515655446004  # z-score for 90th percentile

    for t in tqdm(range(min_context, n), desc="Chronos 1-step-ahead filtering"):
        context = all_obs[:t].astype(np.float32)
        ctx_dates = all_times[:t]

        context_df = pd.DataFrame(
            {
                "id": "0",
                "timestamp": ctx_dates,
                "target": context,
            }
        )

        with torch.no_grad():
            forecast_df = pipeline.predict_df(
                context_df,
                prediction_length=1,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )

        pred_means[t] = forecast_df["0.5"].iloc[0]
        pred_q10[t] = forecast_df["0.1"].iloc[0]
        pred_q90[t] = forecast_df["0.9"].iloc[0]

    pred_std = np.where(
        ~np.isnan(pred_means),
        np.maximum((pred_q90 - pred_q10) / (2 * Z90), 1e-6),
        np.nan,
    ).astype(np.float32)

    # Compute test-set metrics
    test_obs = all_obs[test_idx]
    test_pred = pred_means[test_idx]
    test_std = pred_std[test_idx]

    # Filter out NaN predictions (should be none if min_context < test_start)
    valid = ~np.isnan(test_pred)
    test_obs_valid = test_obs[valid]
    test_pred_valid = test_pred[valid]
    test_std_valid = test_std[valid]

    test_ll = float(
        tagi_metric.log_likelihood(
            prediction=test_pred_valid,
            observation=test_obs_valid,
            std=test_std_valid,
        )
    )
    test_rmse = float(
        np.sqrt(np.nanmean((test_pred_valid - test_obs_valid) ** 2))
    )

    print(f"Chronos test-set log-likelihood: {test_ll:.6f}")
    print(f"Chronos test-set RMSE: {test_rmse:.6f}")

    # Plot test-set predictions
    import matplotlib.pyplot as plt

    DOUBLE_COL = (6.5, 3.5)
    test_times = all_times[test_idx]

    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    ax.plot(test_times, test_obs, color="tab:red", label="Observations")
    ax.plot(test_times, test_pred, color="tab:blue", label=r"Chronos $\mu$")
    ax.fill_between(
        test_times,
        test_pred - test_std,
        test_pred + test_std,
        color="tab:blue",
        alpha=0.3,
        label=r"Chronos $\pm\sigma$",
    )
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    plt.tight_layout()

    output_dir = benchmark_root / "chronos"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "chronos_test_predictions.pdf", format="pdf")
    fig.savefig(output_dir / "chronos_test_predictions.pgf", format="pgf")
    plt.close(fig)

    summary = {
        "experiment_name": f"{base_config['experiment_name']}_benchmark_chronos",
        "model": chronos_model,
        "device": device,
        "test_metrics": {
            "log_likelihood": test_ll,
            "rmse": test_rmse,
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return {
        "condition": "chronos",
        "seed": seed,
        "test_ll": test_ll,
        "test_rmse": test_rmse,
        "val_ll": None,
        "val_rmse": None,
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
        ("Test LL", "test_ll", ".4f"),
        ("Test RMSE", "test_rmse", ".6f"),
        ("Val LL", "val_ll", ".4f"),
        ("Val RMSE", "val_rmse", ".6f"),
    ]

    widths = []
    for header, key, fmt in columns:
        col_values = [
            _format_value(r[key], fmt) if fmt else str(r[key]) for r in results
        ]
        widths.append(max(len(header), max(len(v) for v in col_values)))

    header_line = "  ".join(h.ljust(w) for (h, _, _), w in zip(columns, widths))
    print(header_line)
    print("-" * len(header_line))

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

        test_lls = [r["test_ll"] for r in cond_results if r["test_ll"] is not None]
        test_rmses = [
            r["test_rmse"] for r in cond_results if r["test_rmse"] is not None
        ]
        val_lls = [r["val_ll"] for r in cond_results if r["val_ll"] is not None]
        val_rmses = [
            r["val_rmse"] for r in cond_results if r["val_rmse"] is not None
        ]

        print(f"\n--- {cond.upper()} ({n_runs} runs) ---")
        if test_lls:
            print(
                f"  Test LL:            mean={np.mean(test_lls):.4f}  "
                f"std={np.std(test_lls):.4f}  "
                f"min={np.min(test_lls):.4f}  max={np.max(test_lls):.4f}"
            )
        if test_rmses:
            print(
                f"  Test RMSE:          mean={np.mean(test_rmses):.6f}  "
                f"std={np.std(test_rmses):.6f}  "
                f"min={np.min(test_rmses):.6f}  max={np.max(test_rmses):.6f}"
            )
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
    seeds: list[int] = (1,2,3),
    chronos_model: str = "amazon/chronos-2",
    chronos_device: str = "auto",
    skip_chronos: bool = False,
):
    """Run filter_global_lstm for multiple seeds with global/local weights,
    and optionally Chronos-2 as a baseline.

    Args:
        experiment_config_path: Path to the base YAML config file.
        seeds: List of random seeds to evaluate.
        chronos_model: Chronos model identifier.
        chronos_device: Device for Chronos inference: 'cpu', 'cuda', or 'auto'.
        skip_chronos: If True, skip the Chronos-2 baseline.
    """
    config_path = Path(experiment_config_path)
    with config_path.open("r") as f:
        base_config = yaml.safe_load(f)

    original_name = base_config["experiment_name"]
    global_params_path = base_config.get("lstm_global_params")
    output_root = Path(base_config.get("output_root", "experiments/out"))
    benchmark_root = output_root / f"{original_name}_filter_benchmark"
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

    results = []
    global_result_cache = {}
    for seed in seeds:
        for cond_name, gp in conditions:
            # Global condition is deterministic (pretrained weights override the
            # random seed), so only run it once and reuse the result.
            if cond_name == "global" and gp is not None:
                if "global" not in global_result_cache:
                    result = _run_single(
                        base_config, seed, cond_name, gp, benchmark_root
                    )
                    global_result_cache["global"] = result
                else:
                    print(
                        f"\n{'=' * 60}\n"
                        f"Skipping: condition=global, seed={seed} "
                        f"(deterministic — reusing seed={global_result_cache['global']['seed']} result)"
                        f"\n{'=' * 60}"
                    )
                    result = {**global_result_cache["global"], "seed": seed}
                results.append(result)
            else:
                result = _run_single(
                    base_config, seed, cond_name, gp, benchmark_root
                )
                results.append(result)

    # Run Chronos-2 baseline (deterministic, single run)
    condition_names = [c[0] for c in conditions]
    if not skip_chronos:
        chronos_result = _run_chronos(
            base_config,
            chronos_model,
            benchmark_root,
            chronos_device=chronos_device,
        )
        results.append(chronos_result)
        condition_names.append("chronos")

    # Print individual results
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK RESULTS: {original_name}")
    print(f"Seeds: {list(seeds)}  |  Conditions: {condition_names}")
    print(f"{'=' * 80}\n")
    _print_results_table(results)

    # Print aggregates
    _print_aggregate(results, condition_names)

    # Save full results to JSON
    benchmark_output = {
        "experiment_name": original_name,
        "config_path": str(config_path),
        "seeds": list(seeds),
        "conditions": condition_names,
        "runs": results,
        "benchmark_root": str(benchmark_root),
    }
    benchmark_path = benchmark_root / "summary.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_path.open("w") as f:
        json.dump(benchmark_output, f, indent=2, default=str)
    print(f"\nBenchmark summary saved to: {benchmark_path}")


if __name__ == "__main__":
    fire.Fire(benchmark)
