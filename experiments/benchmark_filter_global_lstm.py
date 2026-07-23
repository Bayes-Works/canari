"""Benchmark filter_global_lstm across selectable model variants.

Usage:
    python -m experiments.benchmark_filter_global_lstm \
        --experiment_config_path experiments/config/benchmark_data/test_10.yaml
    python -m experiments.benchmark_filter_global_lstm \
        --experiment_config_path experiments/config/benchmark_data/test_10.yaml \
        --seeds "[1,2,3,4,5]"
    python -m experiments.benchmark_filter_global_lstm \
        --experiment_config_path experiments/config/benchmark_data/test_10.yaml \
        --models '["local","global_finetune"]'
    python -m experiments.benchmark_filter_global_lstm \
        --experiment_config_path experiments/config/benchmark_data/test_10.yaml \
        --skip_models '["chronos2"]'
    python -m experiments.benchmark_filter_global_lstm \
        --experiment_config_path experiments/config/benchmark_data/test_10.yaml \
        --train_percentages "[25,50,100]"
    python -m experiments.benchmark_filter_global_lstm \
        --experiment_config_path experiments/config/OOD_timeseries/test_2.yaml \
        --models '["global_finetune","global_zeroshot"]' \
        --global_params_paths saved_params/seed_variability
"""

import copy
import json
import re
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


MODEL_ALIASES = {
    "local": "local",
    "global-finetune": "global_finetune",
    "global_finetune": "global_finetune",
    "global-fine-tune": "global_finetune",
    "global_fine_tune": "global_finetune",
    "global": "global_finetune",
    "global-zeroshot": "global_zeroshot",
    "global_zeroshot": "global_zeroshot",
    "global-zero-shot": "global_zeroshot",
    "global_zero_shot": "global_zeroshot",
    "zeroshot": "global_zeroshot",
    "zero-shot": "global_zeroshot",
    "zero_shot": "global_zeroshot",
    "chronos": "chronos2",
    "chronos2": "chronos2",
    "chronos-2": "chronos2",
    "chronos_2": "chronos2",
}

DEFAULT_MODELS = (
    "local",
    "global_finetune",
    "global_zeroshot",
    "chronos2",
)


def _coerce_model_names(value, default: tuple[str, ...] = ()) -> list[str]:
    """Normalize Fire-friendly model selections to canonical model names."""
    if value is None:
        raw_names = list(default)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "null"}:
            raw_names = []
        elif stripped.lower() == "all":
            raw_names = list(DEFAULT_MODELS)
        elif stripped.startswith("["):
            raw_names = json.loads(stripped)
        else:
            raw_names = [item.strip() for item in stripped.split(",")]
    else:
        raw_names = list(value)

    normalized = []
    for name in raw_names:
        key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        if not key:
            continue
        try:
            canonical = MODEL_ALIASES[key]
        except KeyError as exc:
            valid = ", ".join(DEFAULT_MODELS)
            raise ValueError(
                f"Unknown model {name!r}. Valid models are: {valid}."
            ) from exc
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def _coerce_train_percentages(value, default: float) -> list[float]:
    """Normalize train percentages to fractions in (0, 1]."""
    if value is None:
        raw_values = [default]
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            raw_values = json.loads(stripped)
        else:
            raw_values = [item.strip() for item in stripped.split(",")]
    elif isinstance(value, (int, float)):
        raw_values = [value]
    else:
        raw_values = list(value)

    train_splits = []
    for raw_value in raw_values:
        train_split = float(raw_value)
        if train_split > 1.0:
            train_split = train_split / 100.0
        if not 0.0 < train_split <= 1.0:
            raise ValueError(
                "Training percentages must be in (0, 100] or fractions in (0, 1]. "
                f"Got {raw_value!r}."
            )
        if train_split not in train_splits:
            train_splits.append(train_split)
    return train_splits


def _coerce_seeds(value) -> list[int]:
    """Normalize Fire-friendly seed selections."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            raw_values = json.loads(stripped)
        else:
            raw_values = [item.strip() for item in stripped.split(",")]
    elif isinstance(value, (int, np.integer)):
        raw_values = [value]
    else:
        raw_values = list(value)
    return [int(seed) for seed in raw_values]


def _coerce_path_list(value) -> list[Path]:
    """Normalize Fire-friendly path selections to sorted checkpoint paths."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "null"}:
            return []
        if stripped.startswith("["):
            raw_paths = json.loads(stripped)
        else:
            raw_paths = [item.strip() for item in stripped.split(",")]
    else:
        raw_paths = list(value)

    paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(str(raw_path)).expanduser()
        if path.is_dir():
            paths.extend(sorted(p for p in path.iterdir() if p.is_file()))
        else:
            glob_matches = sorted(path.parent.glob(path.name))
            paths.extend(p for p in glob_matches if p.is_file())

    deduped_paths = []
    seen = set()
    for path in paths:
        path_key = str(path)
        if path_key not in seen:
            deduped_paths.append(path)
            seen.add(path_key)
    if not deduped_paths:
        raise ValueError(f"No global parameter files matched {value!r}.")
    return deduped_paths


def _global_params_label(path: Path) -> str:
    seed_match = re.search(r"seed[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if seed_match:
        return f"global_seed{seed_match.group(1)}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")


def _run_experiment_name(
    base_config: dict,
    condition: str,
    global_params_label: str | None,
    train_split: float,
    seed: int,
) -> str:
    condition_label = condition
    if global_params_label:
        condition_label = f"{condition}_{global_params_label}"
    return (
        f"{base_config['experiment_name']}_benchmark_"
        f"{condition_label}_{_train_split_label(train_split)}_seed{seed}"
    )


def _run_output_dir(
    base_config: dict,
    condition: str,
    global_params_label: str | None,
    train_split: float,
    seed: int,
    benchmark_root: Path,
) -> Path:
    experiment_name = _run_experiment_name(
        base_config,
        condition,
        global_params_label,
        train_split,
        seed,
    )
    return benchmark_root / "runs" / f"{experiment_name}_filter"


def _result_from_summary(
    output_dir: Path,
    condition: str,
    global_params_label: str | None,
    global_params_path: str | None,
    train_split: float,
    seed: int,
) -> dict:
    summary = _read_summary(output_dir)
    test_metrics = summary["test_metrics"]
    validation_metrics = summary.get("validation_metrics_best") or {}
    return {
        "condition": condition,
        "global_params_label": global_params_label,
        "global_params_path": global_params_path,
        "train_split": train_split,
        "train_percentage": train_split * 100.0,
        "seed": seed,
        "test_ll": test_metrics["log_likelihood"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics.get("mae"),
        "test_p50": test_metrics.get("p50"),
        "test_p90": test_metrics.get("p90"),
        "val_ll": validation_metrics.get("validation_log_likelihood"),
        "val_rmse": validation_metrics.get("validation_rmse"),
    }


def _train_split_label(train_split: float) -> str:
    percentage = train_split * 100.0
    if percentage.is_integer():
        return f"train{int(percentage):03d}pct"
    return f"train{percentage:.2f}pct".replace(".", "p")


def _build_filter_conditions(
    base_config: dict,
    models: list[str],
    global_params_paths: list[Path] | None = None,
) -> list[dict]:
    global_params_entries = []
    if global_params_paths:
        global_params_entries = [
            {
                "path": str(path),
                "label": _global_params_label(path),
            }
            for path in global_params_paths
        ]
    elif base_config.get("lstm_global_params") is not None:
        global_params_entries = [
            {
                "path": base_config["lstm_global_params"],
                "label": None,
            }
        ]
    conditions = []

    for model_name in models:
        if model_name == "chronos2":
            continue

        if model_name == "local":
            conditions.append(
                {
                    "name": "local",
                    "overrides": {
                        "lstm_global_params": None,
                        "lstm_finetune": False,
                        "lstm_zeroshot": False,
                        "lstm_increase_output_variance": False,
                        "lstm_num_layer": 1,
                        "num_hidden_unit": 64,
                    },
                    "cache_across_seeds": False,
                }
            )
            continue

        if not global_params_entries:
            print(
                f"NOTE: lstm_global_params is null in config; "
                f"skipping {model_name}."
            )
            continue

        if model_name == "global_finetune":
            for global_params_entry in global_params_entries:
                conditions.append(
                    {
                        "name": "global_finetune",
                        "global_params_label": global_params_entry["label"],
                        "global_params_path": global_params_entry["path"],
                        "overrides": {
                            "lstm_global_params": global_params_entry["path"],
                            "lstm_finetune": False,
                            "lstm_zeroshot": False,
                            "lstm_increase_output_variance": True,
                        },
                        "cache_across_seeds": True,
                    }
                )
        elif model_name == "global_zeroshot":
            for global_params_entry in global_params_entries:
                conditions.append(
                    {
                        "name": "global_zeroshot",
                        "global_params_label": global_params_entry["label"],
                        "global_params_path": global_params_entry["path"],
                        "overrides": {
                            "lstm_global_params": global_params_entry["path"],
                            "lstm_finetune": False,
                            "lstm_zeroshot": True,
                            "lstm_increase_output_variance": False,
                        },
                        "cache_across_seeds": True,
                    }
                )

    return conditions


def _run_single(
    base_config: dict,
    seed: int,
    condition: str,
    global_params_label: str | None,
    global_params_path: str | None,
    config_overrides: dict,
    train_split: float,
    benchmark_root: Path,
):
    """Create a modified config and run filter_global_lstm."""
    config = copy.deepcopy(base_config)
    config["lstm_manual_seed"] = seed
    config["train_split"] = train_split
    config.update(config_overrides)
    config["experiment_name"] = _run_experiment_name(
        base_config,
        condition,
        global_params_label,
        train_split,
        seed,
    )
    config["output_root"] = str(benchmark_root / "runs")

    output_root = Path(config["output_root"])
    temp_config_dir = benchmark_root / "temp_configs"
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_config_dir / f"{config['experiment_name']}.yaml"
    with temp_config_path.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"\n{'=' * 60}")
    print(
        f"Running: condition={condition}, "
        f"global_params={global_params_label or 'default'}, "
        f"train_percentage={train_split * 100:.2f}, seed={seed}"
    )
    print(f"{'=' * 60}")

    run_filter(experiment_config_path=str(temp_config_path))

    # filter_global_lstm appends "_filter" to the experiment name internally
    actual_output_dir = output_root / f"{config['experiment_name']}_filter"
    return _result_from_summary(
        actual_output_dir,
        condition,
        global_params_label,
        global_params_path,
        train_split,
        seed,
    )


def _run_chronos(
    base_config: dict,
    chronos_model: str,
    benchmark_root: Path,
    train_split: float,
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
    print(f"Running: condition=chronos2, " f"train_percentage={train_split * 100:.2f}")
    print(f"{'=' * 60}")

    # Use prepare_dataset for consistent data splits (anomaly_slope=0 for filtering)
    config = copy.deepcopy(base_config)
    config["anomaly_slope"] = 0.0
    config["train_split"] = train_split
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
    test_q90 = pred_q90[test_idx]

    # Filter out NaN predictions (should be none if min_context < test_start)
    valid = ~np.isnan(test_pred)
    test_obs_valid = test_obs[valid]
    test_pred_valid = test_pred[valid]
    test_std_valid = test_std[valid]
    test_q90_valid = test_q90[valid]

    # LL / RMSE / MAE in standardized space (scale constants from data_processor)
    mean_col = float(
        np.asarray(
            data_processor.scale_const_mean[data_processor.output_col]
        ).flatten()[0]
    )
    std_col = float(
        np.asarray(data_processor.scale_const_std[data_processor.output_col]).flatten()[
            0
        ]
    )
    test_obs_std_arr = (test_obs_valid - mean_col) / std_col
    test_pred_std_arr = (test_pred_valid - mean_col) / std_col
    test_std_std_arr = test_std_valid / std_col

    test_ll = float(
        tagi_metric.log_likelihood(
            prediction=test_pred_std_arr,
            observation=test_obs_std_arr,
            std=test_std_std_arr,
        )
    )
    res_std = test_pred_std_arr - test_obs_std_arr
    test_rmse = float(np.sqrt(np.nanmean(res_std**2)))
    test_mae = float(np.nanmean(np.abs(res_std)))

    # Np50 / Np90 = gluonts normalized quantile loss on original-space predictions
    denom = float(np.nansum(np.abs(test_obs_valid))) + 1e-8

    def _quantile_loss(target, forecast, q):
        return 2.0 * float(
            np.nansum(
                np.abs((forecast - target) * ((target <= forecast).astype(float) - q))
            )
        )

    test_p50 = _quantile_loss(test_obs_valid, test_pred_valid, 0.5) / denom
    test_p90 = _quantile_loss(test_obs_valid, test_q90_valid, 0.9) / denom

    print(f"Chronos test-set log-likelihood (std space): {test_ll:.6f}")
    print(f"Chronos test-set RMSE (std space): {test_rmse:.6f}")
    print(f"Chronos test-set MAE (std space): {test_mae:.6f}")
    print(f"Chronos test-set Np50: {test_p50:.6f}")
    print(f"Chronos test-set Np90: {test_p90:.6f}")

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

    output_dir = benchmark_root / f"chronos2_{_train_split_label(train_split)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "chronos_test_predictions.pdf", format="pdf")
    fig.savefig(output_dir / "chronos_test_predictions.pgf", format="pgf")
    plt.close(fig)

    summary = {
        "experiment_name": (
            f"{base_config['experiment_name']}_benchmark_"
            f"chronos2_{_train_split_label(train_split)}"
        ),
        "model": chronos_model,
        "device": device,
        "test_metrics": {
            "log_likelihood": test_ll,
            "rmse": test_rmse,
            "mae": test_mae,
            "p50": test_p50,
            "p90": test_p90,
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return {
        "condition": "chronos2",
        "global_params_label": None,
        "global_params_path": None,
        "train_split": train_split,
        "train_percentage": train_split * 100.0,
        "seed": seed,
        "test_ll": test_ll,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_p50": test_p50,
        "test_p90": test_p90,
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
        ("Global Params", "global_params_label", None),
        ("Train %", "train_percentage", ".2f"),
        ("Seed", "seed", None),
        ("LL(std)", "test_ll", ".4f"),
        ("RMSE(std)", "test_rmse", ".6f"),
        ("MAE(std)", "test_mae", ".6f"),
        ("Np50", "test_p50", ".6f"),
        ("Np90", "test_p90", ".6f"),
        ("Val LL", "val_ll", ".4f"),
        ("Val RMSE", "val_rmse", ".6f"),
    ]

    widths = []
    for header, key, fmt in columns:
        col_values = [
            _format_value(r.get(key), fmt) if fmt else _format_value(r.get(key))
            for r in results
        ]
        widths.append(max(len(header), max(len(v) for v in col_values)))

    header_line = "  ".join(h.ljust(w) for (h, _, _), w in zip(columns, widths))
    print(header_line)
    print("-" * len(header_line))

    for r in results:
        row_values = []
        for _, key, fmt in columns:
            val = r.get(key)
            row_values.append(_format_value(val, fmt) if fmt else _format_value(val))
        print("  ".join(v.ljust(w) for v, w in zip(row_values, widths)))


def _condition_label(condition: dict) -> str:
    global_params_label = condition.get("global_params_label")
    if global_params_label:
        return f"{condition['name']}[{global_params_label}]"
    return condition["name"]


def _print_aggregate(results: list[dict]):
    """Print aggregate statistics grouped by condition and training percentage."""
    print(f"\n{'=' * 80}")
    print("AGGREGATE STATISTICS")
    print(f"{'=' * 80}")

    groups = []
    for result in results:
        key = (
            result["condition"],
            result["global_params_label"],
            result["train_percentage"],
        )
        if key not in groups:
            groups.append(key)

    for cond, global_params_label, train_percentage in groups:
        cond_results = [
            r
            for r in results
            if r["condition"] == cond
            and r["global_params_label"] == global_params_label
            and r["train_percentage"] == train_percentage
        ]
        n_runs = len(cond_results)

        test_lls = [r["test_ll"] for r in cond_results if r["test_ll"] is not None]
        test_rmses = [
            r["test_rmse"] for r in cond_results if r["test_rmse"] is not None
        ]
        test_maes = [
            r.get("test_mae") for r in cond_results if r.get("test_mae") is not None
        ]
        test_p50s = [
            r.get("test_p50") for r in cond_results if r.get("test_p50") is not None
        ]
        test_p90s = [
            r.get("test_p90") for r in cond_results if r.get("test_p90") is not None
        ]
        val_lls = [r["val_ll"] for r in cond_results if r["val_ll"] is not None]
        val_rmses = [r["val_rmse"] for r in cond_results if r["val_rmse"] is not None]
        global_params_suffix = (
            f" [{global_params_label}]" if global_params_label else ""
        )

        print(
            f"\n--- {cond.upper()}{global_params_suffix} "
            f"@ {train_percentage:.2f}% TRAIN ({n_runs} runs) ---"
        )
        if test_lls:
            print(
                f"  Test LL (std):      mean={np.mean(test_lls):.4f}  "
                f"std={np.std(test_lls):.4f}  "
                f"min={np.min(test_lls):.4f}  max={np.max(test_lls):.4f}"
            )
        if test_rmses:
            print(
                f"  Test RMSE (std):    mean={np.mean(test_rmses):.6f}  "
                f"std={np.std(test_rmses):.6f}  "
                f"min={np.min(test_rmses):.6f}  max={np.max(test_rmses):.6f}"
            )
        if test_maes:
            print(
                f"  Test MAE (std):     mean={np.mean(test_maes):.6f}  "
                f"std={np.std(test_maes):.6f}  "
                f"min={np.min(test_maes):.6f}  max={np.max(test_maes):.6f}"
            )
        if test_p50s:
            print(
                f"  Test Np50:          mean={np.mean(test_p50s):.6f}  "
                f"std={np.std(test_p50s):.6f}  "
                f"min={np.min(test_p50s):.6f}  max={np.max(test_p50s):.6f}"
            )
        if test_p90s:
            print(
                f"  Test Np90:          mean={np.mean(test_p90s):.6f}  "
                f"std={np.std(test_p90s):.6f}  "
                f"min={np.min(test_p90s):.6f}  max={np.max(test_p90s):.6f}"
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
    seeds: list[int] = (1,2,3,4,5),
    train_percentages: list[float] | str | None = (1.0, 0.8, 0.6, 0.4, 0.2),
    models: list[str] | str | None = None,
    skip_models: list[str] | str | None = None,
    global_params_paths: list[str] | str | None = None,
    resume_existing: bool = False,
    chronos_model: str = "amazon/chronos-2",
    chronos_device: str = "auto",
    skip_chronos: bool = True,
):
    """Run selected filter_global_lstm variants and Chronos-2.

    Args:
        experiment_config_path: Path to the base YAML config file.
        seeds: List of random seeds to evaluate.
        train_percentages: Training data percentages to evaluate. Values can be
            percentages like 25, 50, 100 or fractions like 0.25, 0.5, 1.0.
            Defaults to the config's train_split value.
        models: Models to run. Defaults to all models. Valid names are
            local, global_finetune, global_zeroshot, chronos2.
        skip_models: Models to remove from the selected set.
        global_params_paths: Optional file, glob, directory, or list of files to use
            as lstm_global_params for global variants. Directory inputs expand to
            sorted files, and checkpoint seed labels are included in output names.
        resume_existing: If True, read completed per-run summaries instead of
            recomputing them.
        chronos_model: Chronos model identifier.
        chronos_device: Device for Chronos inference: 'cpu', 'cuda', or 'auto'.
        skip_chronos: Backward-compatible alias for skipping chronos2.
    """
    config_path = Path(experiment_config_path)
    with config_path.open("r") as f:
        base_config = yaml.safe_load(f)

    selected_models = _coerce_model_names(models, DEFAULT_MODELS)
    skipped_models = _coerce_model_names(skip_models)
    if skip_chronos and "chronos2" not in skipped_models:
        skipped_models.append("chronos2")
    selected_models = [
        model_name for model_name in selected_models if model_name not in skipped_models
    ]
    if not selected_models:
        raise ValueError("No models selected to run.")
    selected_seeds = _coerce_seeds(seeds)
    train_splits = _coerce_train_percentages(
        train_percentages,
        default=float(base_config.get("train_split", 1.0)),
    )
    selected_global_params_paths = _coerce_path_list(global_params_paths)

    original_name = base_config["experiment_name"]
    output_root = Path(base_config.get("output_root", "experiments/out"))
    benchmark_root = output_root / f"{original_name}_filter_benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    filter_conditions = _build_filter_conditions(
        base_config,
        selected_models,
        global_params_paths=selected_global_params_paths,
    )

    results = []
    result_cache = {}
    planned_filter_runs = len(train_splits) * len(selected_seeds) * len(
        filter_conditions
    )
    run_index = 0
    for train_split in train_splits:
        for seed in selected_seeds:
            for condition in filter_conditions:
                run_index += 1
                cond_name = condition["name"]
                global_params_label = condition.get("global_params_label")
                global_params_path = condition.get("global_params_path")
                cache_key = (
                    cond_name,
                    train_split,
                    global_params_path,
                )
                expected_output_dir = _run_output_dir(
                    base_config,
                    cond_name,
                    global_params_label,
                    train_split,
                    seed,
                    benchmark_root,
                )
                print(
                    f"\nProgress: filter run {run_index}/{planned_filter_runs}; "
                    f"remaining after this: {planned_filter_runs - run_index}; "
                    f"condition={cond_name}; "
                    f"global_params={global_params_label or 'default'}; "
                    f"train_percentage={train_split * 100:.2f}; seed={seed}",
                    flush=True,
                )
                if resume_existing and (expected_output_dir / "summary.json").exists():
                    print(
                        f"Resuming: found existing summary at {expected_output_dir}",
                        flush=True,
                    )
                    result = _result_from_summary(
                        expected_output_dir,
                        cond_name,
                        global_params_label,
                        global_params_path,
                        train_split,
                        seed,
                    )
                    result_cache[cache_key] = result
                    results.append(result)
                    continue
                if condition["cache_across_seeds"]:
                    if cache_key not in result_cache:
                        result = _run_single(
                            base_config,
                            seed,
                            cond_name,
                            global_params_label,
                            global_params_path,
                            condition["overrides"],
                            train_split,
                            benchmark_root,
                        )
                        result_cache[cache_key] = result
                    else:
                        print(
                            f"\n{'=' * 60}\n"
                            f"Skipping: condition={cond_name}, "
                            f"global_params="
                            f"{global_params_label or 'default'}, "
                            f"train_percentage={train_split * 100:.2f}, seed={seed} "
                            f"(deterministic — reusing seed={result_cache[cache_key]['seed']} result)"
                            f"\n{'=' * 60}"
                        )
                        result = {**result_cache[cache_key], "seed": seed}
                    results.append(result)
                else:
                    result = _run_single(
                        base_config,
                        seed,
                        cond_name,
                        global_params_label,
                        global_params_path,
                        condition["overrides"],
                        train_split,
                        benchmark_root,
                    )
                    results.append(result)

    # Run Chronos-2 baseline (deterministic, single run)
    condition_names = [_condition_label(condition) for condition in filter_conditions]
    if "chronos2" in selected_models:
        for train_split in train_splits:
            chronos_result = _run_chronos(
                base_config,
                chronos_model,
                benchmark_root,
                train_split,
                chronos_device=chronos_device,
            )
            results.append(chronos_result)
        condition_names.append("chronos2")

    if not results:
        raise ValueError(
            "No benchmark runs were executed. Check selected models and "
            "lstm_global_params for global variants."
        )

    # Print individual results
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK RESULTS: {original_name}")
    print(
        f"Seeds: {selected_seeds}  |  "
        f"Train percentages: {[split * 100 for split in train_splits]}  |  "
        f"Global params: {[str(path) for path in selected_global_params_paths] or ['config default']}  |  "
        f"Conditions: {condition_names}"
    )
    print(f"{'=' * 80}\n")
    _print_results_table(results)

    # Print aggregates
    _print_aggregate(results)

    # Save full results to JSON
    benchmark_output = {
        "experiment_name": original_name,
        "config_path": str(config_path),
        "seeds": selected_seeds,
        "train_splits": train_splits,
        "train_percentages": [split * 100.0 for split in train_splits],
        "selected_models": selected_models,
        "skipped_models": skipped_models,
        "global_params_paths": [str(path) for path in selected_global_params_paths],
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
