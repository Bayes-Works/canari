import csv
import json
import multiprocessing as mp
from pathlib import Path

import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pytagi import Normalizer as normalizer
from pytagi import metric

from canari import Model, plot_data, plot_prediction, plot_states
from canari.common import calc_observation
from canari.component import LocalLevel, LstmNetwork, WhiteNoise
from canari.data_visualization import _add_dynamic_grids

try:
    from experiments.utils import prepare_dataset
except ModuleNotFoundError:
    from utils import prepare_dataset


# Module-level state for multiprocessing workers (fork context).
_sigma_v_train_fn = None


def _sigma_v_worker(sv_candidate):
    """Worker for parallel sigma_v grid search."""
    model_dict, metrics, optimal_epoch, optimal_metrics = _sigma_v_train_fn(
        sv_candidate
    )
    return {
        "sigma_v": sv_candidate,
        "model_dict": model_dict,
        "metrics": metrics,
        "optimal_epoch": int(optimal_epoch),
        "optimal_validation_metrics": optimal_metrics,
    }


mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
        "lines.linewidth": 1,
    }
)


def _initialize_constant_trend_baseline_states(
    model: Model,
    init_data: np.ndarray,
    level_var_floor: float = 1e-4,
    trend_var_floor: float = 1e-6,
    acceleration_var: float = 1e-6,
) -> None:
    y = np.asarray(init_data, dtype=float).reshape(-1)
    y = y[np.isfinite(y)]
    if y.size == 0:
        raise ValueError("Baseline initialization window is empty.")

    trend = 0.0
    anchor_len = min(max(3, y.size // 4), y.size)
    level = float(np.median(y[:anchor_len]))

    residuals = y - level
    residual_scale = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))
    if not np.isfinite(residual_scale):
        residual_scale = 0.0
    level_var_floor = max(level_var_floor, residual_scale**2)
    trend_var_floor = max(trend_var_floor, 1e-8)

    for i, state_name in enumerate(model.states_name):
        if state_name == "level":
            model.mu_states[i] = level
            model.var_states[i, i] = level_var_floor
        elif state_name == "trend":
            model.mu_states[i] = trend
            model.var_states[i, i] = trend_var_floor
        elif state_name == "acceleration":
            model.mu_states[i] = 0.0
            model.var_states[i, i] = acceleration_var

    model._mu_local_level = level


def _plot_raw_data(
    dataset: dict,
    output_dir: Path,
    output_filename: str = "raw_data.pdf",
) -> Path:
    plot_data_processor = dataset.get("plot_data_processor", dataset["data_processor"])
    data_df = plot_data_processor.data
    target_col = int(plot_data_processor.output_col[0])
    time_axis = data_df.index
    values = data_df.iloc[:, target_col].to_numpy(dtype=float)
    warmup_time = np.asarray(dataset.get("warmup_time", np.array([], dtype=object)))
    warmup_values = np.asarray(dataset.get("warmup_values", np.array([], dtype=float)))

    train_idx, validation_idx, test_idx = plot_data_processor.get_split_indices()
    unused_train_rows = int(
        np.clip(dataset.get("unused_train_rows", 0), 0, len(train_idx))
    )
    train_rows_full = int(max(len(train_idx), 1))
    train_rows_used = int(max(len(train_idx) - unused_train_rows, 0))
    train_use_ratio = train_rows_used / train_rows_full
    truncated_alpha = float(np.clip(0.10 + 0.35 * train_use_ratio, 0.10, 0.45))
    line_color = "r"

    fig_raw, ax_raw = plt.subplots(figsize=(6.4, 2.6))

    def _valid_segment(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(indices) == 0:
            return np.array([], dtype=object), np.array([], dtype=float)
        segment_time = np.asarray(time_axis[indices])
        segment_values = values[indices]
        valid_mask = ~np.isnat(segment_time)
        return segment_time[valid_mask], segment_values[valid_mask]

    def _span_valid_range(indices: np.ndarray, color: str, alpha: float) -> None:
        segment_time, _ = _valid_segment(indices)
        if len(segment_time) == 0:
            return
        ax_raw.axvspan(
            segment_time[0],
            segment_time[-1],
            color=color,
            alpha=alpha,
            linewidth=0,
        )

    if len(warmup_time) > 0:
        warmup_valid_mask = ~np.isnat(warmup_time)
        ax_raw.plot(
            warmup_time[warmup_valid_mask],
            warmup_values[warmup_valid_mask],
            color="k",
            alpha=0.65,
        )

    used_train_idx = train_idx[unused_train_rows:]
    if unused_train_rows > 0:
        truncated_time, truncated_values = _valid_segment(train_idx[:unused_train_rows])
        ax_raw.plot(
            truncated_time,
            truncated_values,
            color=line_color,
            alpha=truncated_alpha,
        )
    if len(used_train_idx) > 0:
        used_train_time, used_train_values = _valid_segment(used_train_idx)
        ax_raw.plot(
            used_train_time,
            used_train_values,
            color=line_color,
            alpha=0.95,
        )
    if len(validation_idx) > 0:
        validation_time, validation_values = _valid_segment(validation_idx)
        ax_raw.plot(
            validation_time,
            validation_values,
            color=line_color,
            alpha=0.95,
        )
    if len(test_idx) > 0:
        test_time, test_values = _valid_segment(test_idx)
        ax_raw.plot(
            test_time,
            test_values,
            color=line_color,
            alpha=0.95,
        )

    _span_valid_range(validation_idx, color="green", alpha=0.1)

    ax_raw.set_xlabel("Time")
    ax_raw.set_ylabel("Value")
    _add_dynamic_grids(ax_raw, time_axis)
    plt.tight_layout()

    raw_plot_path = output_dir / output_filename
    fig_raw.savefig(raw_plot_path, format="pdf")
    plt.close(fig_raw)
    return raw_plot_path


def _plot_filtered_predictions(
    data_processor,
    mean_all_pred: np.ndarray,
    std_all_pred: np.ndarray,
    mean_all_posterior: np.ndarray,
    std_all_posterior: np.ndarray,
    output_dir: Path,
) -> Path:
    train_idx, validation_idx, test_idx = data_processor.get_split_indices()

    fig_pred, ax_pred = plt.subplots(figsize=(11.5, 4.5))
    plot_data(
        data_processor=data_processor,
        standardization=False,
        plot_column=data_processor.output_col,
        train_label="y",
        validation_label=None,
        test_label=None,
    )
    plot_prediction(
        data_processor=data_processor,
        mean_train_pred=mean_all_pred[train_idx],
        std_train_pred=std_all_pred[train_idx],
        mean_validation_pred=mean_all_pred[validation_idx],
        std_validation_pred=std_all_pred[validation_idx],
        mean_test_pred=mean_all_pred[test_idx],
        std_test_pred=std_all_pred[test_idx],
        color="#1d4ed8",
        train_label=[r"prior $\mu$", r"prior $\pm\sigma$"],
        validation_label=["", ""],
        test_label=["", ""],
    )
    plot_prediction(
        data_processor=data_processor,
        mean_train_pred=mean_all_posterior[train_idx],
        std_train_pred=std_all_posterior[train_idx],
        mean_validation_pred=mean_all_posterior[validation_idx],
        std_validation_pred=std_all_posterior[validation_idx],
        mean_test_pred=mean_all_posterior[test_idx],
        std_test_pred=std_all_posterior[test_idx],
        color="#15803d",
        linestyle="--",
        train_label=[r"posterior $\mu$", r"posterior $\pm\sigma$"],
        validation_label=["", ""],
        test_label=["", ""],
    )

    _add_dynamic_grids(ax_pred, data_processor.data.index)
    ax_pred.legend(loc="upper left", ncol=4)
    plt.tight_layout()

    prediction_plot_path = output_dir / "filtered_predictions.pdf"
    fig_pred.savefig(prediction_plot_path, format="pdf")
    plt.close(fig_pred)
    return prediction_plot_path


def _save_filtered_predictions_csv(
    data_processor,
    mean_all_pred: np.ndarray,
    std_all_pred: np.ndarray,
    output_dir: Path,
) -> Path:
    obs = data_processor.get_data("all").reshape(-1)
    time_axis = data_processor.get_time("all")
    train_idx, validation_idx, test_idx = data_processor.get_split_indices()
    split_labels = np.full(len(obs), "test", dtype=object)
    split_labels[train_idx] = "train"
    split_labels[validation_idx] = "validation"
    split_labels[test_idx] = "test"

    csv_path = output_dir / "filtered_predictions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["time", "split", "obs", "mu_pred", "std_pred", "residual"],
        )
        writer.writeheader()
        for idx in range(len(obs)):
            writer.writerow(
                {
                    "time": str(time_axis[idx]),
                    "split": split_labels[idx],
                    "obs": float(obs[idx]),
                    "mu_pred": float(mean_all_pred[idx]),
                    "std_pred": float(std_all_pred[idx]),
                    "residual": float(obs[idx] - mean_all_pred[idx]),
                }
            )
    return csv_path


def _plot_training_metrics(
    training_metrics_history: list[dict], output_dir: Path
) -> Path | None:
    if len(training_metrics_history) == 0:
        return None

    epochs = [m["epoch"] for m in training_metrics_history]
    val_ll = [m["validation_log_likelihood"] for m in training_metrics_history]
    val_rmse = [m["validation_rmse"] for m in training_metrics_history]
    best_epoch = int(
        min(
            training_metrics_history,
            key=lambda metric_entry: -metric_entry["validation_log_likelihood"],
        )["epoch"]
    )

    fig_metrics, axes_metrics = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes_metrics[0].plot(epochs, val_ll, color="#b91c1c", linewidth=2)
    axes_metrics[0].axvline(
        best_epoch, color="k", linestyle="--", linewidth=1, label="optimal epoch"
    )
    axes_metrics[0].set_ylabel("Validation log-likelihood")
    axes_metrics[0].grid(alpha=0.25)
    axes_metrics[0].legend(loc="best")

    axes_metrics[1].plot(epochs, val_rmse, color="#1d4ed8", linewidth=2)
    axes_metrics[1].axvline(best_epoch, color="k", linestyle="--", linewidth=1)
    axes_metrics[1].set_ylabel("Validation RMSE")
    axes_metrics[1].set_xlabel("Epoch")
    axes_metrics[1].grid(alpha=0.25)
    plt.tight_layout()

    output_path = output_dir / "training_metrics_by_epoch.pdf"
    fig_metrics.savefig(output_path, format="pdf")
    plt.close(fig_metrics)
    return output_path


def main(
    experiment_config_path: str = "/home/dw/canari/experiments/config/ID_timeseries/LTU009EFAPRG024.yaml",
):
    experiment_config_path = Path(experiment_config_path)
    with experiment_config_path.open("r") as f:
        experiment_config = yaml.safe_load(f)
    effective_experiment_config = dict(experiment_config)

    # Apply runtime overrides to the effective config
    effective_experiment_config["anomaly_slope"] = 0.0
    effective_experiment_config["experiment_name"] = (
        f"{effective_experiment_config['experiment_name']}_filter"
    )
    effective_experiment_config.setdefault(
        "anomaly_start_time",
        effective_experiment_config["validation_start"],
    )

    experiment_name = effective_experiment_config["experiment_name"]
    output_root = Path(
        effective_experiment_config.get("output_root", "experiments/out")
    )
    output_dir = output_root / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    used_config_path = output_dir / "experiment_config_used.yaml"
    with used_config_path.open("w") as f:
        yaml.safe_dump(effective_experiment_config, f, sort_keys=False)

    dataset = prepare_dataset(
        train_split=float(effective_experiment_config["train_split"]),
        anomaly_slope=float(effective_experiment_config["anomaly_slope"]),
        experiment_config=effective_experiment_config,
    )

    train_data = dataset["train_data"]
    validation_data = dataset["validation_data"]
    data_processor = dataset["data_processor"]
    all_data = dataset["all_data"]
    warmup_lookback_mu = dataset["warmup_lookback_mu"]
    warmup_lookback_var = dataset["warmup_lookback_var"]

    look_back_len = int(effective_experiment_config["lstm_look_back_len"])
    num_features = int(effective_experiment_config["lstm_num_features"])
    num_layer = int(effective_experiment_config["lstm_num_layer"])
    infer_len = int(effective_experiment_config["lstm_infer_len"])
    num_hidden_unit = int(effective_experiment_config["num_hidden_unit"])
    seed = effective_experiment_config["lstm_manual_seed"]
    smoother = bool(effective_experiment_config["smoother"])
    stateless = bool(effective_experiment_config["lstm_stateless"])
    zero_shot = bool(effective_experiment_config.get("lstm_zeroshot", False))
    finetune = bool(effective_experiment_config["lstm_finetune"])
    increase_output_variance = bool(
        effective_experiment_config.get("lstm_increase_output_variance", False)
    )
    global_params = effective_experiment_config.get("lstm_global_params")
    use_tagiv = bool(effective_experiment_config["use_tagiv"])
    sigma_v = float(effective_experiment_config["sigma_v"])
    if zero_shot:
        max_num_epoch = 1
    else:
        max_num_epoch = int(effective_experiment_config.get("lstm_num_epoch", 50))

    sigma_v_range = effective_experiment_config.get(
        "sigma_v_search_space", [0.02, 0.04, 0.06, 0.08, 0.1, 0.125, 0.15, 0.175, 0.2]
    )
    sigma_v_max_concurrent = int(
        effective_experiment_config.get("sigma_v_max_concurrent", 6)
    )

    if len(warmup_lookback_mu) != look_back_len:
        raise ValueError(
            "Warmup lookback length does not match the LSTM lookback length: "
            f"got global_warmup_lookback_len={len(warmup_lookback_mu)} and "
            f"lstm_look_back_len={look_back_len}."
        )

    validation_obs = data_processor.get_data("validation").flatten()

    def _train_lstm(sv_value: float):
        lstm_kwargs = dict(
            look_back_len=look_back_len,
            num_features=num_features,
            num_layer=num_layer,
            infer_len=infer_len,
            num_hidden_unit=num_hidden_unit,
            device="cpu",
            manual_seed=seed,
            smoother=smoother,
            stateless=stateless,
            finetune=finetune,
            increase_output_variance=increase_output_variance,
            load_lstm_net=global_params,
            model_noise=use_tagiv,
            zeroshot=zero_shot,
        )

        def _build_components():
            parts = [
                LocalLevel(mu_states=[0.0], var_states=[0.0]),
                LstmNetwork(**lstm_kwargs),
            ]
            if not use_tagiv:
                parts.append(WhiteNoise(std_error=sv_value))
            return parts

        try:
            model = Model(*_build_components())
        except RuntimeError as exc:
            if global_params and "Failed to load LSTM network from" in str(exc):
                print(
                    "Warning: incompatible pretrained LSTM weights at "
                    f"'{global_params}'. Falling back to random initialization."
                )
                lstm_kwargs["load_lstm_net"] = None
                lstm_kwargs["finetune"] = False
                lstm_kwargs["increase_output_variance"] = False
                model = Model(*_build_components())
            else:
                raise

        model.lstm_net.teacher_forcing = True

        local_training_metrics = []
        local_optimal_metrics = None

        for epoch in range(max_num_epoch):
            if model.lstm_net.smooth is False:
                model.lstm_output_history.set(warmup_lookback_mu, warmup_lookback_var)

            mu_validation_preds, std_validation_preds, _ = model.lstm_train(
                train_data=train_data,
                validation_data=validation_data,
                white_noise_decay=False,
            )

            mu_validation_preds_unnorm = normalizer.unstandardize(
                mu_validation_preds,
                data_processor.scale_const_mean[data_processor.output_col],
                data_processor.scale_const_std[data_processor.output_col],
            )
            std_validation_preds_unnorm = normalizer.unstandardize_std(
                std_validation_preds,
                data_processor.scale_const_std[data_processor.output_col],
            )

            validation_log_lik = metric.log_likelihood(
                prediction=mu_validation_preds_unnorm,
                observation=validation_obs,
                std=std_validation_preds_unnorm,
            )
            validation_rmse = np.sqrt(
                np.nanmean((mu_validation_preds_unnorm - validation_obs) ** 2)
            )

            epoch_metrics = {
                "epoch": epoch,
                "validation_log_likelihood": float(validation_log_lik),
                "validation_rmse": float(validation_rmse),
            }
            local_training_metrics.append(epoch_metrics)

            model.early_stopping(
                evaluate_metric=-validation_log_lik,
                current_epoch=epoch,
                max_epoch=max_num_epoch,
                skip_epoch=0,
            )

            if local_optimal_metrics is None or (
                validation_log_lik > local_optimal_metrics["validation_log_likelihood"]
            ):
                local_optimal_metrics = epoch_metrics

            if model.stop_training:
                break

        model.lstm_net.teacher_forcing = False
        return (
            model.get_dict(time_step=0),
            local_training_metrics,
            int(model.optimal_epoch),
            local_optimal_metrics,
        )

    ######### sigma_v grid search #########
    sigma_v_grid_search_result = None
    if bool(effective_experiment_config.get("optimize_sigma_v", False)):
        print(
            "---- sigma_v grid search over "
            f"{sigma_v_range} (metric: validation_log_likelihood) ----"
        )

        if sigma_v_max_concurrent > 1:
            global _sigma_v_train_fn
            _sigma_v_train_fn = _train_lstm
            ctx = mp.get_context("fork")
            with ctx.Pool(sigma_v_max_concurrent) as pool:
                grid_results = pool.map(_sigma_v_worker, sigma_v_range)
        else:
            grid_results = []
            for sv_candidate in sigma_v_range:
                sv_dict, sv_metrics, sv_opt_epoch, sv_opt_metrics = _train_lstm(
                    sv_candidate
                )
                grid_results.append(
                    {
                        "sigma_v": sv_candidate,
                        "model_dict": sv_dict,
                        "metrics": sv_metrics,
                        "optimal_epoch": sv_opt_epoch,
                        "optimal_validation_metrics": sv_opt_metrics,
                    }
                )

        for r in grid_results:
            print(
                f"  sigma_v={r['sigma_v']:.4f} -> "
                f"best validation_log_likelihood="
                f"{r['optimal_validation_metrics']['validation_log_likelihood']:.6f} "
                f"(epoch {r['optimal_epoch']})"
            )

        selection_delta = float(
            effective_experiment_config.get("sigma_v_selection_delta", 0.0)
        )
        scored = [
            (r, r["optimal_validation_metrics"]["validation_log_likelihood"])
            for r in grid_results
        ]
        global_best_metric = max(s for _, s in scored)
        tolerated = [
            (r, s) for r, s in scored if s >= global_best_metric - selection_delta
        ]
        best_entry, best_validation_metric = min(
            tolerated, key=lambda rs: rs[0]["sigma_v"]
        )
        best_sigma_v = float(best_entry["sigma_v"])

        sigma_v = best_sigma_v
        sigma_v_grid_search_result = {
            "selection_metric": "validation_log_likelihood",
            "selection_delta": selection_delta,
            "optimal_sigma_v": best_sigma_v,
            "best_validation_metric": float(best_validation_metric),
            "global_best_metric": float(global_best_metric),
            "grid_results": [
                {
                    "sigma_v": float(r["sigma_v"]),
                    "optimal_epoch": int(r["optimal_epoch"]),
                    "validation_log_likelihood": float(
                        r["optimal_validation_metrics"]["validation_log_likelihood"]
                    ),
                    "validation_rmse": float(
                        r["optimal_validation_metrics"]["validation_rmse"]
                    ),
                }
                for r in grid_results
            ],
        }
        print(
            f"---- Optimal sigma_v: {sigma_v:.4f} "
            f"(validation_log_likelihood={best_validation_metric:.6f}) ----"
        )

        training_metrics_history = best_entry["metrics"]
        optimal_validation_metrics = best_entry["optimal_validation_metrics"]
        model = Model.load_dict(best_entry["model_dict"])
    else:
        print(f"---- Training LSTM once with sigma_v={sigma_v:.4f} ----")
        model_dict, training_metrics_history, _, optimal_validation_metrics = (
            _train_lstm(sigma_v)
        )
        model = Model.load_dict(model_dict)

    if model.lstm_net.smooth is False:
        model.lstm_output_history.set(warmup_lookback_mu, warmup_lookback_var)
    mu_filter_preds, std_filter_preds, states = model.filter(
        data=all_data,
        train_lstm=False,
    )

    mu_filter_preds_unnorm = normalizer.unstandardize(
        mu_filter_preds,
        data_processor.scale_const_mean[data_processor.output_col],
        data_processor.scale_const_std[data_processor.output_col],
    )
    std_filter_preds_unnorm = normalizer.unstandardize_std(
        std_filter_preds,
        data_processor.scale_const_std[data_processor.output_col],
    )

    mu_posterior_obs = np.empty(len(states.mu_posterior))
    std_posterior_obs = np.empty(len(states.mu_posterior))
    for t, (mu_p, var_p) in enumerate(zip(states.mu_posterior, states.var_posterior)):
        mu_obs_p, var_obs_p = calc_observation(mu_p, var_p, model.observation_matrix)
        mu_posterior_obs[t] = float(np.asarray(mu_obs_p).flatten()[0])
        std_posterior_obs[t] = float(np.sqrt(np.asarray(var_obs_p).flatten()[0]))

    mu_posterior_obs_unnorm = normalizer.unstandardize(
        mu_posterior_obs,
        data_processor.scale_const_mean[data_processor.output_col],
        data_processor.scale_const_std[data_processor.output_col],
    )
    std_posterior_obs_unnorm = normalizer.unstandardize_std(
        std_posterior_obs,
        data_processor.scale_const_std[data_processor.output_col],
    )

    _, _, test_idx = data_processor.get_split_indices()
    test_obs = data_processor.get_data("test").flatten()
    mu_test_preds_unnorm = mu_filter_preds_unnorm[test_idx]
    std_test_preds_unnorm = std_filter_preds_unnorm[test_idx]

    # LL / RMSE / MAE in standardized space
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
    mu_test_std = mu_filter_preds[test_idx]
    std_test_std = std_filter_preds[test_idx]
    test_obs_std = (test_obs - mean_col) / std_col

    filter_log_lik = metric.log_likelihood(
        prediction=mu_test_std,
        observation=test_obs_std,
        std=std_test_std,
    )
    res_std = mu_test_std - test_obs_std
    filter_rmse = float(np.sqrt(np.nanmean(res_std**2)))
    filter_mae = float(np.nanmean(np.abs(res_std)))

    # P50 / P90 = gluonts normalized quantile loss on original-space predictions
    Z90 = 1.2815515655446004
    q50_pred = mu_test_preds_unnorm
    q90_pred = mu_test_preds_unnorm + Z90 * std_test_preds_unnorm
    denom = float(np.nansum(np.abs(test_obs))) + 1e-8

    def _quantile_loss(target, forecast, q):
        return 2.0 * float(
            np.nansum(
                np.abs((forecast - target) * ((target <= forecast).astype(float) - q))
            )
        )

    filter_p50 = _quantile_loss(test_obs, q50_pred, 0.5) / denom
    filter_p90 = _quantile_loss(test_obs, q90_pred, 0.9) / denom

    print(
        "Validation metrics at optimal epoch: "
        f"epoch={optimal_validation_metrics['epoch']} "
        f"val_ll={optimal_validation_metrics['validation_log_likelihood']:.6f} "
        f"val_rmse={optimal_validation_metrics['validation_rmse']:.6f}"
    )
    print(f"Test-set filter log-likelihood (std space): {filter_log_lik:.6f}")
    print(f"Test-set filter RMSE (std space): {filter_rmse:.6f}")
    print(f"Test-set filter MAE (std space): {filter_mae:.6f}")
    print(f"Test-set filter Np50: {filter_p50:.6f}")
    print(f"Test-set filter Np90: {filter_p90:.6f}")

    raw_plot_path = _plot_raw_data(
        dataset=dataset,
        output_dir=output_dir,
    )
    prediction_plot_path = _plot_filtered_predictions(
        data_processor=data_processor,
        mean_all_pred=mu_filter_preds_unnorm,
        std_all_pred=std_filter_preds_unnorm,
        mean_all_posterior=mu_posterior_obs_unnorm,
        std_all_posterior=std_posterior_obs_unnorm,
        output_dir=output_dir,
    )
    predictions_csv_path = _save_filtered_predictions_csv(
        data_processor=data_processor,
        mean_all_pred=mu_filter_preds_unnorm,
        std_all_pred=std_filter_preds_unnorm,
        output_dir=output_dir,
    )
    training_metrics_plot_path = _plot_training_metrics(
        training_metrics_history=training_metrics_history,
        output_dir=output_dir,
    )

    fig_states, axes_states = plot_states(
        data_processor=data_processor,
        states=states,
        standardization=False,
        states_type="posterior",
    )
    plt.tight_layout()
    states_plot_path = output_dir / "model_states.pdf"
    fig_states.savefig(states_plot_path, format="pdf")
    plt.close(fig_states)

    summary = {
        "experiment_name": experiment_name,
        "sigma_v": float(sigma_v),
        "validation_metrics_best": optimal_validation_metrics,
        "test_metrics": {
            "log_likelihood": float(filter_log_lik),
            "rmse": filter_rmse,
            "mae": filter_mae,
            "p50": filter_p50,
            "p90": filter_p90,
        },
        "artifacts": {
            "raw_plot": str(raw_plot_path),
            "prediction_plot": str(prediction_plot_path),
            "predictions_csv": str(predictions_csv_path),
            "training_metrics_plot": (
                str(training_metrics_plot_path)
                if training_metrics_plot_path is not None
                else None
            ),
            "states_plot": str(states_plot_path),
        },
    }
    if sigma_v_grid_search_result is not None:
        summary["sigma_v_grid_search"] = sigma_v_grid_search_result
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    fire.Fire(main)
