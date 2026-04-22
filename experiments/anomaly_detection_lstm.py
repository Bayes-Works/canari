import fire
import multiprocessing as mp
import os
import numpy as np
from scipy.stats import norm as _norm_dist
import matplotlib.pyplot as plt
from ray import tune
from pytagi import metric
from pytagi import Normalizer as normalizer
from canari import (
    Model,
    Optimizer,
    SKF,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise

try:
    from experiments.utils import prepare_dataset
except ModuleNotFoundError:
    from utils import prepare_dataset

from canari.data_process import DataProcess

from pathlib import Path
import yaml
import json


# Module-level state for multiprocessing workers (fork context)
_sigma_v_train_fn = None


def _sigma_v_worker(sv_candidate):
    """Worker for parallel sigma_v grid search."""
    model, metrics, std_per_epoch = _sigma_v_train_fn(sv_candidate)
    best_epoch = max(0, min(int(model.optimal_epoch), len(metrics) - 1))
    return {
        "sigma_v": sv_candidate,
        "validation_crps": float(metrics[best_epoch]["validation_crps"]),
        "validation_log_likelihood": float(
            metrics[best_epoch]["validation_log_likelihood"]
        ),
        "optimal_epoch": int(model.optimal_epoch),
        "model_dict": model.get_dict(time_step=0),
        "metrics": metrics,
        "std_per_epoch": std_per_epoch,
    }




# Plotting defaults
import matplotlib as mpl

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


def _crps_gaussian(mu: np.ndarray, std: np.ndarray, obs: np.ndarray) -> float:
    """Mean CRPS for Gaussian predictive distributions (closed-form)."""
    z = (obs - mu) / std
    return float(np.nanmean(std * (z * (2 * _norm_dist.cdf(z) - 1) + 2 * _norm_dist.pdf(z) - 1.0 / np.sqrt(np.pi))))


def _estimate_years_per_step(time_axis) -> float:
    """Estimate the sampling interval in years, ignoring trailing NaT values."""

    time_values = np.asarray(time_axis, dtype="datetime64[ns]")
    valid_time = time_values[~np.isnat(time_values)]
    if valid_time.size >= 2:
        step_days = np.diff(valid_time) / np.timedelta64(1, "D")
        step_days = step_days[np.isfinite(step_days) & (step_days > 0)]
        if step_days.size > 0:
            return float(np.median(step_days) / 365.25)

    # Fall back to a weekly cadence, which matches the current experiment setup.
    return float(7.0 / 365.25)



def _plot_multi_realization_skf_result(
    output_path: Path,
    clean_y: np.ndarray,
    eval_y: np.ndarray,
    filter_probs: np.ndarray,
    anom_start: int,
    threshold: float,
    magnitude: float,
    realization_idx: int,
    direction: int,
    detected_idx: int | None,
    scale_mean: float,
    scale_std: float,
) -> None:
    """Save one SKF result plot for a single anomaly realization."""

    step_axis = np.arange(len(eval_y), dtype=int)
    eval_y_plot = normalizer.unstandardize(eval_y, scale_mean, scale_std).reshape(-1)
    clean_y_plot = normalizer.unstandardize(clean_y, scale_mean, scale_std).reshape(-1)
    direction_label = "up" if direction > 0 else "down"

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)

    axes[0].plot(
        step_axis,
        clean_y_plot,
        color="#94a3b8",
        linewidth=1.2,
        linestyle="--",
        label="clean",
    )
    axes[0].plot(
        step_axis,
        eval_y_plot,
        color="#b91c1c",
        linewidth=1.6,
        label="with anomaly",
    )
    axes[0].axvline(anom_start, color="k", linestyle="--", linewidth=1.0)
    if detected_idx is not None:
        axes[0].axvline(detected_idx, color="#2563eb", linestyle=":", linewidth=1.2)
    axes[0].set_ylabel("Value")
    axes[0].set_title(
        f"SKF realization {realization_idx + 1} | mag={magnitude:.1f} | {direction_label}"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(
        step_axis,
        np.asarray(filter_probs).reshape(-1),
        color="#dc2626",
        linewidth=1.6,
        label="P(abnormal)",
    )
    axes[1].axhline(
        threshold, color="k", linestyle="--", linewidth=1.0, label="threshold"
    )
    axes[1].axvline(
        anom_start, color="k", linestyle="--", linewidth=1.0, label="anomaly start"
    )
    if detected_idx is not None:
        axes[1].axvline(
            detected_idx,
            color="#2563eb",
            linestyle=":",
            linewidth=1.2,
            label="first detection",
        )
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Probability")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    plt.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def main(
    experiment_config_path: str = "./experiments/config/ID_timeseries/LGA008EFAPRG910.yaml",
):

    # Read config file
    experiment_config_path = Path(experiment_config_path)
    with experiment_config_path.open("r") as f:
        experiment_config = yaml.safe_load(f)
    experiment_config.setdefault("lstm_early_stopping_metric", "crps")
    experiment_name = experiment_config["experiment_name"]
    output_root = Path(experiment_config.get("output_root", "experiments/out"))
    output_dir = output_root / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    used_config_path = output_dir / "experiment_config_used.yaml"
    with used_config_path.open("w") as f:
        yaml.safe_dump(experiment_config, f, sort_keys=False)

    # Data preperation
    dataset = prepare_dataset(
        train_split=float(experiment_config["train_split"]),
        anomaly_slope=0.0,
        experiment_config=experiment_config,
    )
    train_data = dataset["train_data"]
    validation_data = dataset["validation_data"]
    data_processor = dataset["data_processor"]
    all_data = dataset["all_data"]
    warmup_lookback_mu = dataset["warmup_lookback_mu"]
    warmup_lookback_var = dataset["warmup_lookback_var"]
    train_val = dataset["train_val"]

    # Define model with parameters
    look_back_len = experiment_config["lstm_look_back_len"]
    num_features = experiment_config["lstm_num_features"]
    num_layer = experiment_config["lstm_num_layer"]
    infer_len = experiment_config["lstm_infer_len"]
    num_hidden_unit = experiment_config["num_hidden_unit"]
    seed = experiment_config["lstm_manual_seed"]
    smoother = experiment_config["smoother"]
    sigma_v = experiment_config["sigma_v"]
    stateless = experiment_config["lstm_stateless"]
    zero_shot = experiment_config["lstm_zeroshot"]
    finetune = experiment_config["lstm_finetune"]
    increase_output_variance = bool(
        experiment_config.get("lstm_increase_output_variance", False)
    )
    global_params = experiment_config.get("lstm_global_params")
    use_tagiv = experiment_config["use_tagiv"]
    max_num_epoch = int(experiment_config.get("lstm_num_epoch", 100))
    lstm_early_stopping_metric = str(
        experiment_config.get("lstm_early_stopping_metric", "crps")
    ).strip().lower()
    if lstm_early_stopping_metric == "ll":
        lstm_early_stopping_metric_key = "validation_log_likelihood"
        lstm_early_stopping_mode = "max"
        lstm_early_stopping_metric_label = "LL"
    elif lstm_early_stopping_metric == "crps":
        lstm_early_stopping_metric_key = "validation_crps"
        lstm_early_stopping_mode = "min"
        lstm_early_stopping_metric_label = "CRPS"
    else:
        raise ValueError(
            "lstm_early_stopping_metric must be either 'll' or 'crps'."
        )
    likelihood_covariance_floor = float(
        experiment_config.get("likelihood_covariance_floor", 0.0)
    )
    skf_objective_function = experiment_config.get("skf_objective_function", "ll")
    abnorm_to_norm_prob = float(experiment_config.get("abnorm_to_norm_prob", 0.1))
    default_skf_param = {
        "sigma_v": sigma_v,
        "std_transition_error": float(experiment_config["std_transition_error"]),
        "norm_to_abnorm_prob": float(experiment_config["norm_to_abnorm_prob"]),
        "abnorm_to_norm_prob": abnorm_to_norm_prob,
        "likelihood_covariance_floor": likelihood_covariance_floor,
        "threshold": float(experiment_config.get("anomaly_detection_threshold", 0.4)),
    }
    threshold = default_skf_param["threshold"]
    max_timestep_to_detect = float(experiment_config.get("max_timestep_to_detect ", 156))

    num_realizations = int(experiment_config.get("num_anomaly_realizations", 25))
    anomaly_magnitudes = experiment_config.get("slope_search_space",[0.025, 0.05, 0.075, 0.225, 0.5, 0.75, 1.0])
    sigma_v_range = experiment_config.get("sigma_v_search_space", [0.02, 0.04, 0.06, 0.08, 0.1, 0.125, 0.15, 0.175, 0.2])
    available_cpus = max(1, os.cpu_count() or 1)

    lstm_num_thread = 1
    sigma_v_max_concurrent = 6
    skf_tuning_n_jobs = 6
    skf_eval_n_jobs = 6


    cdf_synthetic_cache = {}
    if skf_objective_function == "cdf":
        default_skf_param["slope"] = float(experiment_config["slope"])
        cdf_num_anomaly = int(experiment_config.get("cdf_num_anomaly", 50))
        cdf_anomaly_start = float(experiment_config.get("cdf_anomaly_start", 0.25))
        cdf_anomaly_end = float(experiment_config.get("cdf_anomaly_end", 0.75))

        cdf_cache_slopes = set(float(m) for m in anomaly_magnitudes)
        cdf_cache_slopes.add(float(default_skf_param["slope"]))
        for slope_value in cdf_cache_slopes:
            slope_anomaly = slope_value / 52
            cdf_synthetic_cache[slope_value] = DataProcess.add_synthetic_anomaly(
                train_val,
                num_samples=cdf_num_anomaly,
                slope=[slope_anomaly, -slope_anomaly],
                anomaly_start=cdf_anomaly_start,
                anomaly_end=cdf_anomaly_end,
            )


    print(
        "Parallelism: "
        f"lstm_threads={lstm_num_thread}, "
        f"sigma_v_workers={sigma_v_max_concurrent}, "
        f"skf_tuning_workers={skf_tuning_n_jobs}, "
        f"skf_eval_workers={skf_eval_n_jobs}"
    )

    optimal_validation_metrics = {}
    training_metrics_history = []
    lstm_std_per_epoch = []
    validation_obs = data_processor.get_data("validation").flatten()

    def _timestamp_at_exclusive_end(end_idx: int):
        """Convert an exclusive split bound into the last in-range timestamp."""

        last_idx = min(max(end_idx - 1, 0), len(data_processor.data.index) - 1)
        return data_processor.data.index[last_idx]

    def _selected_validation_metric(metrics, optimal_epoch: int) -> float:
        best_epoch = max(0, min(int(optimal_epoch), len(metrics) - 1))
        return float(metrics[best_epoch][lstm_early_stopping_metric_key])

    def _train_lstm(sv_value):
        lstm_kwargs = dict(
            look_back_len=look_back_len,
            num_features=num_features,
            num_layer=num_layer,
            infer_len=infer_len,
            num_hidden_unit=num_hidden_unit,
            device="cpu",
            num_thread=lstm_num_thread,
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
            parts = [LocalTrend(), LstmNetwork(**lstm_kwargs)]
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

        # init baseline states
        model.auto_initialize_baseline_states(
            train_data["y"][0 : experiment_config["baseline_init_len"]]
        )
        model.mu_states[model.get_states_index("trend")] = 0.0  # force trend to

        model.lstm_net.teacher_forcing = False

        num_epoch = max_num_epoch
        local_training_metrics = []
        local_lstm_std_per_epoch = [np.array([np.nan]) for _ in range(num_epoch)]
        for epoch in range(num_epoch):
            if model.lstm_net.smooth is False:
                model.lstm_output_history.set(warmup_lookback_mu, warmup_lookback_var)

            mu_validation_preds, std_validation_preds, train_states = model.lstm_train(
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
            validation_crps = _crps_gaussian(
                mu_validation_preds_unnorm, std_validation_preds_unnorm, validation_obs
            )
            epoch_metrics = {
                "epoch": epoch,
                "validation_log_likelihood": float(validation_log_lik),
                "validation_rmse": float(validation_rmse),
                "validation_crps": float(validation_crps),
            }
            local_training_metrics.append(epoch_metrics)
            std_lstm_prior = np.asarray(train_states.get_std("lstm", "prior")).flatten()
            std_lstm_prior = std_lstm_prior[np.isfinite(std_lstm_prior)]
            if std_lstm_prior.size == 0:
                std_lstm_prior = np.array([np.nan])
            local_lstm_std_per_epoch[epoch] = std_lstm_prior

            model.early_stopping(
                evaluate_metric=epoch_metrics[lstm_early_stopping_metric_key],
                current_epoch=epoch,
                max_epoch=num_epoch,
                mode=lstm_early_stopping_mode,
                skip_epoch=0,
            )
            model.metric_optim = model.early_stop_metric

            if model.stop_training:
                model.early_stop_lstm_output_mu = model.lstm_output_history.mu.copy()
                model.early_stop_lstm_output_var = model.lstm_output_history.var.copy()
                break

        return model, local_training_metrics, local_lstm_std_per_epoch

    def _build_skf(trained_model, param):
        resolved_param = {**default_skf_param, **param}
        abnorm_components = [LocalAcceleration(), LstmNetwork(model_noise=use_tagiv)]
        if not use_tagiv:
            abnorm_components.append(WhiteNoise(std_error=resolved_param["sigma_v"]))
        abnorm_model = Model(*abnorm_components)
        skf = SKF(
            norm_model=trained_model,
            abnorm_model=abnorm_model,
            std_transition_error=resolved_param["std_transition_error"],
            norm_to_abnorm_prob=resolved_param["norm_to_abnorm_prob"],
            abnorm_to_norm_prob=resolved_param["abnorm_to_norm_prob"],
            likelihood_covariance_floor=resolved_param["likelihood_covariance_floor"],
        )
        if skf.model["norm_norm"].lstm_net.smooth is False:
            skf.model["norm_norm"].lstm_output_history.set(
                warmup_lookback_mu, warmup_lookback_var
            )
        skf.model["norm_norm"].lstm_net.teacher_forcing = False
        skf.save_initial_states()

        if skf_objective_function == "cdf":
            slope_value = float(resolved_param["slope"])
            synthetic_data = cdf_synthetic_cache.get(slope_value)
            if synthetic_data is None:
                slope_anomaly = slope_value / 52
                synthetic_data = DataProcess.add_synthetic_anomaly(
                    train_val,
                    num_samples=cdf_num_anomaly,
                    slope=[slope_anomaly, -slope_anomaly],
                    anomaly_start=cdf_anomaly_start,
                    anomaly_end=cdf_anomaly_end,
                )
                cdf_synthetic_cache[slope_value] = synthetic_data

            detection_rate, num_false_alarms, _ = skf.detect_synthetic_anomaly(
                data=train_val,
                threshold=resolved_param["threshold"],
                max_timestep_to_detect=max_timestep_to_detect,
                synthetic_data=synthetic_data,
                n_jobs=skf_tuning_n_jobs,
            )
            data_len_year = (
                _timestamp_at_exclusive_end(data_processor.validation_end)
                - data_processor.data.index[data_processor.train_start]
            ).days / 365.25
            false_rate_yearly = num_false_alarms / data_len_year
            metric_optim = skf.objective(
                detection_rate, false_rate_yearly, resolved_param["slope"]
            )
            skf.load_initial_states()
            skf.metric_optim = metric_optim.copy()
            skf.print_metric = {
                "detection_rate": detection_rate,
                "yearly_false_rate": false_rate_yearly,
            }

        else:
            skf.filter(data=all_data)
            log_lik_all = np.nanmean(skf.ll_history)
            skf.metric_optim = -log_lik_all
            skf.load_initial_states()

        return skf

    def model_with_parameters(param, skf_input):
        trained_model = Model.load_dict(skf_input["model_optim_dict"])
        return _build_skf(trained_model, param)

    ######### sigma_v grid search #########
    sigma_v_grid_search_result = None
    if experiment_config.get("optimize_sigma_v", False):
        n_jobs_grid = sigma_v_max_concurrent
        print(
            "---- sigma_v grid search over "
            f"{sigma_v_range} (metric: {lstm_early_stopping_metric_label}) ----"
        )

        if n_jobs_grid > 1:
            global _sigma_v_train_fn
            _sigma_v_train_fn = _train_lstm
            ctx = mp.get_context("fork")
            with ctx.Pool(n_jobs_grid) as pool:
                grid_results = pool.map(_sigma_v_worker, sigma_v_range)
        else:
            grid_results = []
            for sv_candidate in sigma_v_range:
                sv_model, sv_metrics, sv_std = _train_lstm(sv_candidate)
                best_epoch = max(0, min(int(sv_model.optimal_epoch), len(sv_metrics) - 1))
                grid_results.append(
                    {
                        "sigma_v": sv_candidate,
                        "validation_crps": float(sv_metrics[best_epoch]["validation_crps"]),
                        "validation_log_likelihood": float(sv_metrics[best_epoch]["validation_log_likelihood"]),
                        "optimal_epoch": int(sv_model.optimal_epoch),
                        "model_dict": sv_model.get_dict(time_step=0),
                        "metrics": sv_metrics,
                        "std_per_epoch": sv_std,
                    }
                )

        for r in grid_results:
            selected_metric = _selected_validation_metric(r["metrics"], r["optimal_epoch"])
            print(
                f"  sigma_v={r['sigma_v']:.4f} -> "
                f"best {lstm_early_stopping_metric_key}={selected_metric:.6f} "
                f"(epoch {r['optimal_epoch']})"
            )

        selection_delta = float(experiment_config.get("sigma_v_selection_delta", 0.0))
        scored = [
            (r, _selected_validation_metric(r["metrics"], r["optimal_epoch"]))
            for r in grid_results
        ]
        global_best_metric = (
            max(s for _, s in scored)
            if lstm_early_stopping_mode == "max"
            else min(s for _, s in scored)
        )
        if lstm_early_stopping_mode == "max":
            tolerated = [(r, s) for r, s in scored if s >= global_best_metric - selection_delta]
        else:
            tolerated = [(r, s) for r, s in scored if s <= global_best_metric + selection_delta]
        best_entry, best_validation_metric = min(tolerated, key=lambda rs: rs[0]["sigma_v"])
        best_sigma_v = best_entry["sigma_v"]

        sigma_v = best_sigma_v
        default_skf_param["sigma_v"] = sigma_v
        sigma_v_grid_search_result = {
            "selection_metric": lstm_early_stopping_metric,
            "selection_delta": selection_delta,
            "optimal_sigma_v": float(best_sigma_v),
            "best_validation_metric": float(best_validation_metric),
            "global_best_metric": float(global_best_metric),
            "grid_results": [
                {
                    **{
                        k: v
                        for k, v in r.items()
                        if k not in ("model_dict", "metrics", "std_per_epoch")
                    },
                    "selection_metric_value": _selected_validation_metric(
                        r["metrics"], r["optimal_epoch"]
                    ),
                }
                for r in grid_results
            ],
        }
        print(
            "---- Optimal sigma_v: "
            f"{sigma_v:.4f} "
            f"({lstm_early_stopping_metric_key}={best_validation_metric:.6f}) ----"
        )

        model_optim_dict = best_entry["model_dict"]
        lstm_training_metrics = best_entry["metrics"]
        lstm_training_std = best_entry["std_per_epoch"]
        optimal_epoch_idx = int(best_entry["optimal_epoch"])
    else:
        ######### Train LSTM once with config sigma_v #########
        print(f"---- Training LSTM once with sigma_v={sigma_v:.4f} ----")
        trained_lstm_model, lstm_training_metrics, lstm_training_std = _train_lstm(sigma_v)
        model_optim_dict = trained_lstm_model.get_dict(time_step=0)
        optimal_epoch_idx = int(trained_lstm_model.optimal_epoch)

    best_epoch = max(0, min(optimal_epoch_idx, len(lstm_training_metrics) - 1))
    optimal_validation_metrics.update(lstm_training_metrics[best_epoch])
    training_metrics_history.extend(lstm_training_metrics)
    lstm_std_per_epoch.extend(lstm_training_std)
    skf_input = {"model_optim_dict": model_optim_dict}

    ######### Parameter optimization #########
    if bool(experiment_config.get("optimize_skf_parameters", False)):
        num_optimization_trial = int(
            experiment_config.get("num_optimization_trial", 50)
        )
        num_startup_trials = int(experiment_config["num_startup_trials"])
        std_transition_error_range = experiment_config.get(
            "std_transition_error_search_space", [5e-6, 1e-4]
        )
        norm_to_abnorm_prob_range = experiment_config.get(
            "norm_to_abnorm_prob_search_space", [1e-5, 1e-4]
        )
        abnorm_to_norm_prob_range = experiment_config.get(
            "abnorm_to_norm_prob_search_space", [0.1, 0.2]
        )
        threshold_range = experiment_config.get("threshold_search_space", [0.05, 0.5])
        param_space = {

            "std_transition_error": tune.loguniform(
                float(std_transition_error_range[0]),
                float(std_transition_error_range[1]),
            ),
            "norm_to_abnorm_prob": tune.loguniform(
                float(norm_to_abnorm_prob_range[0]),
                float(norm_to_abnorm_prob_range[1]),
            ),
            "abnorm_to_norm_prob": tune.quniform(
                float(abnorm_to_norm_prob_range[0]),
                float(abnorm_to_norm_prob_range[1]),
                1e-2,
            ),
            "threshold":  tune.quniform(
                float(threshold_range[0]),
                float(threshold_range[1]),
                1e-2,
            )
        }
        if skf_objective_function == "cdf":
            param_space["slope"] = tune.choice(anomaly_magnitudes)

        # Define optimizer
        optimizer_mode = "max" if skf_objective_function == "cdf" else "min"
        model_optimizer = Optimizer(
            model=model_with_parameters,
            param=param_space,
            model_input=skf_input,
            num_optimization_trial=num_optimization_trial,
            mode=optimizer_mode,
            num_startup_trials=num_startup_trials,
            max_concurrent=1, # hard set
        )
        model_optimizer.optimize()
        # Get best model
        param = {**default_skf_param, **model_optimizer.get_best_param()}

    else:
        param = default_skf_param.copy()
    threshold = float(param["threshold"])
    skf_optim = model_with_parameters(param, skf_input)

    skf_optim_dict = skf_optim.get_dict()
    skf_optim_dict["model_param"] = param
    skf_optim_dict["cov_names"] = train_data["cov_names"]

    print("Model parameters used:", skf_optim_dict["model_param"])
    print(
        "Validation metrics at optimal epoch: "
        f"epoch={optimal_validation_metrics['epoch']} "
        f"val_ll={optimal_validation_metrics['validation_log_likelihood']:.6f} "
        f"val_crps={optimal_validation_metrics['validation_crps']:.6f} "
        f"val_rmse={optimal_validation_metrics['validation_rmse']:.6f}"
    )

    training_metrics_plot_path = output_dir / "training_metrics_by_epoch.pdf"
    if len(training_metrics_history) > 0:
        epochs = [m["epoch"] for m in training_metrics_history]
        val_ll = [m["validation_log_likelihood"] for m in training_metrics_history]
        val_rmse = [m["validation_rmse"] for m in training_metrics_history]
        best_epoch = int(optimal_validation_metrics["epoch"])

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
        fig_metrics.savefig(training_metrics_plot_path, format="pdf")
        plt.close(fig_metrics)

    lstm_std_plot_path = output_dir / "lstm_std_decay_by_epoch.pdf"
    if len(lstm_std_per_epoch) > 0:
        quantile_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
        quantile_matrix = np.full((len(quantile_levels), max_num_epoch), np.nan)
        for epoch_idx in range(max_num_epoch):
            epoch_vals = np.asarray(lstm_std_per_epoch[epoch_idx]).flatten()
            epoch_vals = epoch_vals[np.isfinite(epoch_vals)]
            if epoch_vals.size > 0:
                quantile_matrix[:, epoch_idx] = np.quantile(epoch_vals, quantile_levels)

        epochs = np.arange(1, max_num_epoch + 1)
        q05, q25, q50, q75, q95 = quantile_matrix

        fig_std, ax_std = plt.subplots(figsize=(10, 4))
        ax_std.fill_between(
            epochs, q05, q95, color="#93c5fd", alpha=0.35, label="q05-q95"
        )
        ax_std.fill_between(
            epochs, q25, q75, color="#2563eb", alpha=0.28, label="q25-q75"
        )
        ax_std.plot(epochs, q50, color="#1e3a8a", linewidth=2.2, label="median (q50)")
        ax_std.set_xlabel("Epoch")
        ax_std.set_ylabel("LSTM prior std")
        ax_std.set_title("LSTM Prior Std Decrease Across Epochs")
        ax_std.set_xlim(1, max_num_epoch)
        ax_std.set_ylim(0, 1)
        tick_step = max(1, max_num_epoch // 10)
        ax_std.set_xticks(np.arange(1, max_num_epoch + 1, tick_step))
        ax_std.grid(alpha=0.25)
        ax_std.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        fig_std.savefig(lstm_std_plot_path, format="pdf")
        plt.close(fig_std)

    ######### Multi-realization anomaly evaluation #########
    total_eval_steps = len(all_data["y"])
    test_start_ratio = data_processor.test_start / total_eval_steps
    anomaly_end_ratio  = (data_processor.test_end - max_timestep_to_detect) / total_eval_steps

    multi_eval_results = {}

    for mag in anomaly_magnitudes:
        mag_key = f"mag_{mag:.3f}"
        mag_plot_dir = output_dir / mag_key
        mag_plot_dir.mkdir(parents=True, exist_ok=True)

        mag_anomaly = mag / 52
        eval_synthetic_data = DataProcess.add_synthetic_anomaly(
                all_data,
                num_samples=num_realizations,
                slope=[mag_anomaly, -mag_anomaly],
                anomaly_start=test_start_ratio,
                anomaly_end=anomaly_end_ratio,
            )

        detection_rate, num_false_alarms, time_to_detection = (
            skf_optim.detect_synthetic_anomaly(
                data=all_data,
                threshold=threshold,
                max_timestep_to_detect=max_timestep_to_detect,
                plot_dir=mag_plot_dir,
                synthetic_data=eval_synthetic_data,
                n_jobs=skf_eval_n_jobs,
            )
        )

        data_len_year = (
            _timestamp_at_exclusive_end(data_processor.test_end)
            - data_processor.data.index[data_processor.train_start]
        ).days / 365.25
        false_rate_yearly = num_false_alarms / data_len_year
        ttd_mean = time_to_detection[0] / 52
        ttd_std = time_to_detection[1] / 52

        multi_eval_results[mag_key] = {
            "probability_of_detection": detection_rate,
            "false_alarm_rate_per_y": false_rate_yearly,
            "time_to_detection_years_mean": ttd_mean,
            "time_to_detection_years_std": ttd_std,
            "num_realizations": num_realizations,
            "plot_directory": str(mag_plot_dir),
        }
        print(
            f"  mag={mag:.4f}:  P(detect)={detection_rate:.2f}  "
            f"FA/yr={false_rate_yearly:.2f}  "
            f"TTD(yr)={ttd_mean:.3f}\u00b1{ttd_std:.3f}"
        )

    print(f"{'='*70}\n")

    # Save summary
    summary = {
        "experiment_name": experiment_name,
        "model_parameters_used": skf_optim_dict["model_param"],
        "optimal_validation_metrics": optimal_validation_metrics,
        "multi_realization_evaluation": multi_eval_results,
        "training_metrics_plot": str(training_metrics_plot_path),
        "lstm_std_distribution_plot": str(lstm_std_plot_path),
    }
    if sigma_v_grid_search_result is not None:
        summary["sigma_v_grid_search"] = sigma_v_grid_search_result
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved outputs to: {output_dir}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    fire.Fire(main)
