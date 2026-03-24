import fire
import numpy as np
import matplotlib.pyplot as plt
import csv
from pytagi import metric
from pytagi import Normalizer as normalizer
from canari import (
    Model,
    Optimizer,
    SKF,
    plot_skf_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise

try:
    from experiments.utils import prepare_dataset
except ModuleNotFoundError:
    from utils import prepare_dataset

from pathlib import Path
import yaml
import json

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

TRANSITION_KEYS = ("norm_norm", "norm_abnorm", "abnorm_norm", "abnorm_abnorm")
TRANSITION_LABELS = {
    "norm_norm": "normal -> normal",
    "norm_abnorm": "normal -> abnormal",
    "abnorm_norm": "abnormal -> normal",
    "abnorm_abnorm": "abnormal -> abnormal",
}
TRANSITION_COLORS = {
    "norm_norm": "#2563eb",
    "norm_abnorm": "#dc2626",
    "abnorm_norm": "#d97706",
    "abnorm_abnorm": "#16a34a",
}


def _history_array(history_dict, key: str, num_steps: int) -> np.ndarray:
    values = np.asarray(history_dict.get(key, []), dtype=float).reshape(-1)
    if values.size >= num_steps:
        return values[:num_steps]
    padded = np.full(num_steps, np.nan, dtype=float)
    if values.size > 0:
        padded[: values.size] = values
    return padded


def _plot_transition_diagnostics(
    skf: SKF,
    all_data: dict,
    anomaly_idx: int,
    output_dir: Path,
    show_anomaly_marker: bool = True,
) -> Path:
    obs = np.asarray(all_data["y"]).reshape(-1)
    num_steps = obs.size
    time_axis = np.asarray(all_data["time"])
    if time_axis.size < num_steps:
        time_axis = np.arange(num_steps)
    else:
        time_axis = time_axis[:num_steps]
    anomaly_idx = int(np.clip(anomaly_idx, 0, max(num_steps - 1, 0)))

    # Panel 1: switch evidence from normal-origin transitions.
    log_lik_norm_norm = _history_array(
        skf.transition_log_likelihood_history, "norm_norm", num_steps
    )
    log_lik_norm_abnorm = _history_array(
        skf.transition_log_likelihood_history, "norm_abnorm", num_steps
    )
    delta_log_lik = log_lik_norm_abnorm - log_lik_norm_norm

    # Panel 2: calibration view for normal branch.
    mu_norm_norm = _history_array(skf.transition_mu_pred_history, "norm_norm", num_steps)
    var_norm_norm = np.maximum(
        _history_array(skf.transition_var_pred_history, "norm_norm", num_steps), 0.0
    )
    sigma_norm_norm = np.sqrt(var_norm_norm)
    abs_residual_norm_norm = np.abs(obs - mu_norm_norm)
    abs_z_norm_norm = np.abs(
        _history_array(skf.transition_zscore_history, "norm_norm", num_steps)
    )

    # Panel 3: marginal regime probability.
    marginal_abnorm = np.asarray(
        skf.filter_marginal_prob_history.get("abnorm", []), dtype=float
    ).reshape(-1)
    if marginal_abnorm.size < num_steps:
        marginal_abnorm = np.pad(
            marginal_abnorm,
            (0, num_steps - marginal_abnorm.size),
            mode="constant",
            constant_values=np.nan,
        )
    else:
        marginal_abnorm = marginal_abnorm[:num_steps]

    # Panel 4: transition posterior composition.
    posterior_prob_by_transition = []
    for transit in TRANSITION_KEYS:
        posterior_prob_by_transition.append(
            _history_array(skf.transition_posterior_prob_history, transit, num_steps)
        )
    posterior_prob_by_transition = np.asarray(posterior_prob_by_transition, dtype=float)
    posterior_prob_by_transition = np.nan_to_num(posterior_prob_by_transition, nan=0.0)
    col_sum = np.sum(posterior_prob_by_transition, axis=0, keepdims=True)
    valid_col = col_sum > 0.0
    posterior_prob_by_transition[:, valid_col[0]] = (
        posterior_prob_by_transition[:, valid_col[0]] / col_sum[:, valid_col[0]]
    )

    fig_diag, ax_diag = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    ax_diag[0].plot(
        time_axis,
        delta_log_lik,
        color="#7c3aed",
        linewidth=1.6,
        label=r"$\Delta \log L$: norm$\to$abnorm - norm$\to$norm",
    )
    ax_diag[0].axhline(0.0, color="k", linestyle=":", linewidth=1.0)
    ax_diag[0].set_ylabel(r"$\Delta \log L$")
    ax_diag[0].set_title("SKF Switch Diagnostics")
    ax_diag[0].grid(alpha=0.25)
    ax_diag[0].legend(loc="best")

    ax_diag[1].plot(
        time_axis,
        abs_residual_norm_norm,
        color="#b91c1c",
        linewidth=1.5,
        label=r"$|y-\mu|$ (norm$\to$norm)",
    )
    ax_diag[1].plot(
        time_axis,
        sigma_norm_norm,
        color="#1d4ed8",
        linewidth=1.5,
        label=r"$\sigma=\sqrt{var}$ (norm$\to$norm)",
    )
    ax_diag[1].set_ylabel(r"$|residual|,\ \sigma$")
    ax_diag[1].grid(alpha=0.25)
    ax_diag[1].legend(loc="upper left")

    ax_z = ax_diag[1].twinx()
    ax_z.plot(
        time_axis,
        abs_z_norm_norm,
        color="#0f766e",
        linewidth=1.2,
        alpha=0.9,
        label=r"$|z|$ (norm$\to$norm)",
    )
    ax_z.axhline(2.0, color="#0f766e", linestyle="--", linewidth=0.9, alpha=0.75)
    ax_z.set_ylabel(r"$|z|$")
    ax_z.legend(loc="upper right")

    ax_diag[2].plot(
        time_axis,
        marginal_abnorm,
        color="#dc2626",
        linewidth=1.8,
        label="P(abnormal)",
    )
    ax_diag[2].set_ylabel("Marginal prob")
    ax_diag[2].set_ylim(-0.02, 1.02)
    ax_diag[2].grid(alpha=0.25)
    ax_diag[2].legend(loc="best")

    ax_diag[3].stackplot(
        time_axis,
        posterior_prob_by_transition,
        labels=[TRANSITION_LABELS[key] for key in TRANSITION_KEYS],
        colors=[TRANSITION_COLORS[key] for key in TRANSITION_KEYS],
        alpha=0.85,
    )
    ax_diag[3].set_ylabel("Transition mix")
    ax_diag[3].set_ylim(0.0, 1.0)
    ax_diag[3].set_xlabel("Time")
    ax_diag[3].grid(alpha=0.2)
    ax_diag[3].legend(loc="upper right", ncol=2, frameon=True)

    if show_anomaly_marker:
        anomaly_time = time_axis[anomaly_idx]
        for axis in ax_diag:
            axis.axvline(anomaly_time, color="k", linestyle="--", linewidth=1.1)
        ax_z.axvline(anomaly_time, color="k", linestyle="--", linewidth=1.1)

    plt.tight_layout()
    diag_plot_path = output_dir / "skf_transition_likelihoods.pdf"
    fig_diag.savefig(diag_plot_path, format="pdf")
    plt.close(fig_diag)
    return diag_plot_path


def _save_transition_diagnostics_csv(
    skf: SKF,
    all_data: dict,
    output_dir: Path,
) -> Path:
    obs = np.asarray(all_data["y"]).reshape(-1)
    num_steps = obs.size
    time_axis = np.asarray(all_data["time"])
    if time_axis.size < num_steps:
        time_axis = np.arange(num_steps)
    else:
        time_axis = time_axis[:num_steps]

    marginal_norm = np.asarray(
        skf.filter_marginal_prob_history.get("norm", []), dtype=float
    ).reshape(-1)
    marginal_abnorm = np.asarray(
        skf.filter_marginal_prob_history.get("abnorm", []), dtype=float
    ).reshape(-1)
    if marginal_norm.size < num_steps:
        marginal_norm = np.pad(
            marginal_norm,
            (0, num_steps - marginal_norm.size),
            mode="constant",
            constant_values=np.nan,
        )
    if marginal_abnorm.size < num_steps:
        marginal_abnorm = np.pad(
            marginal_abnorm,
            (0, num_steps - marginal_abnorm.size),
            mode="constant",
            constant_values=np.nan,
        )

    diag_csv_path = output_dir / "skf_transition_diagnostics.csv"
    fieldnames = ["time", "obs", "marginal_prob_norm", "marginal_prob_abnorm"]
    diagnostics_by_transit = {}
    for transit in TRANSITION_KEYS:
        fieldnames.extend(
            [
                f"{transit}_mu_pred",
                f"{transit}_var_pred",
                f"{transit}_zscore",
                f"{transit}_likelihood",
                f"{transit}_log_likelihood",
                f"{transit}_posterior_prob",
            ]
        )
        diagnostics_by_transit[transit] = {
            "mu_pred": _history_array(skf.transition_mu_pred_history, transit, num_steps),
            "var_pred": _history_array(
                skf.transition_var_pred_history, transit, num_steps
            ),
            "zscore": _history_array(skf.transition_zscore_history, transit, num_steps),
            "likelihood": _history_array(
                skf.transition_likelihood_history, transit, num_steps
            ),
            "log_likelihood": _history_array(
                skf.transition_log_likelihood_history, transit, num_steps
            ),
            "posterior_prob": _history_array(
                skf.transition_posterior_prob_history, transit, num_steps
            ),
        }

    with diag_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(num_steps):
            row = {
                "time": str(time_axis[idx]),
                "obs": float(obs[idx]),
                "marginal_prob_norm": float(marginal_norm[idx]),
                "marginal_prob_abnorm": float(marginal_abnorm[idx]),
            }
            for transit in TRANSITION_KEYS:
                transit_diag = diagnostics_by_transit[transit]
                row[f"{transit}_mu_pred"] = float(transit_diag["mu_pred"][idx])
                row[f"{transit}_var_pred"] = float(transit_diag["var_pred"][idx])
                row[f"{transit}_zscore"] = float(transit_diag["zscore"][idx])
                row[f"{transit}_likelihood"] = float(transit_diag["likelihood"][idx])
                row[f"{transit}_log_likelihood"] = float(
                    transit_diag["log_likelihood"][idx]
                )
                row[f"{transit}_posterior_prob"] = float(
                    transit_diag["posterior_prob"][idx]
                )
            writer.writerow(row)

    return diag_csv_path


def _plot_lstm_embedding(
    mu_embedding: np.ndarray,
    var_embedding: np.ndarray,
    output_dir: Path,
) -> Path:
    mu_embedding = np.asarray(mu_embedding, dtype=float).reshape(-1)
    var_embedding = np.asarray(var_embedding, dtype=float).reshape(-1)
    if mu_embedding.size == 0:
        raise ValueError("LSTM embedding is empty.")
    if mu_embedding.size != var_embedding.size:
        raise ValueError("LSTM embedding mean/variance must have the same length.")

    var_embedding = np.maximum(var_embedding, 0.0)
    std_embedding = np.sqrt(var_embedding)
    embed_idx = np.arange(mu_embedding.size)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(embed_idx, mu_embedding, color="#1d4ed8", linewidth=2, label=r"$\mu$")
    ax.fill_between(
        embed_idx,
        mu_embedding - std_embedding,
        mu_embedding + std_embedding,
        color="#93c5fd",
        alpha=0.35,
        label=r"$\mu \pm \sqrt{var}$",
    )
    ax.set_xlabel("Embedding index")
    ax.set_ylabel("Value")
    ax.set_title("Learned LSTM Embedding")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()

    embedding_plot_path = output_dir / "lstm_embedding.pdf"
    fig.savefig(embedding_plot_path, format="pdf")
    plt.close(fig)
    return embedding_plot_path


def main(
    experiment_config_path: str = "./experiments/config/LGA010ESAPRG988.yaml",
):

    # Read config file
    experiment_config_path = Path(experiment_config_path)
    with experiment_config_path.open("r") as f:
        experiment_config = yaml.safe_load(f)
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
        anomaly_slope=float(experiment_config["anomaly_slope"]),
        experiment_config=experiment_config,
    )
    show_anomaly_marker = not np.isclose(
        float(experiment_config.get("anomaly_slope", 0.0)),
        0.0,
    )
    train_data = dataset["train_data"]
    validation_data = dataset["validation_data"]
    data_processor = dataset["data_processor"]
    all_data = dataset["all_data"]
    time_anomaly = dataset["anomaly_time"]
    warmup_lookback_mu = dataset["warmup_lookback_mu"]
    warmup_lookback_var = dataset["warmup_lookback_var"]

    # Define model with parameters
    look_back_len = experiment_config["lstm_look_back_len"]
    num_features = experiment_config["lstm_num_features"]
    num_layer = experiment_config["lstm_num_layer"]
    infer_len = experiment_config["lstm_infer_len"]
    num_hidden_unit = experiment_config["num_hidden_unit"]
    seed = experiment_config["lstm_manual_seed"]
    smoother = experiment_config["smoother"]
    sigma_v = experiment_config["sigma_v"]
    embed_len = int(
        experiment_config.get(
            "embed_len",
            experiment_config.get("lstm_embed_len", 0),
        )
    )
    stateless = experiment_config["lstm_stateless"]
    zero_shot = experiment_config["lstm_zeroshot"]
    finetune = experiment_config["lstm_finetune"]
    global_params = experiment_config["lstm_global_params"]
    use_tagiv = experiment_config["use_tagiv"]
    max_num_epoch = int(experiment_config.get("lstm_num_epoch", 50))
    update_embedding = bool(
        experiment_config.get("lstm_update_embedding", embed_len > 0)
    )
    likelihood_covariance_floor = float(
        experiment_config.get("likelihood_covariance_floor", 0.1)
    )
    default_skf_param = {
        "sigma_v": sigma_v,
        "std_transition_error": float(experiment_config["std_transition_error"]),
        "norm_to_abnorm_prob": float(experiment_config["norm_to_abnorm_prob"]),
    }
    optimal_validation_metrics = {}
    training_metrics_history = []
    lstm_std_per_epoch = []

    def model_with_parameters(param, capture_training_metrics: bool = False):
        resolved_param = {**default_skf_param, **param}
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
            embed_len=embed_len,
            finetune=finetune,
            load_lstm_net=global_params,
            model_noise=use_tagiv,
            zeroshot=zero_shot,
        )
        try:
            model = Model(
                LocalTrend(),
                LstmNetwork(**lstm_kwargs),
                # WhiteNoise(std_error=sigma_v),
            )
        except RuntimeError as exc:
            if global_params and "Failed to load LSTM network from" in str(exc):
                print(
                    "Warning: incompatible pretrained LSTM weights at "
                    f"'{global_params}'. Falling back to random initialization."
                )
                lstm_kwargs["load_lstm_net"] = None
                lstm_kwargs["finetune"] = False
                model = Model(
                    LocalTrend(),
                    LstmNetwork(**lstm_kwargs),
                    # WhiteNoise(std_error=sigma_v),
                )
            else:
                raise

        # model.auto_initialize_baseline_states(
        #     train_data["y"][0 : experiment_config["baseline_init_len"]]
        # )
        num_epoch = max_num_epoch
        local_training_metrics = []
        local_lstm_std_per_epoch = [np.array([np.nan]) for _ in range(num_epoch)]
        for epoch in range(num_epoch):
            model.lstm_output_history.set(warmup_lookback_mu, warmup_lookback_var)
            
            mu_validation_preds, std_validation_preds, train_states = model.lstm_train(
                train_data=train_data,
                validation_data=validation_data,
                white_noise_max_std=1.0,
                update_embedding=update_embedding,
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

            validation_obs = data_processor.get_data("validation").flatten()
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
            std_lstm_prior = np.asarray(train_states.get_std("lstm", "prior")).flatten()
            std_lstm_prior = std_lstm_prior[np.isfinite(std_lstm_prior)]
            if std_lstm_prior.size == 0:
                std_lstm_prior = np.array([np.nan])
            local_lstm_std_per_epoch[epoch] = std_lstm_prior

            model.early_stopping(
                evaluate_metric=-validation_log_lik,
                current_epoch=epoch,
                max_epoch=num_epoch,
                skip_epoch=0,
            )
            model.metric_optim = model.early_stop_metric

            if model.stop_training:
                break

        if capture_training_metrics:
            best_epoch = int(model.optimal_epoch)
            best_epoch = max(0, min(best_epoch, len(local_training_metrics) - 1))
            optimal_validation_metrics.clear()
            optimal_validation_metrics.update(local_training_metrics[best_epoch])
            training_metrics_history.clear()
            training_metrics_history.extend(local_training_metrics)
            lstm_std_per_epoch.clear()
            lstm_std_per_epoch.extend(local_lstm_std_per_epoch)

        #### Define SKF model with parameters #########
        abnorm_model = Model(
            LocalAcceleration(),
            LstmNetwork(model_noise=use_tagiv),
            # WhiteNoise(),
        )
        skf = SKF(
            norm_model=model,
            abnorm_model=abnorm_model,
            std_transition_error=resolved_param["std_transition_error"],
            norm_to_abnorm_prob=resolved_param["norm_to_abnorm_prob"],
            likelihood_covariance_floor=likelihood_covariance_floor,
        )
        # if skf.model["norm_norm"].lstm_net.smooth is False:
        skf.model["norm_norm"].lstm_output_history.set(
            warmup_lookback_mu, warmup_lookback_var
        )
        skf.save_initial_states()

        skf.filter(data=all_data)
        log_lik_all = np.nanmean(skf.ll_history)
        skf.metric_optim = -log_lik_all

        skf.load_initial_states()

        return skf

    ######### Parameter optimization #########
    if bool(experiment_config.get("optimize_skf_parameters", False)):
        num_optimization_trial = int(experiment_config.get("num_optimization_trial", 50))
        num_startup_trials = int(experiment_config["num_startup_trials"])
        param_space = {
            # "sigma_v": [1e-3, 2e-1],
            "std_transition_error": [1e-7, 1e-5],
            # "norm_to_abnorm_prob": [1e-6, 1e-4],
        }
        # Define optimizer
        model_optimizer = Optimizer(
            model=model_with_parameters,
            param=param_space,
            num_optimization_trial=num_optimization_trial,
            mode="min",
            num_startup_trials=num_startup_trials,
        )
        model_optimizer.optimize()
        # Get best model
        param = {**default_skf_param, **model_optimizer.get_best_param()}

    else:
        param = default_skf_param.copy()
    param["likelihood_covariance_floor"] = likelihood_covariance_floor
    skf_optim = model_with_parameters(param, capture_training_metrics=True)

    skf_optim_dict = skf_optim.get_dict()
    skf_optim_dict["model_param"] = param
    skf_optim_dict["cov_names"] = train_data["cov_names"]

    ######### Detect anomaly #########
    print("Model parameters used:", skf_optim_dict["model_param"])
    print(
        "Validation metrics at optimal epoch: "
        f"epoch={optimal_validation_metrics['epoch']} "
        f"val_ll={optimal_validation_metrics['validation_log_likelihood']:.6f} "
        f"val_rmse={optimal_validation_metrics['validation_rmse']:.6f}"
    )

    anomaly_idx = int(dataset["metadata"]["time_anomaly"])
    filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)
    _, states = skf_optim.smoother()

    skf_transition_plot_path = _plot_transition_diagnostics(
        skf=skf_optim,
        all_data=all_data,
        anomaly_idx=anomaly_idx,
        output_dir=output_dir,
        show_anomaly_marker=show_anomaly_marker,
    )
    skf_transition_csv_path = _save_transition_diagnostics_csv(
        skf=skf_optim,
        all_data=all_data,
        output_dir=output_dir,
    )

    fig, ax = plot_skf_states(
        data_processor=dataset["data_processor"],
        states=states,
        model_prob=filter_marginal_abnorm_prob,
        standardization=True,
        states_type="posterior",
    )
    anomaly_time = dataset["anomaly_time"]
    if show_anomaly_marker:
        ax[-1].axvline(
            x=anomaly_time,
            color="k",
            linestyle="--",
            linewidth=1.2,
            label="Anomaly",
        )
        ax[-1].legend(loc="upper right")
    plt.tight_layout()
    skf_pdf_path = output_dir / "skf_states.pdf"
    fig.savefig(skf_pdf_path, format="pdf")
    plt.close(fig)

    training_metrics_plot_path = output_dir / "training_metrics_by_epoch.pdf"
    if len(training_metrics_history) > 0:
        epochs = [m["epoch"] for m in training_metrics_history]
        val_ll = [m["validation_log_likelihood"] for m in training_metrics_history]
        val_rmse = [m["validation_rmse"] for m in training_metrics_history]
        best_epoch = int(optimal_validation_metrics["epoch"])

        fig_metrics, axes_metrics = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True
        )
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

    lstm_embedding_plot_path = None
    norm_model_embedding = skf_optim.model["norm_norm"].lstm_embedding
    if getattr(norm_model_embedding, "length", 0) > 0:
        lstm_embedding_plot_path = _plot_lstm_embedding(
            norm_model_embedding.mu,
            norm_model_embedding.var,
            output_dir=output_dir,
        )

    threshold = float(experiment_config.get("anomaly_detection_threshold", 0.5))
    pre_anomaly_probs = filter_marginal_abnorm_prob[:anomaly_idx]
    false_alarm_count = int(np.sum(pre_anomaly_probs > threshold))
    false_alarm_rate = (
        float(false_alarm_count / len(pre_anomaly_probs))
        if len(pre_anomaly_probs) > 0
        else 0.0
    )
    print(
        f"False alarm rate before anomaly: {false_alarm_rate:.6f} "
        f"({false_alarm_count}/{len(pre_anomaly_probs)}) at threshold={threshold}"
    )

    detection_summary = {
        "threshold": threshold,
        "anomaly_time": str(time_anomaly),
        "false_alarm_rate_before_anomaly": false_alarm_rate,
        "false_alarm_count_before_anomaly": false_alarm_count,
        "num_points_before_anomaly": int(len(pre_anomaly_probs)),
    }
    detected_rel = np.where(filter_marginal_abnorm_prob[anomaly_idx:] > threshold)[0]
    if detected_rel.size > 0:
        detected_idx = int(anomaly_idx + detected_rel[0])
        detected_time = all_data["time"][detected_idx]
        delay_steps = detected_idx - anomaly_idx
        delay_time = detected_time - time_anomaly
        detection_summary.update(
            {
                "detected": True,
                "detected_time": str(detected_time),
                "delay_steps": int(delay_steps),
                "delay": str(delay_time),
            }
        )
        print(
            "Final SKF anomaly detection: "
            f"anomaly_time={time_anomaly} detected_time={detected_time} "
            f"delay_steps={delay_steps} delay={delay_time}"
        )
    else:
        detection_summary.update(
            {
                "detected": False,
                "detected_time": None,
                "delay_steps": None,
                "delay": None,
            }
        )
        print(
            "Final SKF anomaly detection: "
            f"no detection after anomaly_time={time_anomaly} at threshold={threshold}"
        )

    summary = {
        "experiment_name": experiment_name,
        "model_parameters_used": skf_optim_dict["model_param"],
        "optimal_validation_metrics": optimal_validation_metrics,
        "final_skf_detection": detection_summary,
        "skf_figure_pdf": str(skf_pdf_path),
        "skf_transition_likelihood_plot": str(skf_transition_plot_path),
        "skf_transition_diagnostics_csv": str(skf_transition_csv_path),
        "training_metrics_plot": str(training_metrics_plot_path),
        "lstm_std_distribution_plot": str(lstm_std_plot_path),
        "lstm_embedding_plot": (
            str(lstm_embedding_plot_path) if lstm_embedding_plot_path else None
        ),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved outputs to: {output_dir}")
    print(f"Config YAML: {used_config_path}")
    print(f"SKF figure: {skf_pdf_path}")
    print(f"SKF transition likelihood plot: {skf_transition_plot_path}")
    print(f"SKF transition diagnostics CSV: {skf_transition_csv_path}")
    print(f"Training metrics plot: {training_metrics_plot_path}")
    print(f"LSTM std distribution plot: {lstm_std_plot_path}")
    print(f"LSTM embedding plot: {lstm_embedding_plot_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    fire.Fire(main)
