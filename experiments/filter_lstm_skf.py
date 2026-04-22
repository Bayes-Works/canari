from pathlib import Path

import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import norm as _norm_dist
from tqdm import tqdm

from pytagi import metric
from pytagi import Normalizer as normalizer
from canari import Model, SKF, plot_skf_states
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise

try:
    from experiments.utils import prepare_dataset
except ModuleNotFoundError:
    from utils import prepare_dataset


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


def main(
    experiment_config_path: str = "./experiments/config/OOD_timeseries/original/test_5.yaml",
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
    elif lstm_early_stopping_metric == "crps":
        lstm_early_stopping_metric_key = "validation_crps"
        lstm_early_stopping_mode = "min"
    else:
        raise ValueError(
            "lstm_early_stopping_metric must be either 'll' or 'crps'."
        )
    likelihood_covariance_floor = float(
        experiment_config.get("likelihood_covariance_floor", 0.0)
    )
    abnorm_to_norm_prob = float(experiment_config.get("abnorm_to_norm_prob", 0.1))

    default_skf_param = {
        "sigma_v": sigma_v,
        "std_transition_error": float(experiment_config["std_transition_error"]),
        "norm_to_abnorm_prob": float(experiment_config["norm_to_abnorm_prob"]),
        "abnorm_to_norm_prob": abnorm_to_norm_prob,
        "likelihood_covariance_floor": likelihood_covariance_floor,
        "threshold": float(experiment_config.get("anomaly_detection_threshold", 0.4)),
    }


    validation_obs = data_processor.get_data("validation").flatten()


    def _train_lstm(sv_value):
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

        # model.mu_states[model.get_states_index("trend")] = 0.0  # force trend to

        model.lstm_net.teacher_forcing = False

        num_epoch = max_num_epoch
        local_training_metrics = []
        pbar = tqdm(range(num_epoch), desc=f"LSTM sigma_v={sv_value:.4f}")
        for epoch in pbar:
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
            pbar.set_postfix(
                ll=f"{validation_log_lik:.3f}",
                crps=f"{validation_crps:.3f}",
                rmse=f"{validation_rmse:.3f}",
            )

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
        pbar.close()

        return model, local_training_metrics

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

        return skf

    print(f"Training LSTM with sigma_v={sigma_v:.4f}")
    trained_lstm_model, lstm_training_metrics = _train_lstm(sigma_v)
    best_epoch = max(
        0, min(int(trained_lstm_model.optimal_epoch), len(lstm_training_metrics) - 1)
    )
    optimal_validation_metrics = lstm_training_metrics[best_epoch]

    param = default_skf_param.copy()
    trained_model_clone = Model.load_dict(trained_lstm_model.get_dict(time_step=0))
    skf_optim = _build_skf(trained_model_clone, param)

    print("Model parameters used:", param)
    print(
        "Validation metrics at optimal epoch: "
        f"epoch={optimal_validation_metrics['epoch']} "
        f"val_ll={optimal_validation_metrics['validation_log_likelihood']:.6f} "
        f"val_crps={optimal_validation_metrics['validation_crps']:.6f} "
        f"val_rmse={optimal_validation_metrics['validation_rmse']:.6f}"
    )

    training_metrics_plot_path = output_dir / "training_metrics_by_epoch.pdf"
    if len(lstm_training_metrics) > 0:
        epochs = [m["epoch"] for m in lstm_training_metrics]
        val_ll = [m["validation_log_likelihood"] for m in lstm_training_metrics]
        val_rmse = [m["validation_rmse"] for m in lstm_training_metrics]

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

    filter_marginal_abnorm_prob, skf_states = skf_optim.filter(data=all_data)
    fig_states, _ = plot_skf_states(
        data_processor=data_processor,
        states=skf_states,
        model_prob=filter_marginal_abnorm_prob,
        standardization=False,
        states_type="posterior",
    )
    plt.tight_layout()
    states_plot_path = output_dir / "skf_states_posterior"
    formats = ["pdf", "svg", "pgf"]
    for ext in formats:
        fig_states.savefig(states_plot_path.with_suffix(f".{ext}"), format=ext)
    plt.close(fig_states)



if __name__ == "__main__":
    fire.Fire(main)
