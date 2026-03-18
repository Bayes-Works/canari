import fire
import numpy as np
import matplotlib.pyplot as plt
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


def main(
    experiment_config_path: str = "./experiments/config/anomaly_detection_global_lstm_dummy.yaml",
):

    # Read config file
    experiment_config_path = Path(experiment_config_path)
    with experiment_config_path.open("r") as f:
        experiment_config = yaml.safe_load(f)
    experiment_name = experiment_config["experiment_name"]
    output_root = Path(experiment_config.get("output_root", "experiments/out"))
    output_dir = output_root / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data preperation
    dataset = prepare_dataset(
        train_split=float(experiment_config["train_split"]),
        anomaly_slope=float(experiment_config["anomaly_slope"]),
        experiment_config=experiment_config,
    )
    train_data = dataset["train_data"]
    validation_data = dataset["validation_data"]
    data_processor = dataset["data_processor"]
    all_data = dataset["all_data"]
    time_anomaly = dataset["anomaly_time"]

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
    finetune = experiment_config["lstm_finetune"]
    global_params = experiment_config["lstm_global_params"]
    optimal_validation_metrics = {}

    def model_with_parameters(param, capture_training_metrics: bool = False):
        model = Model(
            LocalTrend(),
            LstmNetwork(
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
                load_lstm_net=global_params,
            ),
            WhiteNoise(std_error=sigma_v),
        )

        model.auto_initialize_baseline_states(train_data["y"][0:24])
        num_epoch = 50
        local_training_metrics = []
        for epoch in range(num_epoch):
            mu_validation_preds, std_validation_preds, _ = model.lstm_train(
                train_data=train_data,
                validation_data=validation_data,
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

            model.early_stopping(
                evaluate_metric=-validation_log_lik,
                current_epoch=epoch,
                max_epoch=num_epoch,
            )
            model.metric_optim = model.early_stop_metric

            if model.stop_training:
                break
        if capture_training_metrics:
            best_epoch = int(model.optimal_epoch)
            best_epoch = max(0, min(best_epoch, len(local_training_metrics) - 1))
            optimal_validation_metrics.clear()
            optimal_validation_metrics.update(local_training_metrics[best_epoch])

        #### Define SKF model with parameters #########
        abnorm_model = Model(
            LocalAcceleration(),
            LstmNetwork(),
            WhiteNoise(),
        )
        skf = SKF(
            norm_model=model,
            abnorm_model=abnorm_model,
            std_transition_error=param["std_transition_error"],
            norm_to_abnorm_prob=param["norm_to_abnorm_prob"],
        )
        skf.save_initial_states()

        skf.filter(data=all_data)
        log_lik_all = np.nanmean(skf.ll_history)
        skf.metric_optim = -log_lik_all

        skf.load_initial_states()

        return skf

    ######### Parameter optimization #########
    if experiment_config["optimize_skf_parameters"]:
        param_space = {
            "sigma_v": [1e-3, 2e-1],
            "std_transition_error": [1e-6, 1e-4],
            "norm_to_abnorm_prob": [1e-6, 1e-4],
        }
        # Define optimizer
        model_optimizer = Optimizer(
            model=model_with_parameters,
            param=param_space,
            num_optimization_trial=experiment_config["num_optimization_trial"],
            mode="min",
            num_startup_trials=experiment_config["num_startup_trial"],
        )
        model_optimizer.optimize()
        # Get best model
        param = model_optimizer.get_best_param()

    else:
        param = {
            "sigma_v": sigma_v,
            "std_transition_error": experiment_config["std_transition_error"],
            "norm_to_abnorm_prob": experiment_config["norm_to_abnorm_prob"],
        }
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

    filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)
    _, states = skf_optim.smoother()

    fig, ax = plot_skf_states(
        data_processor=dataset["data_processor"],
        states=states,
        model_prob=filter_marginal_abnorm_prob,
        standardization=True,
        states_type="prior",
    )
    anomaly_time = dataset["anomaly_time"]
    ax[-1].axvline(
        x=anomaly_time,
        color="k",
        linestyle="--",
        linewidth=1.2,
        label="Anomaly",
    )
    ax[-1].legend(loc="upper right")
    plt.tight_layout()
    skf_svg_path = output_dir / "skf_states.svg"
    fig.savefig(skf_svg_path, format="svg")
    plt.close(fig)

    threshold = float(experiment_config.get("anomaly_detection_threshold", 0.5))
    anomaly_idx = int(dataset["metadata"]["time_anomaly"])
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
        "skf_figure_svg": str(skf_svg_path),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved outputs to: {output_dir}")
    print(f"SKF figure: {skf_svg_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    fire.Fire(main)
