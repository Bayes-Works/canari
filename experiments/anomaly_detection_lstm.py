import fire
import copy
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytagi import metric
from pytagi import Normalizer as normalizer
from canari import (
    DataProcess,
    Model,
    Optimizer,
    SKF,
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise
from experiments.utils import prepare_dataset

from pathlib import Path
import yaml


def main(
    experiment_config_path: str = "./experiments/config/anomaly_detection_global_lstm_dummy.yaml",
):

    # Read config file
    experiment_config_path = Path(experiment_config_path)
    with experiment_config_path.open("r") as f:
        experiment_config = yaml.safe_load(f)

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

    def model_with_parameters(param):
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
            ),
            WhiteNoise(std_error=sigma_v),
        )

        model.auto_initialize_baseline_states(train_data["y"][0:24])
        num_epoch = 50
        for epoch in range(num_epoch):
            mu_validation_preds, std_validation_preds, states = model.lstm_train(
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

            model.early_stopping(
                evaluate_metric=-validation_log_lik,
                current_epoch=epoch,
                max_epoch=num_epoch,
            )
            model.metric_optim = model.early_stop_metric

            if model.stop_training:
                break

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
            "std_transition_error": experiment_config['std_transition_error'],
            "norm_to_abnorm_prob": experiment_config['norm_to_abnorm_prob'],
        }
    skf_optim = model_with_parameters(param)

    skf_optim_dict = skf_optim.get_dict()
    skf_optim_dict["model_param"] = param
    skf_optim_dict["cov_names"] = train_data["cov_names"]

    ######### Detect anomaly #########
    print("Model parameters used:", skf_optim_dict["model_param"])

    filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)
    smooth_marginal_abnorm_prob, states = skf_optim.smoother()

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
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
