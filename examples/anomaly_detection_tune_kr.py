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
from canari.component import LocalTrend, LocalAcceleration, WhiteNoise, KernelRegression


def main(
    num_trial_optimization: int = 20,
    param_optimization: bool = True,
):
    ######### Data processing #########
    data_file = "./data/toy_time_series/exp_sine_dependency.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    df_raw = df_raw[[0]]
    df_raw.columns = ["exp sine"]

    data_file_time = "./data/toy_time_series/sine_datetime.csv"
    time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(time_series[0])
    df_raw.index = time_series
    df_raw.index.name = "date_time"

    # Add synthetic anomaly to data
    trend = np.linspace(0, 0, num=len(df_raw))
    time_anomaly = 120
    new_trend = np.linspace(0, -1, num=len(df_raw) - time_anomaly)
    trend[time_anomaly:] = trend[time_anomaly:] + new_trend
    df_raw = df_raw.add(trend, axis=0)

    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df_raw,
        train_split=0.4,
        validation_split=0.1,
        output_col=output_col,
        standardization=False,
    )
    
    train_data, validation_data, test_data, all_data = data_processor.get_splits()
    # plt.plot(all_data["y"])
    # plt.show()

    ######### Define model with parameters #########
    def model_with_parameters(param):
        model = Model(
            LocalTrend(),
            KernelRegression(period=24,
                            kernel_length=param["kernel_length"],
                            # std_error_cp=param["std_error_cp"],
                            num_control_point=10,
                            mu_control_point = 0.1,
                            var_control_point = 0.1
                            ),
            WhiteNoise(std_error=param["sigma_v"])
        )
        model.auto_initialize_baseline_states(train_data["y"])
        mu_preds, std_preds,_ = model.filter(data=train_data)

        obs = data_processor.get_data("train").flatten()
        log_lik = metric.log_likelihood(
            prediction=mu_preds,
            observation=obs,
            std=std_preds,
        )
        model.metric_optim = -log_lik

        return model

    # ######### Parameter optimization #########
    if param_optimization:
        param_space = {
            "kernel_length": [0.2, 0.99],
            # "std_error_cp": [1e-3, 5e-1],
            "sigma_v": [1e-3, 2e-1],
            # "std_transition_error": [1e-6, 1e-4],
            # "norm_to_abnorm_prob": [1e-6, 1e-4],
        }
        # Define optimizer
        model_optimizer = Optimizer(
            model=model_with_parameters,
            param=param_space,
            num_optimization_trial=num_trial_optimization,
            num_startup_trials=10,
            mode="min",
        )
        model_optimizer.optimize()

        # Get best model
        param = model_optimizer.get_best_param()
        model_optim = model_with_parameters(param)

        model_optim.auto_initialize_baseline_states(train_data["y"])
        abnorm_model = Model(
            LocalAcceleration(),
            KernelRegression(period=24,
                            kernel_length=param["kernel_length"],
                            # std_error_cp=param["std_error_cp"],
                            num_control_point=10,
                            mu_control_point = 0.1,
                            var_control_point = 0.1
                            ),
            WhiteNoise(std_error=param["sigma_v"])
        )
        skf_optim = SKF(
            norm_model=model_optim,
            abnorm_model=abnorm_model,
            std_transition_error=1e-4,
            norm_to_abnorm_prob=1e-5,
        )

        skf_optim_dict = skf_optim.get_dict()
        skf_optim_dict["model_param"] = param
        skf_optim_dict["cov_names"] = train_data["cov_names"]
        with open("saved_params/toy_anomaly_detection_tune_kr.pkl", "wb") as f:
            pickle.dump(skf_optim_dict, f)
    else:
        with open("saved_params/toy_anomaly_detection_tune_kr.pkl", "rb") as f:
            skf_optim_dict = pickle.load(f)
        skf_optim = SKF.load_dict(skf_optim_dict)

    # ######### Detect anomaly #########
    # print("Model parameters used:", skf_optim_dict["model_param"])

    # param_space = {
    #     "kernel_length": 0.8,
    #     "std_error_cp": 1e-3,
    #     "sigma_v": 1e-2,
    #     "std_transition_error": 1e-4,
    #     "norm_to_abnorm_prob": 1e-5,
    # }
    # skf_optim = model_with_parameters(param_space)

    filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)
    smooth_marginal_abnorm_prob, states = skf_optim.smoother()

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        states_type="smooth",
        model_prob=filter_marginal_abnorm_prob,
        states_to_plot=["level", "trend", "kernel regression", "white noise"],
    )
    ax[0].axvline(x=data_processor.data.index[time_anomaly], color="r", linestyle="--")
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
