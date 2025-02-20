import fire
import copy
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import (
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    WhiteNoise,
    Model,
    ModelOptimizer,
    SKF,
    SKFOptimizer,
    load_model_dict,
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer


def main(
    num_trial_optimization: int = 2,
    grid_search: bool = False,
):
    # Read data
    data_file = "./data/toy_time_series/sine.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    data_file_time = "./data/toy_time_series/sine_datetime.csv"
    time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(time_series[0])
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["values"]

    # Add synthetic anomaly to data
    trend = np.linspace(0, 0, num=len(df_raw))
    time_anomaly = 200
    new_trend = np.linspace(0, 2, num=len(df_raw) - time_anomaly)
    trend[time_anomaly:] = trend[time_anomaly:] + new_trend
    df_raw = df_raw.add(trend, axis=0)

    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df_raw,
        time_covariates=["hour_of_day"],
        train_start="2000-01-01 00:00:00",
        train_end="2000-01-09 23:00:00",
        validation_start="2000-01-10 00:00:00",
        validation_end="2000-01-10 23:00:00",
        test_start="2000-01-11 00:00:00",
        output_col=output_col,
    )

    # Define model
    def initialize_model(param):
        return Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=param["look_back_len"],
                num_features=2,
                num_layer=1,
                num_hidden_unit=50,
                device="cpu",
                manual_seed=1,
            ),
            WhiteNoise(std_error=param["sigma_v"]),
        )

    # Define parameter search space
    param = {
        "look_back_len": [10, 52],
        "sigma_v": [1e-3, 2e-1],
    }

    # Define optimizer
    model_optimizer = ModelOptimizer(
        initialize_model=initialize_model,
        train=training,
        param_space=param,
        data_processor=data_processor,
        num_optimization_trial=num_trial_optimization,
    )
    model_optimizer.optimize()

    # Get best model
    model_optim = model_optimizer.get_best_model()

    # Train best model
    model_optim, states_optim, mu_validation_preds, std_validation_preds = training(
        model=model_optim, data_processor=data_processor
    )

    # Save best model for SKF analysis later
    model_optim_dict = model_optim.save_model_dict()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data(
        data_processor=data_processor,
        normalization=True,
        plot_test_data=False,
        plot_column=output_col,
        validation_label="y",
    )
    plot_prediction(
        data_processor=data_processor,
        mean_validation_pred=mu_validation_preds,
        std_validation_pred=std_validation_preds,
        validation_label=[r"$\mu$", f"$\pm\sigma$"],
    )
    plot_states(
        data_processor=data_processor,
        states=states_optim,
        states_to_plot=["local level"],
        sub_plot=ax,
    )
    plt.legend()
    plt.title("Validation predictions")
    plt.show()

    # Define SKF model
    def initialize_skf(skf_param, model_param: dict):
        norm_model = load_model_dict(model_param)
        abnorm_model = Model(
            LocalAcceleration(),
            LstmNetwork(),
            WhiteNoise(),
        )
        skf = SKF(
            norm_model=norm_model,
            abnorm_model=abnorm_model,
            std_transition_error=skf_param["std_transition_error"],
            norm_to_abnorm_prob=skf_param["norm_to_abnorm_prob"],
            abnorm_to_norm_prob=1e-1,
            norm_model_prior_prob=0.99,
        )
        skf.save_initial_states()
        return skf

    # Define parameter search space
    slope_upper_bound = 5e-2
    slope_lower_bound = 1e-3
    if grid_search:
        skf_param = {
            "std_transition_error": [1e-6, 1e-5, 1e-4],
            "norm_to_abnorm_prob": [1e-6, 1e-5, 1e-4],
            "slope": [0.002, 0.004, 0.006, 0.008, 0.01, 0.03, 0.05, 0.07, 0.09],
        }
    else:
        skf_param = {
            "std_transition_error": [1e-6, 1e-3],
            "norm_to_abnorm_prob": [1e-6, 1e-3],
            "slope": [slope_lower_bound, slope_upper_bound],
        }

    # # Plot synthetic anomaly
    synthetic_anomaly_data = DataProcess.add_synthetic_anomaly(
        data_processor.train_split,
        num_samples=1,
        slope=[slope_lower_bound, slope_upper_bound],
    )
    plot_data(
        data_processor=data_processor,
        normalization=True,
        plot_validation_data=False,
        plot_test_data=False,
        plot_column=output_col,
    )
    for ts in synthetic_anomaly_data:
        plt.plot(data_processor.train_time, ts["y"])
    plt.legend(["data without anomaly", "largest anomaly tested","smallest anomaly tested"])
    # plt.legend()
    plt.title("Train data with added synthetic anomalies")
    plt.show()

    # Define optimizer
    skf_optimizer = SKFOptimizer(
        initialize_skf=initialize_skf,
        model=model_optim_dict,
        param_space=skf_param,
        data_processor=data_processor,
        num_synthetic_anomaly=50,
        num_optimization_trial=num_trial_optimization * 2,
        grid_search=grid_search,
    )
    skf_optimizer.optimize()

    # Get best model
    skf_optim = skf_optimizer.get_best_model()
    filter_marginal_abnorm_prob, states = skf_optim.filter(
        data=data_processor.all_data_split
    )

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        states_to_plot=["local level", "local trend", "lstm", "white noise"],
        model_prob=filter_marginal_abnorm_prob,
        color="b",
        legend_location="upper left",
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.show()

    print("Model parameters used:", model_optimizer.param_optim)
    print("SKF model parameters used:", skf_optimizer.param_optim)
    print("-----")


def training(model, data_processor, num_epoch: int = 50):
    """
    Training procedure
    """

    model.auto_initialize_baseline_states(data_processor.train_split["y"][0:23])
    noise_index = model.states_name.index("white noise")
    scheduled_sigma_v = 5
    sigma_v = model.components["white noise"].std_error
    states_optim = None
    mu_validation_preds_optim = None
    std_validation_preds_optim = None

    for epoch in range(num_epoch):
        # Decaying observation's variance
        scheduled_sigma_v = exponential_scheduler(
            curr_v=scheduled_sigma_v,
            min_v=sigma_v,
            decaying_factor=0.9,
            curr_iter=epoch,
        )
        model.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

        mu_validation_preds, std_validation_preds, states = model.lstm_train(
            train_data=data_processor.train_split,
            validation_data=data_processor.validation_split,
        )

        mu_validation_preds_unnorm = normalizer.unstandardize(
            mu_validation_preds,
            data_processor.norm_const_mean[data_processor.output_col],
            data_processor.norm_const_std[data_processor.output_col],
        )

        std_validation_preds_unnorm = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor.norm_const_std[data_processor.output_col],
        )

        validation_log_lik = metric.log_likelihood(
            prediction=mu_validation_preds_unnorm,
            observation=data_processor.validation_data[
                :, data_processor.output_col
            ].flatten(),
            std=std_validation_preds_unnorm,
        )

        model.early_stopping(evaluate_metric=-validation_log_lik, mode="min")

        if epoch == model.optimal_epoch:
            mu_validation_preds_optim = mu_validation_preds.copy()
            std_validation_preds_optim = std_validation_preds.copy()
            states_optim = copy.copy(states)
        if model.stop_training:
            break

    return (
        model,
        states_optim,
        mu_validation_preds_optim,
        std_validation_preds_optim,
    )


if __name__ == "__main__":
    fire.Fire(main)
