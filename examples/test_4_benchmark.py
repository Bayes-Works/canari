import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import (
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    WhiteNoise,
    Model,
    SKF,
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
import fire

# Parameters space for searching
sigma_v_space = [1e-3, 2e-1]
look_back_len_space = [12, 52]
SKF_std_transition_error_space = [1e-6, 1e-3]
SKF_norm_to_abnorm_prob_space = [1e-6, 1e-3]
synthetic_anomaly_slope_space = [1e-3, 5e-2]

# Fix parameters:
sigma_v_fix = 0.0877013151472895
look_back_len_fix = 50
SKF_std_transition_error_fix = 0.0002840197607249382
SKF_norm_to_abnorm_prob_fix = 0.00012306655795926166


def main(
    num_epoch: int = 50,
    model_search: bool = False,
    SKF_search: bool = False,
    num_sample_optimization: int = 100,
    verbose: int = 1,
    grid_search_model: bool = False,
    grid_search_SKF: bool = False,
):
    # Read data
    data_file = "./data/benchmark_data/test_4_data.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(df_raw.iloc[:, 0])
    df_raw = df_raw.iloc[:, 1:]
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["displacement_y", "water_level", "temp_min", "temp_max"]
    lags = [0, 4, 4, 4]
    df_raw = DataProcess.add_lagged_columns(df_raw, lags)
    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df_raw,
        time_covariates=["week_of_year"],
        train_split=0.23,
        validation_split=0.07,
        output_col=output_col,
    )
    train_data, validation_data, _, all_data = data_processor.get_splits()

    # fig, ax = plt.subplots(figsize=(10, 6))
    # plot_data(
    #     data_processor=data_processor,
    #     normalization=True,
    #     plot_test_data=True,
    #     plot_column=output_col,
    #     validation_label="y",
    # )
    # plt.show()

    # Model optimizer function
    def model_optimizer(config, return_model=False):
        sigma_v = config["sigma_v"]
        look_back_len = config["look_back_len"]

        model = Model(
            LocalTrend(var_states=[1e-1, 1e-1]),
            LstmNetwork(
                look_back_len=look_back_len,
                num_features=df_raw.shape[1],
                num_layer=1,
                num_hidden_unit=50,
                device="cpu",
                manual_seed=1,
            ),
            WhiteNoise(std_error=sigma_v),
        )
        model.auto_initialize_baseline_states(train_data["y"][0:47])  # 72

        noise_index = model.states_name.index("white noise")
        scheduled_sigma_v = 5
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
                train_data=train_data,
                validation_data=validation_data,
            )

            mu_validation_preds_unnorm = normalizer.unstandardize(
                mu_validation_preds,
                data_processor.norm_const_mean[output_col],
                data_processor.norm_const_std[output_col],
            )

            std_validation_preds_unnorm = normalizer.unstandardize_std(
                std_validation_preds,
                data_processor.norm_const_std[output_col],
            )

            validation_log_lik = metric.log_likelihood(
                prediction=mu_validation_preds_unnorm,
                observation=data_processor.validation_data[:, output_col].flatten(),
                std=std_validation_preds_unnorm,
            )
            # if return_model:
            #     # Plotting
            #     fig, ax = plt.subplots(figsize=(10, 6))
            #     plot_data(
            #         data_processor=data_processor,
            #         normalization=True,
            #         plot_test_data=False,
            #         plot_column=output_col,
            #         validation_label="y",
            #     )
            #     plot_prediction(
            #         data_processor=data_processor,
            #         mean_validation_pred=mu_validation_preds,
            #         std_validation_pred=std_validation_preds,
            #         validation_label=[r"$\mu$", f"$\pm\sigma$"],
            #     )
            #     plot_states(
            #         data_processor=data_processor,
            #         states=states,
            #         states_to_plot=["local level"],
            #         sub_plot=ax,
            #     )
            #     plt.legend()
            #     plt.show()

            model.early_stopping(evaluate_metric=validation_log_lik, mode="max")
            if model.stop_training:
                break

        if return_model:
            # Plotting
            # fig, ax = plt.subplots(figsize=(10, 6))
            # plot_data(
            #     data_processor=data_processor,
            #     normalization=True,
            #     plot_test_data=False,
            #     plot_column=output_col,
            #     validation_label="y",
            # )
            # plot_prediction(
            #     data_processor=data_processor,
            #     mean_validation_pred=mu_validation_preds,
            #     std_validation_pred=std_validation_preds,
            #     validation_label=[r"$\mu$", f"$\pm\sigma$"],
            # )
            # plot_states(
            #     data_processor=data_processor,
            #     states=states,
            #     states_to_plot=["local level"],
            #     sub_plot=ax,
            # )
            # plt.legend()
            # plt.show()
            return model

        else:
            tune.report({"metric": model.early_stop_metric})

    # Run hyperparameter tuning if model_search = True
    if model_search:
        if grid_search_model:
            model_optimizer_runner = tune.run(
                model_optimizer,
                config={
                    "sigma_v": tune.grid_search([0.1, 0.2, 0.3, 0.4]),
                    "look_back_len": tune.grid_search([12, 26, 52]),
                },
                name="Model optimizer",
                num_samples=1,
                verbose=verbose,
                raise_on_failed_trial=False,
            )
        else:
            model_optimizer_runner = tune.run(
                model_optimizer,
                config={
                    "sigma_v": tune.uniform(sigma_v_space[0], sigma_v_space[-1]),
                    "look_back_len": tune.randint(
                        look_back_len_space[0], look_back_len_space[-1]
                    ),
                },
                search_alg=OptunaSearch(metric="metric", mode="max"),
                name="Model optimizer",
                num_samples=int(num_sample_optimization / 2),
                verbose=verbose,
                raise_on_failed_trial=False,
            )
        # Get the optimal parameters
        model_optim_params = model_optimizer_runner.get_best_config(
            metric="metric", mode="max"
        )

    # Run fix parameters
    else:
        model_optim_params = {
            "sigma_v": sigma_v_fix,
            "look_back_len": look_back_len_fix,
        }

    # Re-train with the optimal parameters
    model_optim = model_optimizer(model_optim_params, return_model=True)
    model_optim_dict = model_optim.save_model_dict()

    # Initialize SKF tuning
    def SKF_optimizer(config, model_params=None, return_model=False):
        std_transition_error = config["std_transition_error"]
        norm_to_abnorm_prob = config["norm_to_abnorm_prob"]
        slope = config["slope"]

        norm_model = load_model_dict(model_params)
        abnorm_model = Model(
            LocalAcceleration(),
            LstmNetwork(),
            WhiteNoise(std_error=model_optim_params["sigma_v"]),
        )
        skf = SKF(
            norm_model=norm_model,
            abnorm_model=abnorm_model,
            std_transition_error=std_transition_error,
            norm_to_abnorm_prob=norm_to_abnorm_prob,
            abnorm_to_norm_prob=1e-1,
            norm_model_prior_prob=0.99,
            conditional_likelihood=False,
        )
        skf.save_initial_states()

        if return_model:
            return skf
        else:
            detection_rate_raw, false_rate = skf.detect_synthetic_anomaly(
                data=train_data,
                num_anomaly=50,
                slope_anomaly=slope,
            )
            if detection_rate_raw < 0.5 or false_rate > 0:
                detection_rate = 2
            else:
                detection_rate = detection_rate_raw
            # Objective function
            score = detection_rate + 5 * slope
            tune.report(
                {
                    "score": score,
                    "detection_rate": detection_rate_raw,
                    "false_alarm_rate": false_rate,
                }
            )

    # Run SKF hyperparameter tuning if SKF_search = True
    if SKF_search:
        if grid_search_SKF:
            SKF_optimizer_runner = tune.run(
                tune.with_parameters(
                    SKF_optimizer,
                    model_params=model_optim_dict,
                ),
                config={
                    "std_transition_error": tune.grid_search([1e-6, 1e-5, 1e-4]),
                    "norm_to_abnorm_prob": tune.grid_search([1e-6, 1e-5, 1e-4]),
                    "slope": tune.grid_search(
                        [0.002, 0.004, 0.006, 0.008, 0.01, 0.03, 0.05, 0.07, 0.09]
                    ),
                    # "slope": tune.grid_search([0.01, 0.02, 0.008]),
                },
                name="SKF_optimizer",
                num_samples=1,
                verbose=verbose,
                raise_on_failed_trial=False,
            )
        else:
            SKF_optimizer_runner = tune.run(
                tune.with_parameters(
                    SKF_optimizer,
                    model_params=model_optim_dict,
                ),
                config={
                    "std_transition_error": tune.loguniform(
                        SKF_std_transition_error_space[0],
                        SKF_std_transition_error_space[-1],
                    ),
                    "norm_to_abnorm_prob": tune.loguniform(
                        SKF_norm_to_abnorm_prob_space[0],
                        SKF_norm_to_abnorm_prob_space[-1],
                    ),
                    "slope": tune.uniform(
                        synthetic_anomaly_slope_space[0],
                        synthetic_anomaly_slope_space[-1],
                    ),
                },
                search_alg=OptunaSearch(metric="score", mode="min"),
                name="SKF_optimizer",
                num_samples=num_sample_optimization,
                verbose=verbose,
                raise_on_failed_trial=False,
            )

        # Get the best parameters for SKF
        SKF_optim_params = SKF_optimizer_runner.get_best_config(
            metric="score", mode="min"
        )

    # Run with fixed parameters
    else:
        SKF_optim_params = {
            "std_transition_error": SKF_std_transition_error_fix,
            "norm_to_abnorm_prob": SKF_norm_to_abnorm_prob_fix,
            "slope": synthetic_anomaly_slope_space[-1],
        }

    # Anomaly Detection with optimal parameters
    skf_optim = SKF_optimizer(
        SKF_optim_params, model_params=model_optim_dict, return_model=True
    )
    filter_marginal_abnorm_prob, _ = skf_optim.filter(data=all_data)
    smooth_marginal_abnorm_prob, states = skf_optim.smoother(data=all_data)

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

    print("Model parameters used:", model_optim_params)
    print("SKF model parameters used:", SKF_optim_params)


if __name__ == "__main__":
    fire.Fire(main)
