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
    ModelOptimizer,
    SKF,
    SKFOptimizer,
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise


def main(
    num_trial_optim_model: int = 50,
    num_trial_optim_skf: int = 300,
    param_optimization: bool = False,
    param_grid_search: bool = False,
    smoother: bool = True,
    plot: bool = False,
):
    ######### Data processing #########
    # Read data
    data_file = "./data/benchmark_data/test_10_data.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(df_raw.iloc[:, 0])
    df_raw = df_raw.iloc[:, 1:]
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["displacement_z", "water_level", "temp_min", "temp_max"]
    lags = [0, 4, 4, 4]
    df_raw = DataProcess.add_lagged_columns(df_raw, lags)
    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df_raw,
        time_covariates=["week_of_year"],
        train_split=0.27975,
        validation_split=0.054,
        output_col=output_col,
    )
    train_data, validation_data, test_data, all_data = data_processor.get_splits()

    ######### Define model with parameters #########
    def model_with_parameters(param, train_data, validation_data):
        model = Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=int(param["look_back_len"]),
                num_features=17,
                num_layer=1,
                infer_len=52 * 3,
                num_hidden_unit=50,
                manual_seed=1,
                smoother=smoother,
            ),
            WhiteNoise(std_error=param["sigma_v"]),
        )

        model.auto_initialize_baseline_states(train_data["y"][0 : 52 * 3])
        mu_validation_preds_optim = None
        std_validation_preds_optim = None
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

            if epoch == model.optimal_epoch:
                mu_validation_preds_optim = mu_validation_preds.copy()
                std_validation_preds_optim = std_validation_preds.copy()

            if model.stop_training:
                break

        return (
            model,
            mu_validation_preds_optim,
            std_validation_preds_optim,
        )

    ######### Define SKF model with parameters #########
    def skf_with_parameters(skf_param_space, model_param: dict, train_data):
        norm_model = Model.load_dict(model_param)

        abnorm_model = Model(
            LocalAcceleration(),
            LstmNetwork(),
            WhiteNoise(),
        )
        skf = SKF(
            norm_model=norm_model,
            abnorm_model=abnorm_model,
            std_transition_error=skf_param_space["std_transition_error"],
            norm_to_abnorm_prob=skf_param_space["norm_to_abnorm_prob"],
        )
        skf.save_initial_states()

        detection_rate, false_rate, false_alarm_train = skf.detect_synthetic_anomaly(
            data=train_data,
            num_anomaly=50,
            slope_anomaly=skf_param_space["slope"],
        )
        skf.metric_optim["detection_rate"] = detection_rate
        skf.metric_optim["false_rate"] = false_rate
        skf.metric_optim["false_alarm_train"] = false_alarm_train

        return skf

    ######### Parameter optimization #########
    if param_optimization:
        # # Optimize for model
        # Define parameter search space
        if param_optimization:
            param_space = {
                "look_back_len": [10, 65],
                "sigma_v": [1e-3, 2e-1],
            }
        elif param_grid_search:
            param_space = {
                "look_back_len": [12, 26, 52],
                "sigma_v": [1e-1, 2e-1, 3e-1, 4e-1],
            }
        # Define optimizer
        model_optimizer = ModelOptimizer(
            model=model_with_parameters,
            param_space=param_space,
            train_data=train_data,
            validation_data=validation_data,
            num_optimization_trial=num_trial_optim_model,
            grid_search=param_grid_search,
        )
        model_optimizer.optimize()
        # Get best model
        param = model_optimizer.get_best_param()

        # Train best model
        model_optim, mu_validation_preds, std_validation_preds = model_with_parameters(
            param, train_data, validation_data
        )

        if plot:
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_data(
                data_processor=data_processor,
                standardization=True,
                plot_test_data=False,
                plot_column=output_col,
                validation_label="y",
            )
            plot_prediction(
                data_processor=data_processor,
                mean_validation_pred=mu_validation_preds,
                std_validation_pred=std_validation_preds,
                validation_label=["mean", "std"],
            )
            plot_states(
                data_processor=data_processor,
                states=model_optim.states,
                standardization=True,
                states_to_plot=["level"],
                sub_plot=ax,
            )
            plt.legend()
            plt.title("Validation predictions")
            plt.show()

        # Save best model for SKF analysis later
        model_optim_dict = model_optim.get_dict(time_step=0)

        # # Optimize for skf
        # Define parameter search space
        slope_upper_bound = 5e-2
        slope_lower_bound = 1e-3

        if plot:
            # # Plot synthetic anomaly
            synthetic_anomaly_data = DataProcess.add_synthetic_anomaly(
                train_data,
                num_samples=1,
                slope=[slope_lower_bound, slope_upper_bound],
            )
            plot_data(
                data_processor=data_processor,
                standardization=True,
                plot_validation_data=False,
                plot_test_data=False,
                plot_column=output_col,
                train_label="data without anomaly",
            )

            train_time = data_processor.get_time("train")
            for ts in synthetic_anomaly_data:
                plt.plot(train_time, ts["y"])
            plt.legend(
                [
                    "data without anomaly",
                    "",
                    "smallest anomaly tested",
                    "largest anomaly tested",
                ]
            )
            plt.title("Train data with added synthetic anomalies")
            plt.show()

        if param_grid_search:
            skf_param_space = {
                "std_transition_error": [1e-6, 1e-5, 1e-4, 1e-3],
                "norm_to_abnorm_prob": [1e-6, 1e-5, 1e-4, 1e-3],
                "slope": [0.002, 0.004, 0.006, 0.008, 0.01, 0.03, 0.05, 0.07, 0.09],
            }
        elif param_optimization:
            skf_param_space = {
                "std_transition_error": [1e-6, 1e-3],
                "norm_to_abnorm_prob": [1e-6, 1e-3],
                "slope": [slope_lower_bound, slope_upper_bound],
            }
        skf_optimizer = SKFOptimizer(
            skf=skf_with_parameters,
            model_param=model_optim_dict,
            param_space=skf_param_space,
            data=train_data,
            num_optimization_trial=num_trial_optim_skf,
            grid_search=param_grid_search,
        )
        skf_optimizer.optimize()
        # Get parameters
        skf_param = skf_optimizer.get_best_param()

        skf_optim = skf_with_parameters(skf_param, model_optim_dict, train_data)
        skf_optim_dict = skf_optim.get_dict()
        skf_optim_dict["model_param"] = param
        skf_optim_dict["skf_param"] = skf_param
        skf_optim_dict["cov_names"] = train_data["cov_names"]
        with open("saved_params/benchmark_10.pkl", "wb") as f:
            pickle.dump(skf_optim_dict, f)
    else:
        # # Load saved skf model
        with open("saved_params/benchmark_10.pkl", "rb") as f:
            skf_optim_dict = pickle.load(f)
        skf_optim = SKF.load_dict(skf_optim_dict)

    ######### Detect anomaly #########
    print("Model parameters used:", skf_optim_dict["model_param"])
    print("SKF model parameters used:", skf_optim_dict["skf_param"])

    filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)
    smooth_marginal_abnorm_prob, states = skf_optim.smoother()

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        model_prob=filter_marginal_abnorm_prob,
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.savefig("./saved_results/BM10.png")
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
