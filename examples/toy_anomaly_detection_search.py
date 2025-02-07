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
    plot_data,
    plot_prediction,
    plot_skf_states,
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
synthetic_anomaly_slope_space = [1e-3, 1e-1]

# Fix parameters:
sigma_v_fix = 1e-2
look_back_len_fix = 12
SKF_std_transition_error_fix = 1e-4
SKF_norm_to_abnorm_prob_fix = 1e-4


def main(
    num_epoch: int = 50,
    model_search: bool = False,
    SKF_search: bool = True,
    num_sample_optimization: int = 5,
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
    train_data, validation_data, _, all_data = data_processor.get_splits()

    # Model optimizer function
    def model_optimizer(config, return_model=False):
        sigma_v = config["sigma_v"]
        look_back_len = config["look_back_len"]

        model = Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=look_back_len,
                num_features=2,
                num_layer=1,
                num_hidden_unit=50,
                device="cpu",
                manual_seed=1,
            ),
            WhiteNoise(std_error=sigma_v),
        )
        model.auto_initialize_baseline_states(train_data["y"][0:23])

        scheduled_sigma_v = 1
        for epoch in range(num_epoch):
            scheduled_sigma_v = exponential_scheduler(
                curr_v=scheduled_sigma_v,
                min_v=sigma_v,
                decaying_factor=0.9,
                curr_iter=epoch,
            )
            noise_index = model.states_name.index("white noise")
            model.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

            mu_validation_preds, std_validation_preds, _ = model.lstm_train(
                train_data=train_data,
                validation_data=validation_data,
            )

            mu_validation_preds = normalizer.unstandardize(
                mu_validation_preds,
                data_processor.norm_const_mean[output_col],
                data_processor.norm_const_std[output_col],
            )

            mse = metric.mse(
                mu_validation_preds,
                data_processor.validation_data[:, output_col].flatten(),
            )

            model.early_stopping(evaluate_metric=mse, mode="min")
            if model.stop_training:
                break

        if return_model:
            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_data(
                data_processor=data_processor,
                normalization=False,
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
            plt.legend()
            plt.show()
            return model

        else:
            tune.report({"metric": model.early_stop_metric})

    # Run hyperparameter tuning if model_search = True
    if model_search:
        model_optimizer_runner = tune.run(
            model_optimizer,
            config={
                "sigma_v": tune.loguniform(sigma_v_space[0], sigma_v_space[-1]),
                "look_back_len": tune.randint(
                    look_back_len_space[0], look_back_len_space[-1]
                ),
            },
            search_alg=OptunaSearch(metric="metric", mode="min"),
            name="Model optimizer",
            num_samples=num_sample_optimization,
            verbose=1,
        )
        # Get the optimal parameters
        model_optim_params = model_optimizer_runner.get_best_config(
            metric="metric", mode="min"
        )

    # Run fix parameters
    else:
        model_optim_params = {
            "sigma_v": sigma_v_fix,
            "look_back_len": look_back_len_fix,
        }

    # Re-train with the optimal parameters
    model_optim = model_optimizer(model_optim_params, return_model=True)
    model_optim_dict = {
        "lstm_net_params": model_optim.lstm_net.get_state_dict(),
        "mu_states": model_optim.mu_states,
        "var_states": model_optim.var_states,
        "sigma_v": model_optim_params["sigma_v"],
        "look_back_len": model_optim_params["look_back_len"],
    }

    # Initialize SKF tuning
    def SKF_optimizer(config, model_params=None, return_model=False):
        std_transition_error = config["std_transition_error"]
        norm_to_abnorm_prob = config["norm_to_abnorm_prob"]
        slope = config["slope"]

        norm_model = Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=model_params["look_back_len"],
                num_features=2,
                num_layer=1,
                num_hidden_unit=50,
                device="cpu",
                manual_seed=1,
            ),
            WhiteNoise(std_error=model_params["sigma_v"]),
        )

        norm_model.set_states(model_params["mu_states"], model_params["var_states"])
        norm_model.lstm_net.load_state_dict(model_optim_dict["lstm_net_params"])

        abnorm_model = Model(LocalAcceleration(), LstmNetwork(), WhiteNoise())
        skf = SKF(
            norm_model=norm_model,
            abnorm_model=abnorm_model,
            std_transition_error=std_transition_error,
            norm_to_abnorm_prob=norm_to_abnorm_prob,
            abnorm_to_norm_prob=1e-1,
            norm_model_prior_prob=0.99,
        )

        if return_model:
            return skf
        else:
            detection_rate = skf.detect_synthetic_anomaly(
                data=train_data, num_anomaly=50, slope_anomaly=slope
            )
            if detection_rate > 0.5:
                detection_rate = 0
            # Objective function
            score = detection_rate - slope / synthetic_anomaly_slope_space[-1] / 2
            tune.report({"score": score})

    # Run SKF hyperparameter tuning if SKF_search = True
    if SKF_search:
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
                    SKF_norm_to_abnorm_prob_space[0], SKF_norm_to_abnorm_prob_space[-1]
                ),
                "slope": tune.uniform(
                    synthetic_anomaly_slope_space[0], synthetic_anomaly_slope_space[-1]
                ),
            },
            search_alg=OptunaSearch(metric="score", mode="max"),
            name="SKF_optimizer",
            num_samples=num_sample_optimization,
        )
        # Get the best parameters for SKF
        SKF_optim_params = SKF_optimizer_runner.get_best_config(
            metric="score", mode="max"
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
