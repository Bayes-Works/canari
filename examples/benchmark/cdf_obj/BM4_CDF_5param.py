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
    num_trial_optim_model: int = 300,
    param_optimization: bool = True,
    smoother: bool = True,
    plot: bool = False,
):
    ######### Data processing #########
    # Read data
    data_file = "./data/benchmark_data/test_4_data.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(df_raw.iloc[:, 0])
    df_raw = df_raw.iloc[:, 1:]
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["y", "water_level", "temp_min", "temp_max"]
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
    train_data, validation_data, test_data, all_data = data_processor.get_splits()
    seed = np.random.randint(0, 100)

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
                manual_seed=seed,
                smoother=smoother,
            ),
            WhiteNoise(std_error=param["sigma_v"]),
        )

        model.auto_initialize_baseline_states(train_data["y"][0 : 52 * 3])
        num_epoch = 50
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

        num_anomaly = 50
        detection_rate, false_rate, false_alarm_train = skf.detect_synthetic_anomaly(
            data=train_data,
            num_anomaly=num_anomaly,
            slope_anomaly=param["slope"] / 52,
        )

        data_len_year = (
            data_processor.data.index[data_processor.train_end]
            - data_processor.data.index[data_processor.train_start]
        ).days / 365.25

        metric_optim = skf.objective(
            detection_rate, false_rate / data_len_year, param["slope"]
        )

        skf.load_initial_states()

        skf.metric_optim = metric_optim

        return skf

    if param_optimization:
        param_space = {
            "look_back_len": [12, 52],
            "sigma_v": [1e-3, 2e-1],
            "std_transition_error": [1e-6, 1e-4],
            "norm_to_abnorm_prob": [1e-6, 1e-4],
            "slope": [0.1, 0.6],
        }
        # Define optimizer
        model_optimizer = ModelOptimizer(
            model=model_with_parameters,
            param_space=param_space,
            train_data=train_data,
            validation_data=validation_data,
            num_optimization_trial=num_trial_optim_model,
            num_startup_trials=50,
            mode="max",
        )
        model_optimizer.optimize()
        # Get best model
        param = model_optimizer.get_best_param()
        skf_optim = model_with_parameters(
            param, train_data, validation_data
        )

        skf_optim_dict = skf_optim.get_dict()
        skf_optim_dict["model_param"] = param
        skf_optim_dict["cov_names"] = train_data["cov_names"]
        with open("saved_params/cdf_obj/BM4_CDF_5param_1.pkl", "wb") as f:
            pickle.dump(skf_optim_dict, f)
    else:
        with open("saved_params/cdf_obj/BM4_CDF_5param_1.pkl", "rb") as f:
            skf_optim_dict = pickle.load(f)
        skf_optim = SKF.load_dict(skf_optim_dict)

    filter_marginal_abnorm_prob, states, *_ = skf_optim.filter(data=all_data)

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        model_prob=filter_marginal_abnorm_prob,
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.savefig("./saved_results/cdf_obj/BM4_CDF_5param_1.png")
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
