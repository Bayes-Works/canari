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
)
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
import fire

# Parameters space for searching
sigma_v_space = [1e-3, 2e-1]
look_back_len_space = [12, 52]
SKF_std_transition_error_space = [1e-6, 1e-4]
SKF_norm_to_abnorm_prob_space = [1e-6, 1e-4]
synthetic_anomaly_slope_space = [1e-3, 1e-1]

# Fix parameters:
sigma_v_fix = 0.00997597533731805
look_back_len_fix = 35
SKF_std_transition_error_fix = 1e-3
SKF_norm_to_abnorm_prob_fix = 6.151115737882297e-06


def main(
    num_epoch: int = 50,
    model_search: bool = True,
    SKF_search: bool = True,
    num_sample_optimization: int = 40,
    verbose: int = 1,
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
    train_data, validation_data, _, all_data = data_processor.get_splits()

    # Define model
    def create_model(config) -> Model:
        model = Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=config["look_back_len"],
                num_features=2,
                num_layer=1,
                num_hidden_unit=50,
                device="cpu",
                manual_seed=1,
            ),
            WhiteNoise(std_error=config["sigma_v"]),
        )
        return model

    # Define search space
    config = {}
    config["look_back_len"] = [12, 52]
    config["sigma_v"] = [1e-3, 2e-1]


def model_train(model, data_processor, train_data, validation_data, num_epoch):
    model.auto_initialize_baseline_states(train_data["y"][0:23])
    noise_index = model.states_name.index("white noise")
    for epoch in range(num_epoch):
        mu_validation_preds, std_validation_preds, _ = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
        )

        mu_validation_preds = normalizer.unstandardize(
            mu_validation_preds,
            data_processor.norm_const_mean[data_processor.output_col],
            data_processor.norm_const_std[data_processor.output_col],
        )

        mse = metric.mse(
            mu_validation_preds,
            data_processor.validation_data[:, data_processor.output_col].flatten(),
        )

        model.early_stopping(evaluate_metric=mse, mode="min")
        if model.stop_training:
            break

    return model


def model_optimizer(
    model, config, train, data_processor, train_data, validation_data, num_epoch
):
    def model_objective(
        config, model, train, data_processor, train_data, validation_data, num_epoch
    ):
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

        trained_model = train(
            model,
            data_processor,
            train_data,
            validation_data,
            num_epoch,
        )
        tune.report({"metric": trained_model.early_stop_metric})

    model_optimizer_runner = tune.run(
        tune.with_parameters(
            model_objective,
            model=model_optim_dict,
        ),
        config={
            "sigma_v": tune.loguniform(sigma_v_space[0], sigma_v_space[-1]),
            "look_back_len": tune.randint(
                look_back_len_space[0], look_back_len_space[-1]
            ),
        },
        search_alg=OptunaSearch(metric="metric", mode="min"),
        name="Model optimizer",
        num_samples=5,
        verbose=1,
        raise_on_failed_trial=False,
    )


if __name__ == "__main__":
    fire.Fire(main)
