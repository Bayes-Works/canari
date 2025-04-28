import os
import fire
import pandas as pd
import numpy as np
import pytest
import matplotlib.pyplot as plt

from pytagi import Normalizer as normalizer
import pytagi.metric as metric

from canari import DataProcess, Model, plot_data, plot_prediction
from canari.component import LocalTrend, LstmNetwork, WhiteNoise


def model_test_runner(model: Model, plot: bool) -> float:
    """
    Run training and forecasting for time-series forecasting model
    """

    output_col = [0]

    # Read data
    data_file = "./data/toy_time_series/sine.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    linear_space = np.linspace(0, 2, num=len(df_raw))
    df_raw = df_raw.add(linear_space, axis=0)
    data_file_time = "./data/toy_time_series/sine_datetime.csv"
    time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(time_series[0])
    df_raw.index = time_series

    # Data processing
    data_processor = DataProcess(
        data=df_raw,
        time_covariates=["hour_of_day"],
        train_split=0.8,
        validation_split=0.2,
        output_col=output_col,
    )
    train_data, validation_data, _, _ = data_processor.get_splits()

    # Initialize model
    model.auto_initialize_baseline_states(train_data["y"][0 : 24 * 2])

    for _ in range(5):
        (mu_validation_preds, std_validation_preds, _) = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
            white_noise_decay=False,
        )

        # Unstandardize
        mu_validation_preds = normalizer.unstandardize(
            mu_validation_preds,
            data_processor.norm_const_mean[output_col],
            data_processor.norm_const_std[output_col],
        )
        std_validation_preds = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor.norm_const_std[output_col],
        )

        # Calculate the log-likelihood metric
        validation_obs = data_processor.get_data("validation").flatten()
        mse = metric.mse(mu_validation_preds, validation_obs)

        # Early-stopping
        model.early_stopping(evaluate_metric=mse, mode="min")
        if model.stop_training:
            break

    # Validation metric
    validation_obs = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_validation_preds, validation_obs)

    if plot:
        plot_data(
            data_processor=data_processor,
            normalization=False,
            plot_column=output_col,
        )
        plot_prediction(
            data_processor=data_processor,
            mean_validation_pred=mu_validation_preds,
            std_validation_pred=std_validation_preds,
        )
        plt.show()

    return mse


def test_model_forecast(run_mode, plot_mode):
    """Test model forecastin with lstm component"""
    # Model
    model = Model(
        LocalTrend(),
        LstmNetwork(
            look_back_len=19,
            num_features=2,
            num_layer=1,
            num_hidden_unit=50,
            device="cpu",
            manual_seed=1,
        ),
        WhiteNoise(std_error=1e-3),
    )
    mse = model_test_runner(model, plot=plot_mode)

    path_metric = "test/saved_metric/test_model_forecast_metric.csv"
    if run_mode == "save_threshold":
        pd.DataFrame({"mse": [mse]}).to_csv(path_metric, index=False)
        print(f"Saved MSE to {path_metric}: {mse}")
    else:
        # load threshold
        threshold = None
        if os.path.exists(path_metric):
            df = pd.read_csv(path_metric)
            threshold = float(df["mse"].iloc[0])

        assert (
            threshold is not None
        ), "No saved threshold found. Run with --mode=save_threshold first to save a threshold."
        assert (
            abs(mse - threshold) < 1e-6
        ), f"MSE {mse} not within tolerance of saved threshold {threshold}"
