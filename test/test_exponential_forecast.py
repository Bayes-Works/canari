import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytagi import Normalizer as normalizer
import pytagi.metric as metric
from canari import DataProcess, Model, plot_data, plot_prediction
from canari.component import LocalTrend, WhiteNoise, Exponential, Periodic
import fire

BASE_DIR = os.path.dirname(__file__)


def model_test_runner(model: Model, plot: bool) -> float:

    output_col = [0]

    # Read data
    data_file = os.path.join(
        BASE_DIR,
        "../data/toy_time_series/synthetic_exponential_localtrend_periodic.csv",
    )
    df_raw = pd.read_csv(data_file, sep=";", parse_dates=["temps"], index_col="temps")
    df = df_raw[["exponential"]]
    df = df[:100]

    # Data processing
    data_processor = DataProcess(
        data=df,
        train_split=0.8,
        validation_split=0.2,
        output_col=output_col,
        standardization=False,
    )

    train_data, validation_data, _, _ = data_processor.get_splits()
    mu_train_pred, std_train_pred, states = model.filter(data=train_data)
    mu_val_pred, std_val_pred, states = model.forecast(data=validation_data)

    validation_obs = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_val_pred, validation_obs)

    if plot:
        plot_data(
            data_processor=data_processor,
            standardization=False,
            plot_column=output_col,
        )
        plot_prediction(
            data_processor=data_processor,
            mean_validation_pred=mu_val_pred,
            std_validation_pred=std_val_pred,
        )
        plt.show()

    return mse


def test_model_forecast(run_mode, plot_mode):

    # Components
    sigma_v = np.sqrt(0.1)
    exponential = Exponential(
        mu_states=[0, 0.013, 9.7, 0, 0],
        var_states=[0.00001**2, 0.005**2, 0.3**2, 0, 0],
    )
    noise = WhiteNoise(std_error=sigma_v)
    localtrend = LocalTrend(
        mu_states=[1.95, -0.00], var_states=[0.1**2, 0.0075**2], std_error=0
    )
    periodic = Periodic(mu_states=[1.9, 1.9], var_states=[0.1**2, 0.1**2], period=52)

    # Model
    model = Model(exponential, noise, periodic, localtrend)
    mse = model_test_runner(model, plot=plot_mode)
    print(f"Mean Squared Error: {mse}")

    path_metric = os.path.join(
        BASE_DIR, "../test/saved_metric/test_exponential_forecast_metric.csv"
    )

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
            mse < threshold
        ), f"MSE {mse} is smaller than the saved threshold {threshold}"
