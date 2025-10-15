import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytagi import Normalizer as normalizer
import pytagi.metric as metric
from canari import DataProcess, Model, plot_data, plot_prediction
from canari.component import LocalTrend, LstmNetwork, WhiteNoise
import fire

BASE_DIR = os.path.dirname(__file__)


def model_test_runner(model: Model, plot: bool) -> float:
    """
    Run training and forecasting model
    """

    output_col = [0]

    # Read data
    data_file = os.path.join(BASE_DIR, "../data/toy_time_series/exp_sine_agvi.csv")
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    data_file_time = os.path.join(BASE_DIR, "../data/toy_time_series/sine_datetime.csv")
    time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(time_series[0])
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["values"]

    # Data processing
    data_processor = DataProcess(
        data=df_raw,
        train_split=0.8,
        validation_split=0.2,
        output_col=output_col,
    )
    train_data, validation_data, _, _ = data_processor.get_splits()

    # Initialize model
    model.auto_initialize_baseline_states(train_data["y"][0:24])
    num_epoch = 50
    for epoch in range(num_epoch):
        (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
        )

        # Unstandardize
        mu_validation_preds = normalizer.unstandardize(
            mu_validation_preds,
            data_processor.scale_const_mean[output_col],
            data_processor.scale_const_std[output_col],
        )
        std_validation_preds = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor.scale_const_std[output_col],
        )

        # Calculate the log-likelihood metric
        validation_obs = data_processor.get_data("validation").flatten()
        mse = metric.mse(mu_validation_preds, validation_obs)

        # Early-stopping
        model.early_stopping(
            evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch
        )
        if epoch == model.optimal_epoch:
            mu_validation_preds_optim = mu_validation_preds

        if model.stop_training:
            break

    # Validation metric
    validation_obs = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_validation_preds_optim, validation_obs)

    if plot:
        plot_data(
            data_processor=data_processor,
            standardization=False,
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
    """Test model forecasting"""
    # Model
    model = Model(
        LocalTrend(),
        LstmNetwork(
            look_back_len=24,
            num_features=1,
            num_layer=1,
            num_hidden_unit=50,
            device="cpu",
            manual_seed=1,
            model_noise=True,
            smoother=False,
        ),
    )
    mse = model_test_runner(model, plot=plot_mode)

    path_metric = os.path.join(
        BASE_DIR, "../test/saved_metric/test_hetero_noise_metric.csv"
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
