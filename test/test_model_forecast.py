import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytagi import Normalizer as normalizer
import pytagi.metric as metric
from canari import DataProcess, Model, plot_data, plot_prediction
from canari.component import LocalTrend, LstmNetwork, WhiteNoise
import pytest

BASE_DIR = os.path.dirname(__file__)


def model_test_runner(model: Model, plot: bool) -> float:
    """
    Run training and forecasting for time-series forecasting model
    """

    output_col = [0]

    # Read data
    data_file = os.path.join(BASE_DIR, "../data/toy_time_series/sine.csv")
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    linear_space = np.linspace(0, 2, num=len(df_raw))
    df_raw = df_raw.add(linear_space, axis=0)
    data_file_time = os.path.join(BASE_DIR, "../data/toy_time_series/sine_datetime.csv")
    time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(time_series[0])
    df_raw.index = time_series

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


@pytest.mark.parametrize("smoother", [(False), (True)], ids=["LSTM", "SLSTM"])
def test_model_forecast(run_mode, plot_mode, smoother):
    """Test model forecasting with LSTM and SLSTM"""
    model = Model(
        LocalTrend(),
        LstmNetwork(
            look_back_len=19,
            num_features=1,
            num_layer=1,
            infer_len=24,
            num_hidden_unit=50,
            device="cpu",
            manual_seed=1,
            smoother=smoother,
        ),
        WhiteNoise(std_error=0.0032322250444898116),
    )
    mse = model_test_runner(model, plot=plot_mode)
    mse = round(mse, 10)

    path_metric = os.path.join(
        BASE_DIR, "../test/saved_metric/test_model_forecast_metric.csv"
    )
    if run_mode == "save_threshold":
        target_column = "SLSTM" if smoother else "LSTM"
        columns = ["LSTM", "SLSTM"]

        if os.path.exists(path_metric):
            df = pd.read_csv(path_metric)
        else:
            df = pd.DataFrame()

        for column in columns:
            if column not in df.columns:
                df[column] = np.nan

        df = df[columns]

        if df.empty:
            df.loc[0] = {column: np.nan for column in columns}

        df.loc[0, target_column] = mse
        df.to_csv(path_metric, index=False)
        print(f"Saved MSE to {path_metric}: {mse}")
    else:
        # load threshold
        threshold = None
        if os.path.exists(path_metric):
            df = pd.read_csv(path_metric)
            target_column = "SLSTM" if smoother else "LSTM"
            if target_column in df.columns:
                value_series = df[target_column].dropna()
                if not value_series.empty:
                    threshold = float(value_series.iloc[0])

        assert (
            threshold is not None
        ), "No saved threshold found. Run with --mode=save_threshold first to save a threshold."
        assert mse <= threshold or np.isclose(
            mse, threshold, rtol=1e-8
        ), f"MSE {mse} is not close enough to the saved threshold {threshold}"
