import sys
import os
import unittest
import pandas as pd
import numpy as np
from pytagi import Normalizer as normalizer
import pytagi.metric as metric
from src import (
    LocalTrend,
    LstmNetwork,
    Autoregression,
    WhiteNoise,
    Model,
)
from examples import DataProcess


def SKF_anomaly_detection_runner(
    model: Model,
) -> float:
    """Run training for time-series forecasting model"""

    # Define parameters
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
        train_start="2000-01-01 00:00:00",
        train_end="2000-01-09 23:00:00",
        validation_start="2000-01-10 00:00:00",
        validation_end="2000-01-11 23:00:00",
        test_start="2000-01-11 23:00:00",
        output_col=output_col,
    )
    train_data, validation_data, _, _ = data_processor.get_splits()

    # -------------------------------------------------------------------------#
    # Training:
    model.auto_initialize_baseline_states(train_data["y"][1:24])

    for _ in range(2):
        (mu_validation_preds, std_validation_preds, _) = model.lstm_train(
            train_data=train_data, validation_data=validation_data
        )

        # Validation metric
        mu_validation_preds = normalizer.unstandardize(
            mu_validation_preds,
            data_processor.data_mean[output_col],
            data_processor.data_std[output_col],
        )
        std_validation_preds = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor.data_std[output_col],
        )

    mse = metric.mse(
        mu_validation_preds, data_processor.validation_data[:, output_col].flatten()
    )

    return mse


class SineSignalWithTrendTest(unittest.TestCase):

    def setUp(self):
        # Mode 1: Save the threshold
        self.mode = None
        # self.mode = "save_threshold"
        for arg in sys.argv:
            if arg.startswith("mode="):
                self.mode = arg.split("=")[1]
                break

        # Mode 2: Load the threshold from CSV
        path_metric = "test/saved_metric/test_model_forescast_metric.csv"
        if self.mode != "save_threshold":
            if os.path.exists(path_metric):
                df = pd.read_csv(path_metric)
                self.threshold = float(df["mse"].iloc[0])

    def test_model(self):
        # Model
        std_observation_noise = 1e-1
        model = Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=12,
                num_features=1,
                num_layer=1,
                num_hidden_unit=50,
                device="cpu",
                manual_seed=1,
            ),
            Autoregression(),
            WhiteNoise(std_error=std_observation_noise),
        )
        mse = model_test_runner(model)

        if self.mode == "save_threshold":
            pd.DataFrame({"mse": [mse]}).to_csv(
                "test/saved_metric/test_model_forescast_metric.csv", index=False
            )
            self.threshold = mse
            print(
                f"Saved MSE to test/saved_metric/test_model_forescast_metric.csv: {mse}"
            )

        # Run the assertion
        self.assertAlmostEqual(mse, self.threshold, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
