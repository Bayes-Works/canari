import optuna
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


# # Read data
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
train_data, validation_data, test_data, all_data = data_processor.get_splits()


class ModelOptimizer:
    def __init__(self, model, params, data_processor, train_data, validation_data):
        self.model = model
        self.data_processor = data_processor
        self.train_data = train_data
        self.validation_data = validation_data
        self.best_params = None
        self.params = params

    def search(self, n_trials=20):
        """
        Optimize the model using Optuna.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=n_trials)

        self.best_params = study.best_params
        print("Best parameters:", self.best_params)
        print("Best validation MSE:", study.best_value)

        # Re-initialize the model with the best parameters
        optimal_model = self._initialize_model(self.best_params)
        return optimal_model

    def _objective(self, trial):
        """
        Objective function for Optuna optimization.
        """
        # Suggest hyperparameters
        sigma_v = trial.suggest_loguniform(params["sigma_v"], 1e-3, 1e-1)
        look_back_len = trial.suggest_int(params["look_back_len"], 12, 48)

        # Initialize the model
        model = self._initialize_model(
            {"sigma_v": sigma_v, "look_back_len": look_back_len}
        )
        model.auto_initialize_baseline_states(self.train_data["y"][1:24])

        # Training process
        scheduled_sigma_v = 1
        num_epoch = 50
        for epoch in range(num_epoch):
            scheduled_sigma_v = exponential_scheduler(
                curr_v=scheduled_sigma_v,
                min_v=sigma_v,
                decaying_factor=0.99,
                curr_iter=epoch,
            )
            noise_index = model.states_name.index("white noise")
            model.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

            mu_validation_preds, _, _ = model.lstm_train(
                train_data=self.train_data,
                validation_data=self.validation_data,
            )

            # Unstandardize predictions
            mu_validation_preds = normalizer.unstandardize(
                mu_validation_preds,
                self.data_processor.norm_const_mean[0],
                self.data_processor.norm_const_std[0],
            )

            mse = metric.mse(
                mu_validation_preds,
                self.data_processor.validation_data[:, 0].flatten(),
            )

            model.early_stopping(evaluate_metric=mse, mode="min")
            if model.stop_training:
                break

        return model.early_stop_metric

    def _initialize_model(self):
        """
        Initialize the model with given parameters.
        """
        sigma_v = params.get("sigma_v", 1e-2)
        look_back_len = params.get("look_back_len", 24)

        lstm_network = LstmNetwork(
            look_back_len=look_back_len,
            num_features=2,
            num_layer=1,
            num_hidden_unit=50,
            device="cpu",
        )

        noise = WhiteNoise(std_error=sigma_v)

        return Model(LocalTrend(), lstm_network, noise)


# Example usage:
# Assuming you have already initialized data_processor, train_data, and validation_data

params = {
    "sigma_v": [1e-3, 1e-1],
    "look_back_len": [12, 48],
}

local_trend = LocalTrend()
lstm_network = LstmNetwork(
    look_back_len=params["look_back_len"],
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
)
noise = WhiteNoise(
    std_error=params["sigma_v"],
)

# Initialize model
model = Model(local_trend, lstm_network, noise)
# Initialize ModelOptimizer
model_optimizer = ModelOptimizer(
    model, params, data_processor, train_data, validation_data
)

# Search for optimal parameters
optimal_model = model_optimizer.search(n_trials=20)
