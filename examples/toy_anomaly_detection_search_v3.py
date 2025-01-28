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

# Component
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=trial.suggest_int("look_back_len", 12, 48),
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
)
noise = WhiteNoise(std_error=trial.suggest_loguniform("sigma_v", 1e-3, 1e-1))

# Model
model = Model(local_trend, lstm_network, noise)
model_optimizer = ModelOptimizer(model)
optimal_model = model_optimizer.search()


# SKF
ab_model = Model(local_acceleration, lstm_network, noise)
skf = SKF(
    norm_model=optimal_model,
    abnorm_model=ab_model,
    std_transition_error="search",
    norm_to_abnorm_prob="search",
    abnorm_to_norm_prob=1e-1,  # Fixed value
    norm_model_prior_prob=0.99,  # Fixed value
)
skf_model_optimizer = SKFOptimizer(skf)
optimal_skf = skf_model_optimizer.search()
