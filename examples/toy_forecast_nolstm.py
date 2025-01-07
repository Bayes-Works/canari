import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import (
    LocalTrend,
    LocalAcceleration,
    Periodic,
    Autoregression,
    WhiteNoise,
    Model,
    plot_with_uncertainty,
)


# # Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Data pre-processing
all_data = {}
all_data["y"] = df_raw.values

# Split into train and test
train_split = 0.8

train_data = {}
train_data["y"] = df_raw.values[: int(len(df_raw) * train_split)]
train_time = df_raw.index[: int(len(df_raw) * train_split)]
test_data = {}
test_data["y"] = df_raw.values[int(len(df_raw) * train_split) :]
test_time = df_raw.index[int(len(df_raw) * train_split) :]

# Components
sigma_v = np.sqrt(1e-6)
local_trend = LocalTrend(mu_states=[5, 0.0], var_states=[1e-12, 1e-12], std_error=0)
local_acceleration = LocalAcceleration(
    mu_states=[5, 0.0, 0.0], var_states=[1e-12, 1e-2, 1e-2], std_error=1e-2
)
periodic = Periodic(period=52, mu_states=[5 * 5, 0], var_states=[1e-12, 1e-12])
AR = Autoregression(std_error=5, phi=0.9, mu_states=[-0.0621], var_states=[6.36e-05])
noise = WhiteNoise(std_error=sigma_v)

# Normal model
model = Model(
    local_trend,
    periodic,
    AR,
    noise,
)

# # #
model.filter(data=train_data)
model.smoother(data=train_data)
mu_test_preds, std_test_preds = model.forecast(test_data)

#  Plot
mu_plot = model.states.mu_posterior
var_plot = model.states.var_posterior


plt.plot(df_raw.index, all_data["y"], color="r", label="obs")
plt.ylabel("y")
plot_with_uncertainty(
    time=test_time,
    mu=mu_test_preds,
    std=std_test_preds,
    color="g",
    label=["mu_test_pred", "±1σ"],
)
plt.legend()
plt.show()
