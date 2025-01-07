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
    SKF,
    plot_with_uncertainty,
)

# Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Add synthetic anomaly to data
time_anomaly = 400
AR_stationary_var = 5**2 / (1 - 0.9**2)
anomaly_magnitude = -(np.sqrt(AR_stationary_var) * 1) / 50
for i in range(time_anomaly, len(df_raw)):
    df_raw.values[i] += anomaly_magnitude * (i - time_anomaly)

# Data pre-processing
all_data = {}
all_data["y"] = df_raw.values

# Components
sigma_v = np.sqrt(1e-6)
local_trend = LocalTrend(mu_states=[5, 0.0], var_states=[1e-12, 1e-12], std_error=0)
local_acceleration = LocalAcceleration(
    mu_states=[5, 0.0, 0.0], var_states=[1e-12, 1e-4, 1e-4], std_error=1e-3
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

#  Abnormal model
ab_model = Model(
    local_acceleration,
    periodic,
    AR,
    noise,
)

# Switching Kalman filter
skf = SKF(
    norm_model=model,
    abnorm_model=ab_model,
    std_transition_error=1e-3,
    norm_to_abnorm_prob=1e-4,
    abnorm_to_norm_prob=1e-4,
    norm_model_prior_prob=0.99,
)

# # # Anomaly Detection
filter_marginal_abnorm_prob, _ = skf.filter(data=all_data)
smooth_marginal_abnorm_prob, states = skf.smoother(data=all_data)

#  Plot
mu_plot = states.mu_posterior
var_plot = states.var_posterior
marginal_abnorm_prob_plot = filter_marginal_abnorm_prob


#  Plot hidden states
local_level_index = skf.states_name.index("local level")
local_trend_index = skf.states_name.index("local trend")
periodic_index = skf.states_name.index("periodic 1")
ar_index = skf.states_name.index("autoregression")

fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=False)
axs[0].plot(df_raw.index, all_data["y"], color="r", label="obs")
axs[0].set_title("Observed Data")
axs[0].set_ylabel("y")
plot_with_uncertainty(
    time=df_raw.index,
    mu=mu_plot[:, local_level_index],
    std=var_plot[:, local_level_index, local_level_index] ** 0.5,
    color="b",
    label=["mu_val_pred", "±1σ"],
    ax=axs[0],
)
axs[0].set_ylabel("Local level")
axs[0].axvline(x=df_raw.index[time_anomaly], color="r", linestyle="--", label="anomaly")


plot_with_uncertainty(
    time=df_raw.index,
    mu=mu_plot[:, local_trend_index],
    std=var_plot[:, local_trend_index, local_trend_index] ** 0.5,
    color="b",
    label=["mu_val_pred", "±1σ"],
    ax=axs[1],
)
axs[1].axhline(y=0, color="black", linestyle="--", label="zero speed")
axs[1].legend()
axs[1].set_ylabel("Local trend")

plot_with_uncertainty(
    time=df_raw.index,
    mu=mu_plot[:, periodic_index],
    std=var_plot[:, periodic_index, periodic_index] ** 0.5,
    color="b",
    label=["mu_val_pred", "±1σ"],
    ax=axs[2],
)
axs[2].set_ylabel("Periodic")

plot_with_uncertainty(
    time=df_raw.index,
    mu=mu_plot[:, ar_index],
    std=var_plot[:, ar_index, ar_index] ** 0.5,
    color="b",
    label=["mu_val_pred", "±1σ"],
    ax=axs[3],
)
axs[3].set_ylabel("autoregression")

axs[4].plot(df_raw.index, marginal_abnorm_prob_plot, color="b")
axs[4].set_xlabel("Time")
axs[4].set_ylabel("Pr(Abnormal)")
axs[4].set_ylim([0, 1])

plt.tight_layout()
plt.show()
