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
    plot_prediction,
    plot_skf_states,
)
from examples import DataProcess

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
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    train_split=1,
    output_col=output_col,
    normalization=False,
)
_, _, _, all_data = data_processor.get_splits()

# Components
sigma_v = np.sqrt(1e-6)
local_trend = LocalTrend(mu_states=[5, 0.0], var_states=[1e-12, 1e-12], std_error=0)
local_acceleration = LocalAcceleration(
    mu_states=[5, 0.0, 0.0], var_states=[1e-12, 1e-4, 1e-4], std_error=1e-3
)
periodic = Periodic(period=52, mu_states=[5 * 5, 0], var_states=[1e-12, 1e-12])
# AR = Autoregression(std_error=5, phi=0.9, mu_states=[-0.0621], var_states=[6.36e-05])
# AR = Autoregression(std_error=5, mu_states=[0.5, -0.0621], var_states=[0.25, 6.36e-05])
AR = Autoregression(mu_states=[0.5, -0.0621], var_states=[0.25, 6.36e-05])
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
marginal_abnorm_prob_plot = filter_marginal_abnorm_prob
fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=states,
    model_prob=marginal_abnorm_prob_plot,
    color="b",
)
fig.suptitle("SKF hidden states", fontsize=10, y=1)
plt.show()
