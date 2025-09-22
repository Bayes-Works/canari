import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_skf_states,
)
from canari.component import (
    LocalTrend,
    LocalAcceleration,
    WhiteNoise,
    Periodic,
    Autoregression,
    BoundedAutoregression,
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
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    train_split=1,
    output_col=output_col,
    standardization=False,
)
_, _, _, all_data = data_processor.get_splits()


# Components
local_trend = LocalTrend(mu_states=[5, 0.0], var_states=[1e-1, 1e-6])
local_acceleration = LocalAcceleration(
    mu_states=[5, 0.0, 0.0], var_states=[1e-1, 1e-6, 1e-3]
)
periodic = Periodic(period=52, mu_states=[5 * 5, 0], var_states=[1e-12, 1e-12])
# Gamma can be changed to see how it affects the anomaly detection
bar = BoundedAutoregression(
    std_error=5,
    phi=0.9,
    mu_states=[-0.0621, -0.0621],
    var_states=[6.36e-05, 6.36e-05],
    gamma=1.5e0,
)

# Normal model
model = Model(
    local_trend,
    periodic,
    bar,
)

#  Abnormal model
ab_model = Model(
    local_acceleration,
    periodic,
    bar,
)

# Switching Kalman filter
skf = SKF(
    norm_model=model,
    abnorm_model=ab_model,
    std_transition_error=1e-3,
    norm_to_abnorm_prob=1e-4,
)

# Anomaly Detection
filter_marginal_abnorm_prob, states = skf.filter(data=all_data)
smooth_marginal_abnorm_prob, states = skf.smoother()

#  Plot
marginal_abnorm_prob_plot = filter_marginal_abnorm_prob
fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=states,
    states_type="prior",
    model_prob=marginal_abnorm_prob_plot,
    color="b",
)
fig.suptitle("SKF hidden states", fontsize=10, y=1)
time = data_processor.get_time(split="all")
ax[0].axvline(x=time[time_anomaly], color="r", linestyle="--")
ax[-1].axvline(x=time[time_anomaly], color="r", linestyle="--")
plt.show()
