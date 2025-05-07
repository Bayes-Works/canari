import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari import (
    DataProcess,
    Model,
    plot_states,
)
from canari.component import LocalTrend, Periodic, WhiteNoise, Autoregression, BoundedAutoregression

# # Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# # Add synthetic anomaly to data
# time_anomaly = 400
# AR_stationary_var = 5**2 / (1 - 0.9**2)
# anomaly_magnitude = -(np.sqrt(AR_stationary_var) * 1) / 50
# for i in range(time_anomaly, len(df_raw)):
#     df_raw.values[i] += anomaly_magnitude * (i - time_anomaly)

# Data pre-processing
all_data = {}
all_data["y"] = df_raw.values

# Split into train and test
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    train_split=0.8,
    validation_split=0.2,
    output_col=output_col,
    normalization=False,
)
train_data, validation_data, _, _ = data_processor.get_splits()

# Components
sigma_v = np.sqrt(1e-6)
local_trend = LocalTrend(mu_states=[5, 0.0], var_states=[1e-1, 1e-6], std_error=0)
periodic = Periodic(period=52, mu_states=[5 * 5, 0], var_states=[1e-12, 1e-12])

# Different cases for ar components
# Case 1: regular ar, with process error and phi provided
ar = Autoregression(
    std_error=5, phi=0.9, mu_states=[-0.0621], var_states=[6.36e-05]
)

bar = BoundedAutoregression(
    std_error=5, 
    phi=0.9,
    mu_states=[-0.0621, -0.0621],
    var_states=[6.36e-05, 6.36e-05],
    gamma=2,
)

# Normal model
model = Model(
    local_trend,
    periodic,
    # ar,
    bar
)

# # #
model.filter(data=train_data)
model.smoother(data=train_data)

# #  Plot
plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="prior",
)
plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="posterior",
)
plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="smooth",
)
plt.show()
