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

# Read data
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
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    train_split=0.8,
    validation_split=0.2,
    output_col=output_col,
    standardization=False,
)
train_data, validation_data, _, _ = data_processor.get_splits()

# Components
sigma_v = np.sqrt(1e-6)
local_trend = LocalTrend(mu_states=[5, 0.0], var_states=[1e-1, 1e-6], std_error=0)
periodic = Periodic(period=52, mu_states=[5 * 5, 0], var_states=[1e-12, 1e-12])
# Gamma can be changed to see how it constrains the autoregression differently
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
    bar
)

mu_obs_preds,std_obs_preds,_ = model.filter(data=train_data)
model.smoother()

#  Plot
fig, axes=plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="smooth",
)
fig.suptitle("Smoother States")
fig, axes=plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="posterior",
)
fig.suptitle("Posterior States")
fig, axes=plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="prior",
)
fig.suptitle("Prior States")
plt.show()
