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
    plot_data,
    plot_prediction,
    plot_states,
)
from examples import DataProcess


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
local_trend = LocalTrend(mu_states=[5, 0.0], var_states=[1e-12, 1e-12], std_error=0)
local_acceleration = LocalAcceleration(
    mu_states=[5, 0.0, 0.0], var_states=[1e-12, 1e-2, 1e-2], std_error=1e-2
)
periodic = Periodic(period=52, mu_states=[5 * 5, 0], var_states=[1e-12, 1e-12])
# AR = Autoregression(std_error=5, phi=0.9, mu_states=[-0.0621], var_states=[6.36e-05])
# AR = Autoregression(phi=0.9, mu_states=[-0.0621, 0], var_states=[6.36e-05, 1000], var_W2bar=5000)
# AR = Autoregression(std_error=5, mu_states=[0.5, -0.0621], var_states=[0.25, 6.36e-05])
AR = Autoregression(mu_states=[0.5, -0.0621, 0], var_states=[0.25, 6.36e-05, 100], var_W2bar=100)
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
mu_validation_preds, std_validation_preds = model.forecast(validation_data)

#  Plot

plot_data(
    data_processor=data_processor,
    plot_column=output_col,
    validation_label="y",
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds,
    std_validation_pred=std_validation_preds,
    validation_label=[r"$\mu$", f"$\pm\sigma$"],
)
plt.legend(loc="upper left")  # Change "upper right" to your desired location
# for i in range(len(model.states.var_prior)):
#     print(model.states.var_prior[i][model.ar_error_index, model.ar_error_index])
plot_states(data_processor=data_processor,
            states=model.states,
            states_type='prior',
            )
plt.show()
