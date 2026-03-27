import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari import (
    DataProcess,
    Model,
    plot_states,
    plot_data, 
    plot_prediction,
)
from canari.component import LocalTrend, KernelRegression, WhiteNoise

# Read data
data_file = "./data/toy_time_series/exp_sine_dependency.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
df_raw = df_raw[[0]]
df_raw.columns = ["exp sine"]

data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"


# Split into train and test
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    train_split=0.8,
    validation_split=0.2,
    output_col=output_col,
    standardization=False,
)
train_data, validation_data, _, all_data = data_processor.get_splits()

# Components

model = Model(
    LocalTrend(),
    KernelRegression(period=24,
                    kernel_length=0.8,
                    num_control_point=10,
                    mu_control_point = 0.1,
                    var_control_point = 0.1
                    ),
    WhiteNoise(std_error=1e-2)
)
model.auto_initialize_baseline_states(train_data["y"])

mu_obs_preds,std_obs_preds,_ = model.filter(data=train_data)
model.forecast(data=validation_data)
model.smoother()

#  Plot
fig, axes=plot_states(
    data_processor=data_processor,
    states=model.states,
    states_to_plot=["level", "trend", "kernel regression", "white noise"],
    standardization=False,
    states_type="smooth"
)
plot_data(
    data_processor=data_processor,
    standardization=False,
    plot_column=output_col,
    sub_plot=axes[0],
)
plt.show()
