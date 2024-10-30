import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import (
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    Periodic,
    Autoregression,
    WhiteNoise,
    Model,
    PlotStates,
)
from examples import DataProcess
import time


# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
linear_space = np.linspace(0, 2, num=len(df_raw))
df_raw = df_raw.add(linear_space, axis=0)


data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Resampling data
df = df_raw.resample("H").mean()

# Define parameters
output_col = [0]
num_epoch = 20

data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day", "day_of_week"],
    train_start="2000-01-01 00:00:00",
    train_end="2000-01-07 21:00:00",
    validation_start="2000-01-07 22:00:00",
    validation_end="2000-01-09 23:00:00",
    output_col=output_col,
)
train_data, validation_data, test_data = data_processor.get_splits()

# Model
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=24,
        num_features=3,
        num_layer=4,
        num_hidden_unit=50,
        device="cpu",
    ),
    Autoregression(),
    WhiteNoise(std_error=1e-3),
)
model.auto_initialize_baseline_states(train_data["y"])

# Training
for epoch in range(num_epoch):
    (mu_validation_preds, var_validation_preds, smoother_states) = model.lstm_train(
        train_data=train_data, validation_data=validation_data
    )

    # mu_smooths = smoother_states.mu_smooths
    # var_smooths = smoother_states.var_smooths
    mu_smooths = smoother_states.mu_posteriors[:-1, :]
    var_smooths = smoother_states.var_posteriors[:-1, :, :]
    # mu_smooths = smoother_states.mu_smooths[:-1, :]
    # var_smooths = smoother_states.var_smooths[:-1, :, :]

    # #  Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data_processor.train_time, train_data["y"], color="r", label="train_obs")
    plt.plot(
        data_processor.validation_time,
        validation_data["y"],
        color="r",
        linestyle="--",
        label="validation_obs",
    )
    PlotStates(
        time=data_processor.train_time,
        mu=mu_smooths,
        var=var_smooths,
        index=0,
        color="k",
    )
    PlotStates(
        time=data_processor.validation_time,
        mu=mu_validation_preds,
        var=var_validation_preds,
        color="b",
    )
    plt.legend()
    filename = f"saved_results/smoother#{epoch}.png"
    plt.savefig(filename)
    plt.close()
