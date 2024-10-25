import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import (
    LocalTrend,
    LstmNetwork,
    Periodic,
    Autoregression,
    WhiteNoise,
    Model,
)
from examples import DataProcess
import time


# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

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
    train_end="2000-01-08 21:00:00",
    validation_start="2000-01-08 22:00:00",
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
        num_layer=2,
        num_hidden_unit=50,
    ),
    Autoregression(),
    WhiteNoise(std_error=0.1),
)

# Training
for epoch in range(num_epoch):
    (mu_validation_preds, var_validation_preds) = model.lstm_train(
        train_data=train_data, validation_data=validation_data
    )


#  Plot
idx_train = range(0, len(train_data["y"]))
idx_val = range(len(train_data["y"]), len(validation_data["y"]) + len(train_data["y"]))

plt.figure(figsize=(10, 6))
plt.plot(idx_train, train_data["y"], color="r", label="train_obs")
plt.plot(
    idx_val, validation_data["y"], color="r", linestyle="--", label="validation_obs"
)
plt.plot(idx_val, mu_validation_preds, color="b", label="mu_pred")
std = np.sqrt(var_validation_preds)
plt.fill_between(
    idx_val,
    mu_validation_preds - std,
    mu_validation_preds + std,
    color="blue",
    alpha=0.2,
    label="±1 std",
)
plt.legend()
plt.show()
