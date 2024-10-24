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

data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day", "day_of_week"],
    train_start="2000-01-01 00:00:00",
    train_end="2000-01-08 21:00:00",
    validation_start="2000-01-08 22:00:00",
    validation_end="2000-01-09 23:00:00",
)
train_data, validation_data, test_data = data_processor.get_splits()

# Define parameters
output_col = [0]
num_epoch = 10

# Model
model = Model(
    LocalTrend(std_error=0, mu_states=[0, 0], var_states=[1e-6, 1e-6]),
    LstmNetwork(
        look_back_len=24,
        num_features=3,
        num_layer=1,
        num_hidden_unit=50,
    ),
    # Autoregression(std_error=0.0, mu_states=[0], var_states=[0.00]),
    WhiteNoise(std_error=0.1),
)

for epoch in range(num_epoch):
    (mu_validation_preds, var_validation_preds) = model.lstm_train(
        train_data=train_data, validation_data=validation_data
    )


# mu_validation_preds, var_validation_preds = model.lstm_train(
#     train_data=train_data, validation_data=validation_data, num_epoch=num_epoch
# )

idx_train = range(0, len(train_data))
idx_val = range(len(train_data), len(validation_data) + len(train_data))

plt.figure(figsize=(10, 6))
plt.plot(idx_train, train_data[:, 0], color="r")
plt.plot(idx_val, validation_data[:, 0], color="b")
plt.plot(idx_val, mu_validation_preds, color="k")
plt.show()


check = 1
