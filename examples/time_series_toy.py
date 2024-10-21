import pandas as pd
from src import (
    LocalTrend,
    LstmNetwork,
    Autoregression,
    WhiteNoise,
    Model,
)
from examples import DataProcess


# # Read data
data_file = "./data/toy_time_series/sine_trend.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/sine_trend_datetime.csv"
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
    train_end="2000-01-13 21:00:00",
    validation_start="2000-01-13 22:00:00",
    validation_end="2000-01-14 21:00:00",
    test_start="2000-01-14 22:00:00",
)
train_data, validation_data, test_data = data_processor.get_splits()

# Define parameters
output_col = [0]

# Model
model = Model(
    LocalTrend(std_error=0.1, mu_states=[0.1, 0.1], var_states=[0.1, 0.1]),
    LstmNetwork(
        look_back_len=10,
        num_features=3,
        num_layer=2,
        num_hidden_unit=50,
    ),
    Autoregression(std_error=0.1, mu_states=[0.1], var_states=[0.1]),
    WhiteNoise(std_error=0.1),
)

model.lstm_train(train_data=train_data, validation_data=validation_data, num_epoch=50)

check = 1
