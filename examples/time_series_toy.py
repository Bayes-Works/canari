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
    plot_with_uncertainty,
)
from examples import DataProcess
from pytagi import exponential_scheduler


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
num_epoch = 50

data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day", "day_of_week"],
    train_start="2000-01-01 00:00:00",
    train_end="2000-01-09 23:00:00",
    validation_start="2000-01-10 00:00:00",
    validation_end="2000-01-11 23:00:00",
    test_start="2000-01-11 23:00:00",
    output_col=output_col,
)
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
sigma_v = 5e-2
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=10,
        num_features=3,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
    ),
    Autoregression(),
    WhiteNoise(std_error=sigma_v),
)
model.auto_initialize_baseline_states(train_data["y"][1:24])

# Training
scheduled_sigma_v = 2
for epoch in range(num_epoch):
    # Decaying observation's variance
    scheduled_sigma_v = exponential_scheduler(
        curr_v=scheduled_sigma_v, min_v=sigma_v, decaying_factor=0.99, curr_iter=epoch
    )
    noise_index = model.states_name.index("white noise")
    model.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

    (mu_validation_preds, std_validation_preds, smoother_states) = model.lstm_train(
        train_data=train_data, validation_data=validation_data
    )

#  Plot
plt.plot(data_processor.train_time, train_data["y"], color="r", label="train_obs")
plt.plot(
    data_processor.validation_time,
    validation_data["y"],
    color="r",
    linestyle="--",
    label="validation_obs",
)
plot_with_uncertainty(
    time=data_processor.validation_time,
    mu=mu_validation_preds,
    std=std_validation_preds,
    color="b",
    label=["mu_val_pred", "±1σ"],
)
plt.legend()
plt.show()
