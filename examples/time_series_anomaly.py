import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src import (
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    WhiteNoise,
    Model,
    plot_with_uncertainty,
    SKF,
)
from examples import DataProcess
import time


# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
linear_space = np.linspace(0, 1, num=len(df_raw))
point_anomaly = 110
trend = np.linspace(0, 1, num=len(df_raw) - point_anomaly)
linear_space[point_anomaly:] = linear_space[point_anomaly:] + trend
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
num_epoch = 100

data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day", "day_of_week"],
    train_start="2000-01-01 00:00:00",
    train_end="2000-01-06 20:00:00",
    validation_start="2000-01-06 21:00:00",
    validation_end="2000-01-07 22:00:00",
    test_start="2000-01-07 23:00:00",
    output_col=output_col,
)
train_data, validation_data, test_data = data_processor.get_splits()

combined_x = np.concatenate(
    [train_data["x"], validation_data["x"], test_data["x"]],
    axis=0,
)
combined_y = np.concatenate(
    [train_data["y"], validation_data["y"], test_data["y"]],
    axis=0,
)
all_data = {"x": combined_x, "y": combined_y}

# Normal model
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=5,
        num_features=3,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    ),
    WhiteNoise(std_error=1e-6),
)
model.auto_initialize_baseline_states(train_data["y"])

# TODO: model.LstmNetwork, model.WhiteNoise
#  Abnormal model
ab_model = Model(
    LocalAcceleration(),
    LstmNetwork(
        look_back_len=24,
        num_features=3,
        num_layer=1,
        num_hidden_unit=30,
        device="cpu",
    ),
    WhiteNoise(std_error=1e-6),
)


# Switching Kalman filter
prob_norm_to_ab = 0.01
prob_ab_to_norm = 0.99
prior_prob_norm = 0.99

skf = SKF(
    normal_model=model,
    abnormal_model=ab_model,
    std_transition_error=1e-1,
    prob_norm_to_ab=prob_norm_to_ab,
    prob_ab_to_norm=prob_ab_to_norm,
    prob_norm=prior_prob_norm,
)

# for epoch in range(num_epoch):
for epoch in tqdm(range(num_epoch), desc="Training Progress", unit="epoch"):
    (mu_validation_preds, var_validation_preds, smoother_states) = skf.lstm_train(
        train_data=train_data, validation_data=validation_data
    )

prob_abnorm = skf.filter(data=all_data)


#  Plot
plt.figure(figsize=(10, 6))
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
    var=var_validation_preds,
    color="b",
    label=["mu_val_pred", "±1σ"],
)
plt.legend()
plt.show()


t = range(len(all_data["y"]))
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
axs[0].plot(t, all_data["y"], color="r", label="obs")
axs[0].axvline(x=point_anomaly, color="k", linestyle="--")
axs[0].legend()
axs[0].set_title("Observed Data")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Y")

axs[1].plot(t, prob_abnorm, color="blue", label="probability")
axs[1].axvline(x=point_anomaly, color="k", linestyle="--")
axs[1].legend()
axs[1].set_title("Probability of Abnormality")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Probability")

plt.tight_layout()
plt.show()
