import pandas as pd
from pytagi import Normalizer as normalizer
from pytagi import exponential_scheduler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from examples import DataProcess
from src import (
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    WhiteNoise,
    Model,
    SKF,
    plot_with_uncertainty,
)
import pytagi.metric as metric


# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Add synthetic anomaly to data
trend = np.linspace(0, 0, num=len(df_raw))
time_anomaly = 200
new_trend = np.linspace(0, 2, num=len(df_raw) - time_anomaly)
trend[time_anomaly:] = trend[time_anomaly:] + new_trend
df_raw = df_raw.add(trend, axis=0)

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    train_start="2000-01-01 00:00:00",
    train_end="2000-01-09 23:00:00",
    validation_start="2000-01-10 00:00:00",
    validation_end="2000-01-10 23:00:00",
    test_start="2000-01-11 00:00:00",
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Components
sigma_v = 1e-2
local_trend = LocalTrend(var_states=[1e-2, 1e-2])
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=12,
    num_features=1,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
    # manual_seed=1,
)
noise = WhiteNoise(std_error=sigma_v)

# Normal model
model = Model(
    local_trend,
    lstm_network,
    noise,
)

#  Abnormal model
ab_model = Model(
    local_acceleration,
    lstm_network,
    noise,
)

# Switching Kalman filter
skf = SKF(
    norm_model=model,
    abnorm_model=ab_model,
    std_transition_error=1e-4,
    norm_to_abnorm_prob=1e-4,
    abnorm_to_norm_prob=1e-1,
    norm_model_prior_prob=0.99,
)
skf.auto_initialize_baseline_states(train_data["y"][0:23])

#  Training
num_epoch = 50
scheduled_sigma_v = 1
for epoch in tqdm(range(num_epoch), desc="Training Progress", unit="epoch"):
    # # Decaying observation's variance
    scheduled_sigma_v = exponential_scheduler(
        curr_v=scheduled_sigma_v, min_v=sigma_v, decaying_factor=0.9, curr_iter=epoch
    )
    noise_index = model.states_name.index("white noise")
    skf.model["norm_norm"].process_noise_matrix[noise_index, noise_index] = (
        scheduled_sigma_v**2
    )

    # Train the model
    (mu_validation_preds, std_validation_preds, train_states) = skf.lstm_train(
        train_data=train_data, validation_data=validation_data
    )

    # # Unstandardize the predictions
    mu_validation_preds = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.data_mean[output_col],
        data_processor.data_std[output_col],
    )
    std_validation_preds = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.data_std[output_col],
    )


mse = metric.mse(
    mu_validation_preds, data_processor.validation_data[:, output_col].flatten()
)
print(f"MSE           : {mse: 0.4f}")

# # # Anomaly Detection
skf.model["norm_norm"].process_noise_matrix[noise_index, noise_index] = sigma_v**2
filter_marginal_abnorm_prob, _ = skf.filter(data=all_data)
smooth_marginal_abnorm_prob, states = skf.smoother(data=all_data)

#  Plot
mu_plot = states.mu_posterior
var_plot = states.var_posterior
marginal_abnorm_prob_plot = filter_marginal_abnorm_prob

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    data_processor.train_time,
    data_processor.train_data[:, output_col].flatten(),
    color="r",
    label="train_obs",
)
ax.plot(
    data_processor.validation_time,
    data_processor.validation_data[:, output_col].flatten(),
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

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
ax.grid(True, which="major", axis="x", linewidth=1.0, linestyle="-")
ax.grid(True, which="minor", axis="x", linewidth=0.5, linestyle="--")
ax.set_xlabel("Time")
plt.legend()
plt.title("Validation predictions")
plt.tight_layout()
plt.show()


#  Plot hidden states
local_level_index = skf.states_name.index("local level")
local_trend_index = skf.states_name.index("local trend")
lstm_index = skf.states_name.index("lstm")
white_noise_index = skf.states_name.index("white noise")

fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=False)
axs[0].plot(data_processor.time, all_data["y"], color="r", label="obs")
axs[0].set_title("Observed Data")
axs[0].set_ylabel("y")
plot_with_uncertainty(
    time=data_processor.time,
    mu=mu_plot[:, local_level_index],
    std=var_plot[:, local_level_index, local_level_index] ** 0.5,
    color="b",
    label=["mu_val_pred", "±1σ"],
    ax=axs[0],
)
axs[0].axvline(
    x=data_processor.time[time_anomaly], color="b", linestyle="--", label="anomaly"
)
axs[0].xaxis.set_major_locator(mdates.DayLocator())
axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
axs[0].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
axs[0].grid(True, which="major", axis="x", linewidth=1.0, linestyle="-")
axs[0].grid(True, which="minor", axis="x", linewidth=0.5, linestyle="--")
axs[0].set_ylabel("Local level")


plot_with_uncertainty(
    time=data_processor.time,
    mu=mu_plot[:, local_trend_index],
    std=var_plot[:, local_trend_index, local_trend_index] ** 0.5,
    color="b",
    label=["mu_val_pred", "±1σ"],
    ax=axs[1],
)
axs[1].axvline(
    x=data_processor.time[time_anomaly], color="b", linestyle="--", label="anomaly"
)
axs[1].axhline(y=0, color="black", linestyle="--", label="zero speed")
axs[1].xaxis.set_major_locator(mdates.DayLocator())
axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
axs[1].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
axs[1].grid(True, which="major", axis="x", linewidth=1.0, linestyle="-")
axs[1].grid(True, which="minor", axis="x", linewidth=0.5, linestyle="--")
axs[1].legend()
axs[1].set_ylabel("Local trend")

plot_with_uncertainty(
    time=data_processor.time,
    mu=mu_plot[:, lstm_index],
    std=var_plot[:, lstm_index, lstm_index] ** 0.5,
    color="b",
    label=["mu_val_pred", "±1σ"],
    ax=axs[2],
)
axs[2].axvline(
    x=data_processor.time[time_anomaly], color="b", linestyle="--", label="anomaly"
)
axs[2].xaxis.set_major_locator(mdates.DayLocator())
axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
axs[2].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
axs[2].grid(True, which="major", axis="x", linewidth=1.0, linestyle="-")
axs[2].grid(True, which="minor", axis="x", linewidth=0.5, linestyle="--")
axs[2].set_ylabel("LSTM")

plot_with_uncertainty(
    time=data_processor.time,
    mu=mu_plot[:, white_noise_index],
    std=var_plot[:, white_noise_index, white_noise_index] ** 0.5,
    color="b",
    label=["mu_val_pred", "±1σ"],
    ax=axs[3],
)
axs[3].axvline(
    x=data_processor.time[time_anomaly], color="b", linestyle="--", label="anomaly"
)
axs[3].xaxis.set_major_locator(mdates.DayLocator())
axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
axs[3].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
axs[3].grid(True, which="major", axis="x", linewidth=1.0, linestyle="-")
axs[3].grid(True, which="minor", axis="x", linewidth=0.5, linestyle="--")
axs[3].set_ylabel("White noise residual")

axs[4].plot(data_processor.time, marginal_abnorm_prob_plot, color="b")
axs[4].axvline(
    x=data_processor.time[time_anomaly], color="b", linestyle="--", label="anomaly"
)
axs[4].xaxis.set_major_locator(mdates.DayLocator())
axs[4].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
axs[4].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
axs[4].grid(True, which="major", axis="x", linewidth=1.0, linestyle="-")
axs[4].grid(True, which="minor", axis="x", linewidth=0.5, linestyle="--")
axs[4].set_xlabel("Time")
axs[4].set_ylabel("Pr(Abnormal)")

plt.tight_layout()
plt.show()
