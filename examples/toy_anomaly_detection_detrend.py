import copy
import pandas as pd
from pytagi import Normalizer as normalizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytagi.metric as metric
from statsmodels.tsa.seasonal import seasonal_decompose
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_data,
    plot_prediction,
    plot_skf_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise


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
trend = np.linspace(0, 2, num=len(df_raw))
time_anomaly = 120
new_trend = np.linspace(0, -1, num=len(df_raw) - time_anomaly)
trend[time_anomaly:] = trend[time_anomaly:] + new_trend
df_raw = df_raw.add(trend, axis=0)

# Detrending data
df_detrend = df_raw.copy()
df_detrend = df_detrend.interpolate()
decomposition = seasonal_decompose(df_detrend["values"], model="additive", period=24)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
residual = residual.fillna(0)
df_detrend.values = seasonal + residual + np.mean(trend)
# decomposition.plot()
# plt.plot(seasonal + residual)
# fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
# axs[0].plot(df_detrend["values"])
# axs[1].plot(seasonal)
# axs[2].plot(residual)
# plt.show()

# # Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_detrend,
    time_covariates=["hour_of_day"],
    train_split=0.4,
    validation_split=0.1,
    output_col=output_col,
)
train_data, validation_data, _, _ = data_processor.get_splits()

data_values = np.concatenate(
    (df_raw.values, data_processor.data.values[:, 1].reshape(-1, 1)), axis=1
)
all_data_norm = normalizer.standardize(
    data=data_values,
    mu=data_processor.norm_const_mean,
    std=data_processor.norm_const_std,
)
all_data = {}
all_data["x"] = all_data_norm[:, data_processor.covariates_col]
all_data["y"] = all_data_norm[:, data_processor.output_col]

# Components
sigma_v = 3e-2
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=10,
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
    # manual_seed=1,
)
noise = WhiteNoise(std_error=sigma_v)

# Model for training lstm
model_lstm = Model(local_trend, lstm_network, noise)
model_lstm.auto_initialize_baseline_states(train_data["y"][0:23])

# Switching Kalman filter
model = Model(
    local_trend,
    lstm_network,
    noise,
)
ab_model = Model(
    local_acceleration,
    lstm_network,
    noise,
)
skf = SKF(
    norm_model=model,
    abnorm_model=ab_model,
    std_transition_error=1e-4,
    norm_to_abnorm_prob=1e-4,
    abnorm_to_norm_prob=1e-1,
    norm_model_prior_prob=0.99,
)
skf.auto_initialize_baseline_states(all_data["y"][0 : 24 * 2])

#  Training
num_epoch = 50
states_optim = None
mu_validation_preds_optim = None
std_validation_preds_optim = None

for epoch in tqdm(range(num_epoch), desc="Training Progress", unit="epoch"):
    # Train the model
    (mu_validation_preds, std_validation_preds, states) = model_lstm.lstm_train(
        train_data=train_data, validation_data=validation_data
    )

    # # Unstandardize the predictions
    mu_validation_preds_unnorm = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.norm_const_mean[output_col],
        data_processor.norm_const_std[output_col],
    )

    std_validation_preds_unnorm = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.norm_const_std[output_col],
    )

    validation_obs = data_processor.get_data("validation").flatten()
    validation_log_lik = metric.log_likelihood(
        prediction=mu_validation_preds_unnorm,
        observation=validation_obs,
        std=std_validation_preds_unnorm,
    )

    # Early-stopping
    model_lstm.early_stopping(evaluate_metric=-validation_log_lik, mode="min")
    if epoch == model_lstm.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds.copy()
        std_validation_preds_optim = std_validation_preds.copy()
        states_optim = copy.copy(states)
    if model_lstm.stop_training:
        break

print(f"Optimal epoch       : {model_lstm.optimal_epoch}")
print(f"Validation log-likelihood  :{model_lstm.early_stop_metric: 0.4f}")

# # Anomaly Detection
skf.model["norm_norm"].lstm_net = model_lstm.lstm_net
skf.lstm_net = model_lstm.lstm_net
filter_marginal_abnorm_prob, _ = skf.filter(data=all_data)
smooth_marginal_abnorm_prob, states = skf.smoother(data=all_data)

# # Plot
marginal_abnorm_prob_plot = filter_marginal_abnorm_prob
fig, ax = plt.subplots(figsize=(10, 6))
plot_data(
    data_processor=data_processor,
    plot_column=output_col,
    normalization=True,
    plot_test_data=False,
    validation_label="y",
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds,
    std_validation_pred=std_validation_preds,
    validation_label=[r"$\mu$", f"$\pm\sigma$"],
)
ax.set_xlabel("Time")
plt.legend()
plt.title("Validation predictions")
plt.tight_layout()
plt.show()

# plot skf results
time = data_processor.get_time("all")
fig, ax = plot_skf_states(
    data_processor=data_processor,
    plot_observation=False,
    normalization=True,
    states=states,
    model_prob=marginal_abnorm_prob_plot,
    color="b",
)
ax[0].plot(time, all_data["y"], color="red")
ax[0].axvline(
    x=data_processor.data.index[time_anomaly],
    color="r",
    linestyle="--",
)
fig.suptitle("SKF hidden states", fontsize=10, y=1)
plt.show()
