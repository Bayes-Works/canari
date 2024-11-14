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
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
import copy

# # Read data
data_file = "./data/HQ_LGA007Y.csv"
df_raw = pd.read_csv(data_file, skiprows=0, delimiter=",", header=None)
lags = [0, 4, 4, 4]
df_raw = DataProcess.add_lagged_columns(df_raw, lags)

# Data processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    train_start="0",
    train_end="215",
    validation_start="216",
    validation_end="268",
    test_start="269",
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Define parameters
num_epoch = 100
patience = 10
log_lik_optimal = -1e100
mse_optim = 1e100
epoch_optimal = 0
lstm_param_optimal = {}
ll = []

# Components
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=26,
    num_features=df_raw.shape[1],
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
)
noise = WhiteNoise(std_error=5e-2)

# Switching Kalman filter
normal_to_abnormal_prob = 1e-4
abnormal_to_normal_prob = 1e-1
normal_model_prior_prob = 0.99

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
    normal_model=model,
    abnormal_model=ab_model,
    std_transition_error=1e-4,
    normal_to_abnormal_prob=normal_to_abnormal_prob,
    abnormal_to_normal_prob=abnormal_to_normal_prob,
    normal_model_prior_prob=normal_model_prior_prob,
)
skf.auto_initialize_baseline_states(train_data["y"])


#  Training

for epoch in tqdm(range(num_epoch), desc="Training Progress", unit="epoch"):
    # Train the model
    (mu_validation_preds, std_validation_preds, smoother_states) = skf.lstm_train(
        train_data=train_data, validation_data=validation_data
    )

    # Unstandardize the predictions
    mu_validation_preds = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.data_mean[output_col],
        data_processor.data_std[output_col],
    )
    std_validation_preds = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.data_std[output_col],
    )

    # Calculate the log-likelihood metric
    log_lik = metric.log_likelihood(
        prediction=mu_validation_preds,
        observation=data_processor.validation_data[:, output_col].flatten(),
        std=std_validation_preds,
    )

    # Early stopping
    if log_lik > log_lik_optimal:
        log_lik_optimal = copy.copy(log_lik)
        epoch_optimal = copy.copy(epoch)
        lstm_param_optimal = copy.copy(skf.norm_model.lstm_net.get_state_dict())
    ll.append(log_lik)
    if (epoch - epoch_optimal) > patience:
        break


# Load back the LSTM's optimal parameters
skf.norm_model.lstm_net.load_state_dict(lstm_param_optimal)

# Anomaly Detection
_, _, prob_abnorm = skf.filter(data=all_data)
# _, _, prob_abnorm = skf.smoother(data=all_data)

print(f"Optimal epoch: {epoch_optimal}")

#  Plot
t = range(len(all_data["y"]))
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=False)

axs[0].plot(
    data_processor.train_time,
    data_processor.train_data[:, output_col].flatten(),
    color="r",
    label="train_obs",
)
axs[0].plot(
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
    ax=axs[0],
)
axs[0].legend()

axs[1].plot(ll, color="r", label="log_likelihood")

axs[2].plot(t, all_data["y"], color="r", label="obs")
axs[2].legend()
axs[2].set_title("Observed Data")
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Y")

axs[3].plot(t, prob_abnorm, color="blue", label="probability")
axs[3].legend()
axs[3].set_title("Probability of Abnormality")
axs[3].set_xlabel("Time")
axs[3].set_ylabel("Probability")

plt.tight_layout()
plt.show()
