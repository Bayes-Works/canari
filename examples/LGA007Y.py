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
data_file = "./data/HQ_LGA007Y.csv"
df_raw = pd.read_csv(data_file, skiprows=0, delimiter=",", header=None)

# Define parameters
output_col = [0]
num_epoch = 50

data_processor = DataProcess(
    data=df_raw,
    train_start="0",
    train_end="260",
    validation_start="261",
    validation_end="320",
    test_start="321",
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


# Components
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=26,
    num_features=5,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
)
noise = WhiteNoise(std_error=1e-1)

# Normal model
model = Model(
    local_trend,
    lstm_network,
    noise,
)

# TODO: model.LstmNetwork, model.WhiteNoise
#  Abnormal model
ab_model = Model(
    local_acceleration,
    lstm_network,
    noise,
)

# Switching Kalman filter
normal_to_abnormal_prob = 1e-4
abnormal_to_normal_prob = 1e-1
normal_model_prior_prob = 0.99

skf = SKF(
    normal_model=model,
    abnormal_model=ab_model,
    std_transition_error=1e-4,
    normal_to_abnormal_prob=normal_to_abnormal_prob,
    abnormal_to_normal_prob=abnormal_to_normal_prob,
    normal_model_prior_prob=normal_model_prior_prob,
)
skf.auto_initialize_baseline_states(train_data["y"])

for epoch in tqdm(range(num_epoch), desc="Training Progress", unit="epoch"):
    (mu_validation_preds, var_validation_preds, smoother_states) = skf.lstm_train(
        train_data=train_data, validation_data=validation_data
    )

# _, _, prob_abnorm = skf.filter(data=all_data)
_, _, prob_abnorm = skf.smoother(data=all_data)

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
axs[0].legend()
axs[0].set_title("Observed Data")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Y")

axs[1].plot(t, prob_abnorm, color="blue", label="probability")
axs[1].legend()
axs[1].set_title("Probability of Abnormality")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Probability")

plt.tight_layout()
plt.show()
