import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from examples import DataProcess
from pytagi import Normalizer as normalizer
from pytagi import exponential_scheduler
from src import (
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    WhiteNoise,
    Model,
    plot_with_uncertainty,
    SKF,
)


# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
linear_space = np.linspace(0, 2, num=len(df_raw))
point_anomaly = 195
trend = np.linspace(0, 2, num=len(df_raw) - point_anomaly)
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
    train_end="2000-01-09 23:00:00",
    validation_start="2000-01-10 00:00:00",
    validation_end="2000-01-10 23:00:00",
    test_start="2000-01-11 00:00:00",
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Define parameters
num_epoch = 30

# Components
sigma_v = 5e-2
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=5,
    num_features=3,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
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
normal_to_abnormal_prob = 1e-4
abnormal_to_normal_prob = 1e-1
normal_model_prior_prob = 0.99

skf = SKF(
    norm_model=model,
    abnorm_model=ab_model,
    std_transition_error=1e-3,
    norm_to_abnorm_prob=normal_to_abnormal_prob,
    abnorm_to_norm_prob=abnormal_to_normal_prob,
    norm_model_prior_prob=normal_model_prior_prob,
)
skf.auto_initialize_baseline_states(train_data["y"][1:24])

#  Training
scheduled_sigma_v = 2
for epoch in tqdm(range(num_epoch), desc="Training Progress", unit="epoch"):
    # Decaying observation's variance
    scheduled_sigma_v = exponential_scheduler(
        curr_v=scheduled_sigma_v, min_v=sigma_v, decaying_factor=0.99, curr_iter=epoch
    )
    noise_index = model.states_name.index("white noise")
    model.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

    # Train the model
    (mu_validation_preds, std_validation_preds, train_states) = skf.lstm_train(
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

    # #  Plot hidden states
    # t = range(len(train_data["y"]))
    # fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
    # axs[0].plot(t, train_data["y"], color="r", label="obs")
    # axs[0].legend()
    # axs[0].set_title("Observed Data")
    # axs[0].set_ylabel("y")
    # plot_with_uncertainty(
    #     time=t,
    #     mu=train_states.mu_prior[:-1, 0],
    #     std=train_states.var_prior[:-1, 0, 0] ** 0.5,
    #     color="b",
    #     label=["mu_val_pred", "±1σ"],
    #     ax=axs[0],
    # )
    # plot_with_uncertainty(
    #     time=t,
    #     mu=train_states.mu_smooth[:-1, 0],
    #     std=train_states.var_smooth[:-1, 0, 0] ** 0.5,
    #     color="r",
    #     label=["mu_val_pred", "±1σ"],
    #     ax=axs[0],
    # )
    # axs[0].set_title("local level")

    # plot_with_uncertainty(
    #     time=t,
    #     mu=train_states.mu_smooth[1:, 1],
    #     std=train_states.var_smooth[1:, 1, 1] ** 0.5,
    #     color="b",
    #     label=["mu_val_pred", "±1σ"],
    #     ax=axs[1],
    # )
    # axs[1].set_title("local trend")

    # plot_with_uncertainty(
    #     time=t,
    #     mu=train_states.mu_smooth[1:, 3],
    #     std=train_states.var_smooth[1:, 3, 3] ** 0.5,
    #     color="b",
    #     label=["mu_val_pred", "±1σ"],
    #     ax=axs[2],
    # )
    # axs[2].set_title("lstm")

    # plot_with_uncertainty(
    #     time=t,
    #     # mu=train_states.mu_smooth[1:, 4],
    #     # std=train_states.var_smooth[1:, 4, 4] ** 0.5,
    #     mu=train_states.mu_posterior[1:, 4],
    #     std=train_states.var_posterior[1:, 4, 4] ** 0.5,
    #     color="b",
    #     label=["mu_val_pred", "±1σ"],
    #     ax=axs[3],
    # )
    # axs[3].set_title("white noise")

    # plt.tight_layout()
    # plt.show()


# Anomaly Detection
prob_abnorm, states = skf.filter(data=all_data)
_, _ = skf.smoother(data=all_data)

mu_plot = states.mu_posterior
var_plot = states.var_posterior

# mu_plot = states.mu_smooth
# var_plot = states.var_smooth

#  Plot
t = range(len(all_data["y"]))
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
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
axs[0].set_title("Validation predictions")

axs[1].plot(t, all_data["y"], color="r", label="obs")
axs[1].legend()
axs[1].set_title("Observed Data")
axs[1].set_ylabel("y")

axs[2].plot(t, prob_abnorm, color="blue")
axs[2].set_title("Probability of Abnormal Model")
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Prob abonormal")

plt.tight_layout()
plt.show()

#  Plot hidden states
fig, axs = plt.subplots(5, 1, figsize=(10, 8), sharex=False)
axs[0].plot(t, all_data["y"], color="r", label="obs")
axs[0].legend()
axs[0].set_title("Observed Data")
axs[0].set_ylabel("y")
plot_with_uncertainty(
    time=t,
    mu=mu_plot[:, 0],
    std=var_plot[:, 0, 0] ** 0.5,
    color="b",
    label=["mu_val_pred", "±1σ"],
    ax=axs[0],
)
axs[0].set_title("local level")

plot_with_uncertainty(
    time=t,
    mu=mu_plot[:, 1],
    std=var_plot[:, 1, 1] ** 0.5,
    color="b",
    label=["mu_val_pred", "±1σ"],
    ax=axs[1],
)
axs[1].set_title("local trend")


plot_with_uncertainty(
    time=t,
    mu=mu_plot[:, 3],
    std=var_plot[:, 3, 3] ** 0.5,
    color="b",
    label=["mu_val_pred", "±1σ"],
    ax=axs[2],
)
axs[2].set_title("lstm")

plot_with_uncertainty(
    time=t,
    mu=mu_plot[:, 4],
    std=var_plot[:, 4, 4] ** 0.5,
    color="b",
    label=["mu_val_pred", "±1σ"],
    ax=axs[3],
)
axs[3].set_title("white noise")

axs[4].plot(t, prob_abnorm, color="r")
axs[4].set_title("Probability of Abnormal Model")
axs[4].set_xlabel("Time")
axs[4].set_ylabel("Prob abonormal")

plt.tight_layout()
plt.show()
