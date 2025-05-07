import copy
import pandas as pd
from pytagi import Normalizer as normalizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytagi.metric as metric
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_data,
    plot_prediction,
    plot_skf_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise

# load target data
data_file_target = "./data/benchmark_data/test_11_data.csv"
df_target = pd.read_csv(
    data_file_target,
    usecols=[0, 1],  # Read only the first two columns
    skiprows=1,
    header=None,
    names=["date_time", "values"],
    parse_dates=["date_time"],
    index_col="date_time",
)

# Build data processor for the target series
data_processor = DataProcess(
    data=df_target,
    time_covariates=["week_of_year"],
    train_split=0.3,
    validation_split=0.1,
    output_col=[0],
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Components
sigma_v = 0.2
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=19,
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
    manual_seed=1,
    load_lstm_net="./saved_params/lstm_net_test.pth",
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
skf.auto_initialize_baseline_states(train_data["y"][0:104])

#  Training
num_epoch = 200
states_optim = None
mu_validation_preds_optim = None
std_validation_preds_optim = None
output_col = [0]

for epoch in tqdm(range(num_epoch), desc="Training Progress", unit="epoch"):
    # Train the model
    (mu_validation_preds, std_validation_preds, states) = skf.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
        white_noise_decay=True,
        white_noise_max_std=3,
        white_noise_decay_factor=0.9,
    )

    # # Unstandardize the predictions
    mu_validation_preds_unnorm = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.norm_const_mean[data_processor.output_col],
        data_processor.norm_const_std[data_processor.output_col],
    )

    std_validation_preds_unnorm = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.norm_const_std[data_processor.output_col],
    )

    validation_obs = data_processor.get_data("validation").flatten()
    validation_log_lik = metric.log_likelihood(
        prediction=mu_validation_preds_unnorm,
        observation=validation_obs,
        std=std_validation_preds_unnorm,
    )

    # Early-stopping
    skf.early_stopping(evaluate_metric=-validation_log_lik, mode="min")
    if epoch == skf.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds.copy()
        std_validation_preds_optim = std_validation_preds.copy()
        states_optim = copy.copy(states)
        state_dict = skf.lstm_net.state_dict()
    if skf.stop_training:
        break

print(f"Optimal epoch       : {skf.optimal_epoch}")
print(f"Validation log-likelihood  :{skf.early_stop_metric: 0.4f}")

# Load the optimal model
skf.lstm_net.load_state_dict(state_dict)


# # Anomaly Detection
filter_marginal_abnorm_prob, _ = skf.filter(data=all_data)
smooth_marginal_abnorm_prob, states = skf.smoother(data=all_data)

# # Plot
marginal_abnorm_prob_plot = smooth_marginal_abnorm_prob
fig, ax = plt.subplots(figsize=(10, 6))
plot_data(
    data_processor=data_processor,
    plot_column=output_col,
    normalization=True,
    plot_test_data=False,
    sub_plot=ax,
    validation_label="y",
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds_optim,
    std_validation_pred=std_validation_preds_optim,
    sub_plot=ax,
    validation_label=[r"$\mu$", f"$\pm\sigma$"],
)
ax.set_xlabel("Time")
plt.title("Validation predictions")
plt.tight_layout()
plt.legend()
plt.show()


fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=states,
    states_to_plot=["local level", "local trend", "lstm", "white noise"],
    model_prob=marginal_abnorm_prob_plot,
    # normalization=True,
    color="b",
)
fig.suptitle("SKF hidden states", fontsize=10, y=1)
plt.show()

fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=states,
    states_to_plot=["local level", "local trend", "lstm", "white noise"],
    model_prob=marginal_abnorm_prob_plot,
    states_type="smooth",
    # normalization=True,
    color="b",
)
fig.suptitle("SKF hidden states", fontsize=10, y=1)
plt.show()
