import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from canari.component import LocalTrend, LstmNetwork, Autoregression

# # Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Change the trend of data
trend_true = 0.0
df_raw.values += (
    np.arange(len(df_raw.values), dtype="float64").reshape(-1, 1) * trend_true
)

# # Skip resampling data
df = df_raw

# Define parameters
output_col = [0]
num_epoch = 200
train_val_split = [0.6, 0.23]

data_processor = DataProcess(
    data=df,
    time_covariates=["week_of_year"],
    train_split=train_val_split[0],
    validation_split=train_val_split[1],
    output_col=output_col,
)
obs_norm_const_mean = data_processor.norm_const_mean[output_col].item()
obs_norm_const_std = data_processor.norm_const_std[output_col].item()
time_covariate_norm_const_mean = data_processor.norm_const_mean[
    data_processor.covariates_col
].item()
time_covariate_norm_const_std = data_processor.norm_const_std[
    data_processor.covariates_col
].item()

trend_true_norm = trend_true / (obs_norm_const_std + 1e-10)
level_true_norm = (5.0 - obs_norm_const_mean) / (obs_norm_const_std + 1e-10)
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

train_index, val_index, test_index = data_processor.get_split_indices()
time_covariate_info = {
    "initial_time_covariate": data_processor.data.values[
        val_index[-1], data_processor.covariates_col
    ].item(),
    "mu": time_covariate_norm_const_mean,
    "std": time_covariate_norm_const_std,
}

LSTM = LstmNetwork(
    look_back_len=52,
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
)

# Define AR model
AR_process_error_var_prior = 1e4
var_W2bar_prior = 1e4
AR = Autoregression(
    mu_states=[0, 0, 0, 0, 0, AR_process_error_var_prior],
    var_states=[1e-06, 0.01, 0, AR_process_error_var_prior, 0, var_W2bar_prior],
)

model = Model(
    LocalTrend(
        mu_states=[level_true_norm, trend_true_norm],
        var_states=[1e-12, 1e-12],
        std_error=0,
    ),  # True baseline values
    LSTM,
    AR,
)
model._mu_local_level = level_true_norm


# Training
for epoch in range(num_epoch):
    mu_validation_preds, std_validation_preds, states = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )

    # Unstandardize the predictions
    mu_validation_preds_unnorm = normalizer.unstandardize(
        mu_validation_preds,
        obs_norm_const_mean,
        obs_norm_const_std,
    )
    std_validation_preds_unnorm = normalizer.unstandardize_std(
        std_validation_preds,
        obs_norm_const_std,
    )

    # Calculate the evaluation metric
    validation_obs = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_validation_preds_unnorm, validation_obs)

    validation_log_lik = metric.log_likelihood(
        prediction=mu_validation_preds_unnorm,
        observation=validation_obs,
        std=std_validation_preds_unnorm,
    )

    # Early-stopping
    model.early_stopping(evaluate_metric=-validation_log_lik, mode="min")

    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds_unnorm.copy()
        std_validation_preds_optim = std_validation_preds_unnorm.copy()
        states_optim = copy.copy(states)
    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

model_dict = model.get_dict()
model_dict["states_optimal"] = states_optim

# Save model_dict to local
import pickle

with open("saved_params/toy_model_dict.pkl", "wb") as f:
    pickle.dump(model_dict, f)

####################################################################
############ Use pretrained model to generate data #################
####################################################################

with open("saved_params/toy_model_dict.pkl", "rb") as f:
    pretrained_model_dict = pickle.load(f)
phi_index = pretrained_model_dict["phi_index"]
W2bar_index = pretrained_model_dict["W2bar_index"]
autoregression_index = pretrained_model_dict["autoregression_index"]

print(
    "phi_AR =", pretrained_model_dict["states_optimal"].mu_prior[-1][phi_index].item()
)
print(
    "sigma_AR =",
    np.sqrt(pretrained_model_dict["states_optimal"].mu_prior[-1][W2bar_index].item()),
)

train_val_data = np.concatenate(
    (train_data["y"].reshape(-1), validation_data["y"].reshape(-1))
)

pretrained_model = Model(
    LocalTrend(
        mu_states=pretrained_model_dict["mu_states"][0:2].reshape(-1),
        var_states=np.diag(pretrained_model_dict["var_states"][0:2, 0:2]),
    ),
    LSTM,
    Autoregression(
        std_error=np.sqrt(
            pretrained_model_dict["states_optimal"].mu_prior[-1][W2bar_index].item()
        ),
        phi=pretrained_model_dict["states_optimal"].mu_prior[-1][phi_index].item(),
        mu_states=[pretrained_model_dict["mu_states"][autoregression_index].item()],
        var_states=[
            pretrained_model_dict["var_states"][
                autoregression_index, autoregression_index
            ].item()
        ],
    ),
)
pretrained_model.lstm_net.load_state_dict(pretrained_model_dict["lstm_network_params"])

# Generate data
pretrained_model.filter(train_data, train_lstm=False)
pretrained_model.filter(validation_data, train_lstm=False)
generated_ts, _, _, _ = pretrained_model.generate_time_series(
    num_time_series=3,
    num_time_steps=len(train_val_data),
    time_covariates=data_processor.time_covariates,
    time_covariate_info=time_covariate_info,
)

time_idx = np.arange(
    len(
        np.concatenate(
            (
                train_data["y"].reshape(-1),
                validation_data["y"].reshape(-1),
                generated_ts[0],
            )
        )
    )
)
val_end_idx = len(train_val_data)
train_end_idx = int(train_val_split[0] / np.sum(train_val_split) * val_end_idx)

# Plot
# # Plot states from training
state_type = "prior"
fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type=state_type,
    states_to_plot=["local level", "local trend", "lstm", "autoregression"],
)
plot_data(
    data_processor=data_processor,
    normalization=False,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax[0],
    plot_test_data=False,
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds_optim,
    std_validation_pred=std_validation_preds_optim,
    sub_plot=ax[0],
)
ax[0].set_xticklabels([])
ax[0].set_title("Hidden states in training and validation forecast")

# # Plot generated time series
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 1)
ax0 = plt.subplot(gs[0])
for i in range(len(generated_ts)):
    ax0.plot(
        np.concatenate(
            (
                train_data["y"].reshape(-1),
                validation_data["y"].reshape(-1),
                generated_ts[i],
            )
        )
    )
ax0.plot(
    np.concatenate((train_data["y"].reshape(-1), validation_data["y"].reshape(-1))),
    color="k",
    label="original",
)
ax0.axvline(
    x=len(train_data["y"]) + len(validation_data["y"]), color="r", linestyle="--"
)
ax0.axvspan(time_idx[train_end_idx], time_idx[val_end_idx], color="green", alpha=0.1)
ax0.axvspan(time_idx[val_end_idx], time_idx[-1], color="k", alpha=0.1)
ax0.legend(loc="lower left")
ax0.set_title("Data generation")
plt.show()
