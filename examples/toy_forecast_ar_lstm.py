import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from canari.component import LocalTrend, LstmNetwork, Autoregression


###########################
###########################
#  Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Add trend
trend_true = 0.1
df_raw["values"] += np.arange(len(df_raw)) * trend_true

# Data processor initialization
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=0.6,
    validation_split=0.2,
    output_col=output_col,
)

train_data, val_data, test_data, normalized_data = data_processor.get_splits()

# Normalization constants
norm_const_mean = data_processor.norm_const_mean[output_col].item()
norm_const_std = data_processor.norm_const_std[output_col].item()

# Define model components
trend_norm = trend_true / (norm_const_std + 1e-10)
level_norm = (5.0 - norm_const_mean) / (norm_const_std + 1e-10)
mu_W2bar_prior = 1e4
var_AR_prior = 1e4
var_W2bar_prior = 1e4

lstm = LstmNetwork(
    look_back_len=52,
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
    # manual_seed=1,
)
ar = Autoregression(
    mu_states=[0, 0, 0, 0, 0, mu_W2bar_prior],
    var_states=[1e-6, 0.01, 0, var_AR_prior, 0, var_W2bar_prior],
)

model = Model(
    LocalTrend(
        mu_states=[level_norm, trend_norm], var_states=[1e-12, 1e-12], std_error=0
    ),
    lstm,
    ar,
)
model._mu_local_level = level_norm

###########################
###########################
# Training model
num_epochs = 200
states_optim = None
optimal_mu_val_preds = None
optimal_std_val_preds = None
for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
    mu_validation_preds, std_validation_preds, states = model.lstm_train(
        train_data=train_data,
        validation_data=val_data,
    )

    # Unstandardize the predictions
    mu_pred_unnorm = normalizer.unstandardize(
        mu_validation_preds, norm_const_mean, norm_const_std
    )
    std_pred_unnorm = normalizer.unstandardize_std(std_validation_preds, norm_const_std)

    # Calculate the evaluation metric
    obs_validation = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_pred_unnorm, obs_validation)
    val_log_lik = metric.log_likelihood(mu_pred_unnorm, obs_validation, std_pred_unnorm)

    # Early-stopping
    model.early_stopping(evaluate_metric=-val_log_lik, mode="min")
    if epoch == model.optimal_epoch:
        optimal_mu_val_preds = mu_validation_preds.copy()
        optimal_std_val_preds = std_validation_preds.copy()
        states_optim = copy.copy(states)

    if model.stop_training:
        break

# save model
model_dict = model.get_dict()
model_dict["mu_states_optimal"] = states_optim.mu_prior[-1]

print(f"Optimal epoch: {model.optimal_epoch}")
print(f"Validation MSE: {model.early_stop_metric:.4f}")


###########################
###########################
# Reload pretrained model
# Load learned parameters from the saved trained model
phi_index = model_dict["phi_index"]
W2bar_index = model_dict["W2bar_index"]
autoregression_index = model_dict["autoregression_index"]
mu_W2bar_learn = model_dict["mu_states_optimal"][W2bar_index].item()
phi_AR_learn = model_dict["mu_states_optimal"][phi_index].item()
mu_AR = model_dict["mu_states"][autoregression_index].item()
var_AR = model_dict["var_states"][autoregression_index, autoregression_index].item()

print("Learned phi_AR =", phi_AR_learn)
print("Learned sigma_AR =", np.sqrt(mu_W2bar_learn))

# # # # # # #
# Define pretrained model:
pretrained_model = Model(
    LocalTrend(
        mu_states=[level_norm, trend_norm], var_states=[1e-12, 1e-12], std_error=0
    ),
    lstm,
    Autoregression(
        std_error=np.sqrt(mu_W2bar_learn),
        phi=phi_AR_learn,
        mu_states=[mu_AR],
        var_states=[var_AR],
    ),
)

# load lstm's component
pretrained_model.lstm_net.load_state_dict(model.lstm_net.state_dict())

# filter and smoother
pretrained_model.filter(normalized_data, train_lstm=False)
pretrained_model.smoother(normalized_data)


# # Plotting results at the optimal epoch when training model
state_type = "prior"
fig, ax = plot_states(
    data_processor=data_processor,
    states=states_optim,
    normalization=True,
    states_type=state_type,
    states_to_plot=[
        "local level",
        "local trend",
        "lstm",
        "autoregression",
        "phi",
        "W2bar",
    ],
)
plot_data(
    data_processor=data_processor,
    normalization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax[0],
    plot_test_data=False,
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=optimal_mu_val_preds,
    std_validation_pred=optimal_std_val_preds,
    sub_plot=ax[0],
)
fig.suptitle("Hidden states at the optimal epoch in training", fontsize=10, y=1)
plt.show()

# # Plotting results from pre-trained model
fig, ax = plot_states(
    data_processor=data_processor,
    states=pretrained_model.states,
    states_type=state_type,
    normalization=True,
    states_to_plot=[
        "local level",
        "local trend",
        "lstm",
        "autoregression",
    ],
)
plot_data(
    data_processor=data_processor,
    normalization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax[0],
    plot_test_data=False,
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=optimal_mu_val_preds,
    std_validation_pred=optimal_std_val_preds,
    sub_plot=ax[0],
)
fig.suptitle("Hidden states estimated by the pre-trained model", fontsize=10, y=1)
plt.show()
