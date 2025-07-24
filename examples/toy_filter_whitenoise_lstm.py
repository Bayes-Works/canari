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
from canari.component import LocalTrend, WhiteNoise, LstmNetwork, Autoregression


###########################
###########################
#  Read data
data_file = "./data/toy_time_series/simple_syn_ar_std01_phi07_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Data processor initialization
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=0.6,
    validation_split=0.2,
    output_col=output_col,
    # standardization=False,
)
data_processor.scale_const_mean = np.array([0, 2.6068333e+01])
data_processor.scale_const_std = np.array([1, 15.090957])

train_data, val_data, test_data, standardized_data = data_processor.get_splits()

# Standardization constants
scale_const_mean = data_processor.scale_const_mean[output_col].item()
scale_const_std = data_processor.scale_const_std[output_col].item()
print(data_processor.scale_const_mean, data_processor.scale_const_std)

# # Define model components
# trend_norm = trend_true / (scale_const_std + 1e-10)
# level_norm = (5.0 - scale_const_mean) / (scale_const_std + 1e-10)

lstm = LstmNetwork(
    look_back_len=52,
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
    # manual_seed=1,
)

model = Model(
    LocalTrend(
        mu_states=[0, 0], var_states=[1e-12, 1e-12], std_error=0
    ),
    LocalTrend(),
    lstm,
    WhiteNoise(std_error=0.003),
)
model._mu_local_level = 0
# model.auto_initialize_baseline_states(train_data["y"][0:24])

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

    # # Unstandardize the predictions
    # mu_pred_unnorm = normalizer.unstandardize(
    #     mu_validation_preds, scale_const_mean, scale_const_std
    # )
    # std_pred_unnorm = normalizer.unstandardize_std(
    #     std_validation_preds, scale_const_std
    # )

    # Calculate the evaluation metric
    obs_validation = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_validation_preds, obs_validation)
    val_log_lik = metric.log_likelihood(mu_validation_preds, obs_validation, std_validation_preds)

    # Early-stopping
    model.early_stopping(
        evaluate_metric=-val_log_lik, current_epoch=epoch, max_epoch=num_epochs
    )
    if epoch == model.optimal_epoch:
        optimal_mu_val_preds = mu_validation_preds.copy()
        optimal_std_val_preds = std_validation_preds.copy()
        states_optim = copy.copy(states)
        noise_index = model.get_states_index("white noise")
        sigma_v_learnt = np.sqrt(model.process_noise_matrix[noise_index, noise_index])

    model.set_memory(states=states, time_step=0)
    if model.stop_training:
        break

# save model
model_dict = model.get_dict()
model_dict["mu_states_optimal"] = states_optim.mu_prior[-1]

print(f"Optimal epoch: {model.optimal_epoch}")
print(f"Validation MSE: {model.early_stop_metric:.4f}")

true_sigma_AR = 0.1
true_phi_AR = 0.7
stationary_std_AR = true_sigma_AR / np.sqrt(1 - true_phi_AR**2)
stationary_std_AR /= (scale_const_std + 1e-10)


###########################
###########################
# Reload pretrained model
# Load learned parameters from the saved trained model
print("Learned sigma_v =", sigma_v_learnt)
print('----------------------------------------------')
print("Stationary std AR =", stationary_std_AR)

# # # # # # #
# Define pretrained model:
pretrained_model = Model(
    LocalTrend(
        mu_states=[0, 0], var_states=[1e-12, 1e-12], std_error=0
    ),
    lstm,
    WhiteNoise(std_error=sigma_v_learnt),
)

# load lstm's component
pretrained_model.lstm_net.load_state_dict(model.lstm_net.state_dict())

# filter and smoother
pretrained_model.filter(standardized_data, train_lstm=False)
pretrained_model.smoother()


# # Plotting results at the optimal epoch when training model
state_type = "prior"
fig, ax = plot_states(
    data_processor=data_processor,
    states=states_optim,
    standardization=False,
    states_type=state_type,
    states_to_plot=[
        "level",
        "trend",
        "lstm",
        "white noise",
    ],
)
plot_data(
    data_processor=data_processor,
    standardization=False,
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
    standardization=False,
    states_to_plot=[
        "level",
        "trend",
        "lstm",
        "white noise",
    ],
)
plot_data(
    data_processor=data_processor,
    standardization=False,
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
