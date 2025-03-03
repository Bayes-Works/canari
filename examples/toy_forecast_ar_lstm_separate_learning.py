import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from src import (
    LocalLevel,
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    Periodic,
    Autoregression,
    WhiteNoise,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec

# # Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# # Skip resampling data
df = df_raw

# Define parameters
output_col = [0]
num_epoch = 50

data_processor = DataProcess(
    data=df,
    train_split=0.8,
    validation_split=0.2,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

sigma_v = 0.5310128153876991
# sigma_v = np.sqrt(1e-6)/(data_processor.norm_const_std[output_col].item()+1e-10)
noise = WhiteNoise(std_error=sigma_v)
AR = Autoregression(std_error=0.23146312, phi=0.9, mu_states=[0], var_states=[1e-06])

####################################################################################################
######################################## Train LSTM ################################################
####################################################################################################

model = Model(
    LocalTrend(mu_states=[-0.00902307, 0.0], var_states=[1e-12, 1e-12], std_error=0), # True baseline values
    # LocalTrend(),
    LstmNetwork(
        look_back_len=52,
        num_features=1,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    ),
    noise,
    # AR,
)
# model.auto_initialize_baseline_states(train_data["y"][0:52*6])

# Training
scheduled_sigma_v = 5
for epoch in range(num_epoch):
    # Decaying observation's variance
    scheduled_sigma_v = exponential_scheduler(
        curr_v=scheduled_sigma_v, min_v=sigma_v, decaying_factor=0.9, curr_iter=epoch
    )
    noise_index = model.states_name.index("white noise")
    model.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

    mu_validation_preds, std_validation_preds, states = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )

    # Unstandardize the predictions
    mu_validation_preds_unnorm = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.norm_const_mean[output_col],
        data_processor.norm_const_std[output_col],
    )
    std_validation_preds_unnorm = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.norm_const_std[output_col],
    )

    # Calculate the log-likelihood metric
    # mse = metric.mse(
    #     mu_validation_preds_unnorm, data_processor.validation_data[:, output_col].flatten()
    # )

    validation_log_lik = metric.log_likelihood(
        prediction=mu_validation_preds_unnorm,
        observation=data_processor.validation_data[:, output_col].flatten(),
        std=std_validation_preds_unnorm,
    )
    

    # Early-stopping
    model.early_stopping(evaluate_metric=-validation_log_lik, mode="min")

    if epoch == model.optimal_epoch:
            mu_validation_preds_optim = mu_validation_preds.copy()
            std_validation_preds_optim = std_validation_preds.copy()
            states_optim = copy.copy(states)
    if model.stop_training:
        break

# Saved the trained lstm network
params_path = 'saved_params/lstm_net_temp.pth'
model.lstm_net.load_state_dict(model.early_stop_lstm_param)
model.lstm_net.save(filename = params_path)

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")


# ####################################################################################################
# ############################### Use AR to learn mdoel errors #######################################
# ####################################################################################################

# Define AR
AR_process_error_var_prior = 1
var_W2bar_prior = 1
# Case 4: Fully online AR, learn both phi and process error online. phi should converge to ~0.9, W2bar should converge to ~25.
# AR = Autoregression(mu_states=[0, 0, 0, 0, 0, AR_process_error_var_prior],var_states=[1e-06, 0.25, 0, AR_process_error_var_prior, 1e-6, var_W2bar_prior])

# Case 2: AR with process error provided, learn phi online. It should converge to ~0.9
# AR = Autoregression(std_error=0.23146312, mu_states=[0, 0, 0], var_states=[1e-06, 0.25, 0])

# Case 3
AR = Autoregression(phi=0.9, mu_states=[0, 0, 0, AR_process_error_var_prior],var_states=[1e-06, AR_process_error_var_prior, 1e-6, var_W2bar_prior])

model_learn_ar = Model(
    LocalTrend(mu_states=[-0.00902307, 0.0], var_states=[1e-12, 1e-12], std_error=0), # True baseline values
    # LocalTrend(),
    LstmNetwork(
        look_back_len=52,
        num_features=1,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        load_lstm_net = params_path,
    ),
    AR
)

# # model_learn_ar.set_states(model.early_stop_init_mu_states)
# model_learn_ar.mu_states[0:3] = model.early_stop_init_mu_states[0:3]
# model_learn_ar.var_states[0:3, 0:3] = model.early_stop_init_var_states[0:3, 0:3]

# # #
model_learn_ar.filter(data=train_data, train_lstm=False)
model_learn_ar.smoother(data=train_data)
mu_validation_preds_ar, std_validation_preds_ar = model_learn_ar.forecast(validation_data)

# ####################################################################################################
# ########################################### Plot ###################################################
# ####################################################################################################
# state_type = "prior"
state_type = "posterior"

#  Plot states from lstm learner
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(4, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
plot_data(
    data_processor=data_processor,
    normalization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax0,
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds_optim,
    std_validation_pred=std_validation_preds_optim,
    validation_label=[r"$\mu$", f"$\pm\sigma$"],
    sub_plot=ax0,
)
plot_states(
    data_processor=data_processor,
    states=states_optim,
    states_type=state_type,
    states_to_plot=['local level'],
    sub_plot=ax0,
)
ax0.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=states_optim,
    states_type=state_type,
    states_to_plot=['local trend'],
    sub_plot=ax1,
)
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=states_optim,
    states_type=state_type,
    states_to_plot=['lstm'],
    sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=states_optim,
    states_type=state_type,
    states_to_plot=['white noise'],
    # states_to_plot=['autoregression'],
    sub_plot=ax3,
)

#  Plot states from AR learner
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(4, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
plot_data(
    data_processor=data_processor,
    normalization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax0,
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds_ar,
    std_validation_pred=std_validation_preds_ar,
    validation_label=[r"$\mu$", f"$\pm\sigma$"],
    sub_plot=ax0,
)
plot_states(
    data_processor=data_processor,
    states=model_learn_ar.states,
    states_type=state_type,
    states_to_plot=['local level'],
    sub_plot=ax0,
)
ax0.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=model_learn_ar.states,
    states_type=state_type,
    states_to_plot=['local trend'],
    sub_plot=ax1,
)
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=model_learn_ar.states,
    states_type=state_type,
    states_to_plot=['lstm'],
    sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=model_learn_ar.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax3,
)
print(f"States prior at the last step: ")
print(model_learn_ar.states.mu_prior[-1])
plt.show()