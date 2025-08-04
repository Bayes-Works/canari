import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from canari.component import LocalTrend, LstmNetwork, Autoregression, WhiteNoise
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
import pickle

data_file = "./data/benchmark_data/test_5_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=0.289,
    validation_split=0.0693,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

LSTM = LstmNetwork(
        look_back_len=17,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
    )

model = Model(
    LocalTrend(),
    LSTM,
    WhiteNoise(std_error=0.028761848586134717),
)
# model._mu_local_level = 0
model.auto_initialize_baseline_states(train_data["y"][0:52 * 3 + 1])

states_optim = None
mu_validation_preds_optim = None
std_validation_preds_optim = None
num_epoch = 50
# Training 
for epoch in range(num_epoch):
    mu_validation_preds, std_validation_preds, states = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )

    # Unstandardize the predictions
    mu_validation_preds_unnorm = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.scale_const_mean[output_col],
        data_processor.scale_const_std[output_col],
    )
    std_validation_preds_unnorm = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.scale_const_std[output_col],
    )

    # Calculate the evaluation metric
    validation_obs = data_processor.get_data("validation").flatten()
    validation_log_lik = metric.log_likelihood(
        prediction=mu_validation_preds_unnorm,
        observation=validation_obs,
        std=std_validation_preds_unnorm,
    )

    # Early-stopping
    model.early_stopping(evaluate_metric=-validation_log_lik, mode="min",
                         current_epoch=epoch, max_epoch=num_epoch)

    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds.copy()
        std_validation_preds_optim = std_validation_preds.copy()
        states_optim = copy.copy(states)
        noise_index = model.get_states_index("white noise")
        sigma_v_optim = np.sqrt(model.process_noise_matrix[noise_index, noise_index])
        print("The optimal sigma_v is:", sigma_v_optim)
    model.set_memory(states=states, time_step=0)
    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

################## Run the training data with optimal model ##################
# # Model the posterior of white noise using AR
op_model_dict = model.get_dict()
# Print the keys of the model_dict
op_model_dict['states_optimal'] = states_optim
op_model_dict['sigma_v_optimal'] = sigma_v_optim
op_model_dict['early_stop_init_mu_states'] = model.early_stop_init_mu_states
op_model_dict['early_stop_init_var_states'] = model.early_stop_init_var_states

op_model = Model(
    LocalTrend(mu_states=op_model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=np.diag(op_model_dict['states_optimal'].var_prior[0][0:2, 0:2])),
    LSTM,
    WhiteNoise(std_error=op_model_dict["sigma_v_optimal"]),
)

op_model.lstm_net.load_state_dict(model.lstm_net.state_dict())
op_model.filter(train_data,train_lstm=False)

AR_process_error_var_prior = 1
var_W2bar_prior = 1
AR = Autoregression(mu_states=[0, 0.5, 0, 0, 0, AR_process_error_var_prior],var_states=[1e-06, 0.25, 0, AR_process_error_var_prior, 0, var_W2bar_prior])
ar_model = Model(
    AR,
)

# Get the white noise states from the optimal epoch
mu_lstm_posterior = op_model.states.get_mean(states_type="posterior", states_name="lstm", standardization=True)
mu_lstm_prior = op_model.states.get_mean(states_type="prior", states_name="lstm", standardization=True)
lstm_residual = mu_lstm_posterior - mu_lstm_prior
white_noise_mu = op_model.states.get_mean(
                states_type="posterior", states_name="white noise", standardization=True
            )
white_noise_std = op_model.states.get_std(
    states_type="posterior", states_name="white noise", standardization=True
)

sum_residual_mu = lstm_residual + white_noise_mu
sum_residual_std = white_noise_std

# Plot the white noise states
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sum_residual_mu, label="White noise mean")
ax.fill_between(
    np.arange(len(sum_residual_mu)),
    sum_residual_mu - sum_residual_std,
    sum_residual_mu + sum_residual_std,
    alpha=0.2,
    label="White noise std",
)

# Put white noise states into a data_processor
residual_obs = copy.deepcopy(train_data)
# Replayce residual_obs['y'] with white noise mu
residual_obs['y'] = sum_residual_mu.reshape(-1, 1)
residual_obs['y_var'] = sum_residual_std.reshape(-1, 1) ** 2

# Run ar_model on the white noise states using filter
ar_model.filter(residual_obs)

# Print the phi and sigma_ar
print("The phi is:", ar_model.states.get_mean(states_type="prior", states_name="phi")[-1])
print("The sigma_ar is:", np.sqrt(ar_model.states.get_mean(states_type="prior", states_name="W2bar")[-1]))


# Plot states in ar_model
state_type = "posterior"
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(4, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=ar_model.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax0,
)
ax0.set_xticklabels([])
ax0.set_title("Posterior")
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=ar_model.states,
    states_type=state_type,
    states_to_plot=['phi'],
    sub_plot=ax1,
)
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=ar_model.states,
    states_type=state_type,
    states_to_plot=['AR_error'],
    sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=ar_model.states,
    states_type=state_type,
    states_to_plot=['W2bar'],
    sub_plot=ax3,
)
ax3.set_xticklabels([])

state_type = "prior"
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(4, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=ar_model.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax0,
)
ax0.set_xticklabels([])
ax0.set_title("Prior")
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=ar_model.states,
    states_type=state_type,
    states_to_plot=['phi'],
    sub_plot=ax1,
)
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=ar_model.states,
    states_type=state_type,
    states_to_plot=['AR_error'],
    sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=ar_model.states,
    states_type=state_type,
    states_to_plot=['W2bar'],
    sub_plot=ax3,
)
ax3.set_xticklabels([])
plt.show()

###################### Save the parameters for two models ######################
model_dict = model.get_dict()
# Print the keys of the model_dict
model_dict['states_optimal'] = states_optim
model_dict['sigma_v_optimal'] = sigma_v_optim
model_dict['early_stop_init_mu_states'] = model.early_stop_init_mu_states
model_dict['early_stop_init_var_states'] = model.early_stop_init_var_states
model_dict['phi_ar'] = ar_model.states.get_mean(states_type="prior", states_name="phi")[-1]
model_dict['sigma_ar'] = np.sqrt(ar_model.states.get_mean(states_type="prior", states_name="W2bar")[-1])

print("statationary_ar_error_var =", np.sqrt(model_dict['sigma_ar']**2/(1 - model_dict['phi_ar']**2)))

# Save model_dict to local
import pickle
with open("saved_params/real_ts5_lstmres.pkl", "wb") as f:
    pickle.dump(model_dict, f)

####################################################################
######################### Pretrained model #########################
####################################################################
pretrained_model = Model(
    LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=np.diag(model_dict['states_optimal'].var_prior[0][0:2, 0:2])),
    LSTM,
    WhiteNoise(std_error=model_dict["sigma_v_optimal"]),
)

pretrained_model.lstm_net.load_state_dict(model.lstm_net.state_dict())

pretrained_model.filter(all_data,train_lstm=False)
# pretrained_model.smoother()

#  Plot
state_type = "posterior"
#  Plot states from pretrained model
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(4, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
plot_data(
    data_processor=data_processor,
    standardization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax0,
)
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=pretrained_model.states,
    states_type=state_type,
    states_to_plot=['level'],
    sub_plot=ax0,
)
ax0.set_xticklabels([])
ax0.set_title("Hidden states estimated by the pre-trained model")
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=pretrained_model.states,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax1,
)
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=pretrained_model.states,
    states_type=state_type,
    states_to_plot=['lstm'],
    sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=pretrained_model.states,
    states_type=state_type,
    states_to_plot=['white noise'],
    sub_plot=ax3,
)
ax3.set_xticklabels([])
# plt.show()

state_type = "prior"
# Plot states from AR learner
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(4, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
# ax4 = plt.subplot(gs[4])
# ax5 = plt.subplot(gs[5])
plot_data(
  data_processor=data_processor,
  standardization=True,
  plot_column=output_col,
  validation_label="y",
  sub_plot=ax0,
)
plot_prediction(
  data_processor=data_processor,
  mean_validation_pred=mu_validation_preds_optim,
  std_validation_pred=std_validation_preds_optim,
#   validation_label=[r"$\mu$", f"$\pm\sigma$"],
  sub_plot=ax0,
)
plot_states(
  data_processor=data_processor,
  standardization=True,
  states=states_optim,
  states_type=state_type,
  states_to_plot=['level'],
  sub_plot=ax0,
)
ax0.set_xticklabels([])
ax0.set_title("Hidden states at the optimal epoch in training")
plot_states(
  data_processor=data_processor,
  standardization=True,
  states=states_optim,
  states_type=state_type,
  states_to_plot=['trend'],
  sub_plot=ax1,
)
ax1.set_xticklabels([])
plot_states(
  data_processor=data_processor,
  standardization=True,
  states=states_optim,
  states_type=state_type,
  states_to_plot=['lstm'],
  sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
  data_processor=data_processor,
  standardization=True,
  states=states_optim,
  states_type=state_type,
  states_to_plot=['white noise'],
  sub_plot=ax3,
)

state_type = "posterior"
# Plot states from AR learner
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(4, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
# ax4 = plt.subplot(gs[4])
# ax5 = plt.subplot(gs[5])
plot_data(
  data_processor=data_processor,
  standardization=True,
  plot_column=output_col,
  validation_label="y",
  sub_plot=ax0,
)
plot_prediction(
  data_processor=data_processor,
  mean_validation_pred=mu_validation_preds_optim,
  std_validation_pred=std_validation_preds_optim,
#   validation_label=[r"$\mu$", f"$\pm\sigma$"],
  sub_plot=ax0,
)
plot_states(
  data_processor=data_processor,
  standardization=True,
  states=states_optim,
  states_type=state_type,
  states_to_plot=['level'],
  sub_plot=ax0,
)
ax0.set_xticklabels([])
ax0.set_title("Hidden states at the optimal epoch in training")
plot_states(
  data_processor=data_processor,
  standardization=True,
  states=states_optim,
  states_type=state_type,
  states_to_plot=['trend'],
  sub_plot=ax1,
)
ax1.set_xticklabels([])
plot_states(
  data_processor=data_processor,
  standardization=True,
  states=states_optim,
  states_type=state_type,
  states_to_plot=['lstm'],
  sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
  data_processor=data_processor,
  standardization=True,
  states=states_optim,
  states_type=state_type,
  states_to_plot=['white noise'],
  sub_plot=ax3,
)
plt.show()