import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from canari.component import LocalTrend, LstmNetwork, Autoregression
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
from pytagi import Normalizer


# # # Read data
data_file = "./data/benchmark_data/test_9_data.csv"
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
    train_split=0.28,
    validation_split=0.07,
    output_col=output_col,
)
data_processor.scale_const_mean, data_processor.scale_const_std = Normalizer.compute_mean_std(
                data_processor.data.iloc[0 : 52].values
            )

num_epoch = 200

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Define AR model
AR_process_error_var_prior = 1e4
var_W2bar_prior = 1e4
AR = Autoregression(mu_states=[0, 0, 0, 0, 0, AR_process_error_var_prior],var_states=[1e-06, 0.01, 0, AR_process_error_var_prior, 0, var_W2bar_prior])
LSTM = LstmNetwork(
        look_back_len=27,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

model = Model(
    LocalTrend(),
    LSTM,
    AR,
)
# model._mu_local_level = 0
model.auto_initialize_baseline_states(train_data["y"][0:52*4])


# Training
for epoch in range(num_epoch):
    mu_validation_preds, std_validation_preds, states = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )
    model.set_memory(states=model.states, time_step=0)

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
    mse = metric.mse(
        mu_validation_preds_unnorm, validation_obs
    )

    validation_log_lik = metric.log_likelihood(
        prediction=mu_validation_preds_unnorm,
        observation=validation_obs,
        std=std_validation_preds_unnorm,
    )

    # Early-stopping
    model.early_stopping(evaluate_metric=-validation_log_lik, mode="min",
                        #  current_epoch=epoch, max_epoch=num_epoch, skip_epoch = 150)
                         current_epoch=epoch, max_epoch=num_epoch)
    
    # model.early_stopping(evaluate_metric=mse, mode="min")


    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds.copy()
        std_validation_preds_optim = std_validation_preds.copy()
        states_optim = copy.copy(states)
    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")


model_dict = model.get_dict()
model_dict['states_optimal'] = states_optim
model_dict['early_stop_init_mu_states'] = model.early_stop_init_mu_states
model_dict['early_stop_init_var_states'] = model.early_stop_init_var_states

# # Save model_dict to local
# import pickle
# with open("saved_params/real_ts9_tsmodel_raw.pkl", "wb") as f:
#     pickle.dump(model_dict, f)

####################################################################
######################### Pretrained model #########################
####################################################################
phi_index = model_dict["states_name"].index("phi")
W2bar_index = model_dict["states_name"].index("W2bar")
autoregression_index = model_dict["states_name"].index("autoregression")

print("phi_AR =", model_dict['states_optimal'].mu_prior[-1][phi_index].item())
print("sigma_AR =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()))
pretrained_model = Model(
    # LocalTrend(mu_states=model_dict["mu_states"][0:2].reshape(-1), var_states=np.diag(model_dict["var_states"][0:2, 0:2])),
    LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                   mu_states=[model_dict["mu_states"][autoregression_index].item()], 
                   var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
)

pretrained_model.lstm_net.load_state_dict(model.lstm_net.state_dict())

pretrained_model.filter(normalized_data,train_lstm=False)
pretrained_model.smoother()

#  Plot
state_type = "prior"
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
    states_to_plot=['autoregression'],
    sub_plot=ax3,
)
ax3.set_xticklabels([])
# plt.show()

state_type = "prior"
# Plot states from AR learner
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(6, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
ax4 = plt.subplot(gs[4])
ax5 = plt.subplot(gs[5])
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
  states_to_plot=['autoregression'],
  sub_plot=ax3,
)
ax3.set_xticklabels([])
if "phi" in model.states_name:
  plot_states(
    data_processor=data_processor,
    standardization=True,
    states=states_optim,
    states_type=state_type,
    states_to_plot=['phi'],
    sub_plot=ax4,
  )
  ax4.set_xticklabels([])
if "W2bar" in model.states_name:
  plot_states(
    data_processor=data_processor,
    standardization=True,
    states=states_optim,
    states_type=state_type,
    states_to_plot=['W2bar'],
    sub_plot=ax5,
  )
plt.show()