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

# Change the trend of data
trend_true = 0.
df_raw.values+=np.arange(len(df_raw.values),dtype="float64").reshape(-1, 1)*trend_true

# # Skip resampling data
df = df_raw

# Define parameters
output_col = [0]
num_epoch = 200
train_val_split = [0.6, 0.2]

data_processor = DataProcess(
    data=df,
    time_covariates=["week_of_year"],
    train_split=train_val_split[0],
    validation_split=train_val_split[1],
    # train_split=1,
    # validation_split=0,
    output_col=output_col,
)

trend_true_norm = trend_true/(data_processor.norm_const_std[output_col].item() + 1e-10)
level_true_norm = (5.0 - data_processor.norm_const_mean[output_col].item())/(data_processor.norm_const_std[output_col].item() + 1e-10)
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()
# Unstandardize the validation_data['x']
validation_data_x_unnorm = normalizer.unstandardize(
    validation_data['x'],
    data_processor.norm_const_mean[1],
    data_processor.norm_const_std[1],
)
time_covariate_info = {'initial_time_covariate': validation_data_x_unnorm[-1].item(),
                        'mu': data_processor.norm_const_mean[1], 
                        'std': data_processor.norm_const_std[1]}

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
AR = Autoregression(mu_states=[0, 0, 0, 0, 0, AR_process_error_var_prior],var_states=[1e-06, 0.01, 0, AR_process_error_var_prior, 0, var_W2bar_prior])

model = Model(
    LocalTrend(mu_states=[level_true_norm, trend_true_norm], var_states=[1e-12, 1e-12], std_error=0), # True baseline values
    # LocalTrend(),
    LSTM,
    AR,
)
model._mu_local_level = level_true_norm
# model.auto_initialize_baseline_states(train_data["y"][0:51])


# Training
for epoch in range(num_epoch):
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

    # Calculate the evaluation metric
    mse = metric.mse(
        mu_validation_preds_unnorm, data_processor.validation_data[:, output_col].flatten()
    )

    validation_log_lik = metric.log_likelihood(
        prediction=mu_validation_preds_unnorm,
        observation=data_processor.validation_data[:, output_col].flatten(),
        std=std_validation_preds_unnorm,
    )

    # Early-stopping
    # model.early_stopping(evaluate_metric=-validation_log_lik, mode="min", skip_epoch=50)
    model.early_stopping(evaluate_metric=-validation_log_lik, mode="min")
    # model.early_stopping(evaluate_metric=mse, mode="min")


    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds.copy()
        std_validation_preds_optim = std_validation_preds.copy()
        states_optim = copy.copy(states)
    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")


model_dict = model.save_model_dict()
model_dict['states_optimal'] = states_optim

# Save model_dict to local
import pickle
with open("saved_params/toy_model_dict.pkl", "wb") as f:
    pickle.dump(model_dict, f)

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
    normalization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax0,
    plot_test_data=False,
)
plot_prediction(
  data_processor=data_processor,
  mean_validation_pred=mu_validation_preds_optim,
  std_validation_pred=std_validation_preds_optim,
  sub_plot=ax0,
)
plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type=state_type,
    states_to_plot=['local level'],
    sub_plot=ax0,
)
ax0.set_xticklabels([])
ax0.set_title("Hidden states estimated by the pre-trained model")
plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type=state_type,
    states_to_plot=['local trend'],
    sub_plot=ax1,
)
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type=state_type,
    states_to_plot=['lstm'],
    sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax3,
)
ax3.set_xticklabels([])
plt.show()


####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
import pickle
with open("saved_params/toy_model_dict.pkl", "rb") as f:
    pretrained_model_dict = pickle.load(f)
print("phi_AR =", pretrained_model_dict['states_optimal'].mu_prior[-1][pretrained_model_dict['phi_index']].item())
print("sigma_AR =", np.sqrt(pretrained_model_dict['states_optimal'].mu_prior[-1][pretrained_model_dict['W2bar_index']].item()))
pretrained_model = Model(
    LocalTrend(mu_states=[level_true_norm, trend_true_norm], var_states=[1e-12, 1e-12], std_error=0), # True baseline values
    # LocalTrend(mu_states=pretrained_model_dict["mu_states"][0:2].reshape(-1), var_states=np.diag(pretrained_model_dict["var_states"][0:2, 0:2])),
    # LocalTrend(mu_states=pretrained_model_dict["mu_states"][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(
        std_error=np.sqrt(pretrained_model_dict['states_optimal'].mu_prior[-1][pretrained_model_dict['W2bar_index']].item()), 
        phi=pretrained_model_dict['states_optimal'].mu_prior[-1][pretrained_model_dict['phi_index']].item(), 
        # std_error = 0.23146312, # True error
        # phi=0.9,                # True phi
        mu_states=[pretrained_model_dict["mu_states"][pretrained_model_dict['autoregression_index']].item()], 
        var_states=[pretrained_model_dict["var_states"][pretrained_model_dict['autoregression_index'], pretrained_model_dict['autoregression_index']].item()]),
)

pretrained_model.lstm_net.load_state_dict(pretrained_model_dict["lstm_network_params"])

pretrained_model.filter(train_data,train_lstm=False)
pretrained_model.filter(validation_data,train_lstm=False)
generated_ts = pretrained_model.generate(num_time_series=5, num_time_steps=len(normalized_data['y'])*2, time_covariates=data_processor.time_covariates, time_covariate_info=time_covariate_info)

time_idx = np.arange(len(np.concatenate((train_data['y'].reshape(-1), validation_data['y'].reshape(-1), generated_ts[0]))))
train_val_idx = np.arange(len(np.concatenate((train_data['y'].reshape(-1), validation_data['y'].reshape(-1)))))
train_end_idx = int(train_val_split[0]/np.sum(train_val_split)*len(train_val_idx))
val_end_idx = len(train_val_idx)
# Plot generated_ts[0]
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 1)
ax0 = plt.subplot(gs[0])
for i in range(len(generated_ts)):
    # Concatenate the generated time series at the end of train_data['y'], validation_data['y']
    ax0.plot(np.concatenate((train_data['y'].reshape(-1), validation_data['y'].reshape(-1), generated_ts[i])))
ax0.plot(np.concatenate((train_data['y'].reshape(-1), validation_data['y'].reshape(-1))),color='k',label='original')
# Plot red dashed vertical line at the end of train_data['y'], validation_data['y']
ax0.axvline(x=len(train_data['y'])+len(validation_data['y']), color='r', linestyle='--')
ax0.axvspan(time_idx[train_end_idx], time_idx[val_end_idx], color="green", alpha=0.1)
ax0.axvspan(time_idx[val_end_idx], time_idx[-1], color="k", alpha=0.1)
ax0.legend(loc='lower left')
plt.show()