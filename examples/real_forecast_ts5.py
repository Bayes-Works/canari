import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import (
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
import copy
from matplotlib import gridspec


# # Read data
data_file = "./data/benchmark_data/CASC_LGA007PIAP_E010_2024_07_obs.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/benchmark_data/CASC_LGA007PIAP_E010_2024_07_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# # Skip resampling data
df = df_raw

# Define parameters
output_col = [0]
num_epoch = 100

data_processor = DataProcess(
    data=df,
    train_split=0.28,
    validation_split=0.07,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
AR_process_error_var_prior = 1e4
var_W2bar_prior = 1e4
AR = Autoregression(mu_states=[0, 0, 0, 0, 0, AR_process_error_var_prior],var_states=[1e-06, 0.01, 0, AR_process_error_var_prior, 1e-6, var_W2bar_prior])

sigma_v = 1e-6

model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=26,
        num_features=1,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    ),
    AR,
    # WhiteNoise(std_error=sigma_v),
)
# model.auto_initialize_baseline_states(train_data["y"][10:10+52+52+52])
model.auto_initialize_baseline_states(train_data["y"][10:52*4])

# Training
scheduled_sigma_v = 1
for epoch in range(num_epoch):

    # # Decaying observation's variance
    # scheduled_sigma_v = exponential_scheduler(
    #     curr_v=scheduled_sigma_v, min_v=sigma_v, decaying_factor=0.9, curr_iter=epoch
    # )
    # noise_index = model.states_name.index("white noise")
    # model.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
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
    model.early_stopping(evaluate_metric=-validation_log_lik, mode="min")
    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
        states_optim = copy.copy(states)
    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

# # Saved the trained lstm network
# params_path = 'saved_params/lstm_CASC_LGA007EFAPRG910_2024_07.pth'
# model.lstm_net.load_state_dict(model.early_stop_lstm_param)
# model.lstm_net.save(filename = params_path)

#  Plot
state_type = "prior"
#  Plot states from AR learner
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
    normalization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax0,
    plot_test_data=False,
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds,
    std_validation_pred=std_validation_preds,
    validation_label=[r"$\mu$", f"$\pm\sigma$"],
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
ax0.set_title("States at the last epoch")
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
if "phi" in model.states_name:
    plot_states(
        data_processor=data_processor,
        states=model.states,
        states_type=state_type,
        states_to_plot=['phi'],
        sub_plot=ax4,
    )
    ax4.set_xticklabels([])
if "W2bar" in model.states_name:
    plot_states(
        data_processor=data_processor,
        states=model.states,
        states_type=state_type,
        states_to_plot=['W2bar'],
        sub_plot=ax5,
    )

state_type = "prior"
#  Plot states from AR learner
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
ax0.set_title("States at the optimal epoch")
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
    states_to_plot=['autoregression'],
    sub_plot=ax3,
)
ax3.set_xticklabels([])
if "phi" in model.states_name:
    plot_states(
        data_processor=data_processor,
        states=states_optim,
        states_type=state_type,
        states_to_plot=['phi'],
        sub_plot=ax4,
    )
    ax4.set_xticklabels([])
if "W2bar" in model.states_name:
    plot_states(
        data_processor=data_processor,
        states=states_optim,
        states_type=state_type,
        states_to_plot=['W2bar'],
        sub_plot=ax5,
    )
print(f"Initial states for optimal epoch: ")
print(model.early_stop_init_mu_states)
print(model.early_stop_init_var_states)
print(f"States prior at the last step: ")
print(states_optim.mu_prior[-1])
plt.show()
