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
import pickle


# # Read data
data_file = "./data/toy_time_series/synthetic_simple_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

# LT anomaly
anm_start_index = 52*10
anm_mag = 0.1/52
# anm_baseline = np.linspace(0, 3, num=len(df_raw))
anm_baseline = np.arange(len(df_raw)) * anm_mag
# Set the first 52*12 values in anm_baseline to be 0
anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
anm_baseline[:anm_start_index] = 0

df_raw = df_raw.add(anm_baseline, axis=0)

data_file_time = "./data/toy_time_series/synthetic_simple_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=0.289,
    validation_split=0.0693*2,
    output_col=output_col,
)

num_epoch = 200

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

with open("saved_params/toy_simple_model_confidence.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=19,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

model = Model(
    # LocalTrend(mu_states=[0, 0], var_states=[1e-12, 1e-12]),
    LocalTrend(mu_states=model_dict['early_stop_init_mu_states'][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(), 
                   mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()], 
                   var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
)
model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

mu_all_preds_optim, std_all_preds_optim, states_optim = model.filter(normalized_data)
model.smoother(train_data)

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
from src.data_visualization import determine_time
time = determine_time(data_processor, len(normalized_data["y"]))
plot_data(
  data_processor=data_processor,
  normalization=True,
  plot_column=output_col,
  validation_label="y",
  sub_plot=ax0,
)
ax0.plot(time, mu_all_preds_optim)
ax0.fill_between(
  time,
  mu_all_preds_optim - std_all_preds_optim,
  mu_all_preds_optim + std_all_preds_optim,
  alpha=0.2,
  color="gray",
)
ax0.axvline(x=time[anm_start_index], color='r', linestyle='--')
# plot_prediction(
#   data_processor=data_processor,
#   mean_validation_pred=mu_all_preds_optim,
#   std_validation_pred=std_all_preds_optim,
# #   validation_label=[r"$\mu$", f"$\pm\sigma$"],
#   sub_plot=ax0,
# )
plot_states(
  data_processor=data_processor,
  states=states_optim,
  states_type=state_type,
  states_to_plot=['local level'],
  sub_plot=ax0,
)
ax0.set_xticklabels([])
ax0.set_title("Hidden states at the optimal epoch in training")
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
mu_state_prior = states_optim.get_mean(states_type='prior')['lstm']
std_state_prior = states_optim.get_std(states_type='prior')['lstm']
mu_state_posterior = states_optim.get_mean(states_type='posterior')['lstm']
std_state_posterior = states_optim.get_std(states_type='posterior')['lstm']
# Compute the KL divergence between the prior and posterior distributions
kl_divergence = metric.kl_divergence(mu_state_prior, std_state_prior)
ax4.plot(time, kl_divergence)
ax4.axvline(x=time[anm_start_index], color='r', linestyle='--')

plt.show()