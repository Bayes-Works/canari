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
    SKF,
    plot_data,
    plot_prediction,
    plot_states,
)
from src.hsl_detection import hsl_detection
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
import pickle
import src.common as common

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params)


# # Read data
data_file = "./data/toy_time_series/synthetic_simple_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

anm_start_index = 52*10

# LT anomaly
# anm_mag = 0.010416667/10
anm_mag = 0.3/52
# anm_baseline = np.linspace(0, 3, num=len(df_raw))
anm_baseline = np.arange(len(df_raw)) * anm_mag
# Set the first 52*12 values in anm_baseline to be 0
anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
anm_baseline[:anm_start_index] = 0

# # LL anomaly
# anm_mag = 0.5
# anm_baseline = np.zeros_like(df_raw)
# anm_baseline[anm_start_index:] += anm_mag

df_raw = df_raw.add(anm_baseline, axis=0)

data_file_time = "./data/toy_time_series/synthetic_simple_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Data pre-processing
output_col = [0]
train_split=0.289
validation_split=0.0693*2

# Remove the last 52*5 rows in df_raw
train_split = train_split * len(df_raw) / len(df_raw[:-52*5])
validation_split = validation_split * len(df_raw) / len(df_raw[:-52*5])
df_raw = df_raw[:-52*5]

data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=train_split,
    validation_split=validation_split,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()


####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
with open("saved_params/toy_simple_model.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=19,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

print("phi_AR =", model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item())
print("sigma_AR =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()))


norm_model = Model(
    LocalTrend(mu_states=model_dict['early_stop_init_mu_states'][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(), 
                   mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()], 
                   var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
)
norm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

abnorm_model = Model(
    LocalAcceleration(mu_states=[model_dict['early_stop_init_mu_states'][0].item(), model_dict['early_stop_init_mu_states'][1].item(), 0], var_states=[1e-12, 1e-12, 1e-4]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()),
                   phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(),
                   mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()],
                   var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
)

skf = SKF(
    norm_model=norm_model,
    abnorm_model=abnorm_model,
    std_transition_error=1e-4,
    norm_to_abnorm_prob=1e-4,
    abnorm_to_norm_prob=1e-1,
    norm_model_prior_prob=0.99,
)

# # Anomaly Detection
filter_marginal_abnorm_prob, states = skf.filter(data=normalized_data)
# smooth_marginal_abnorm_prob, states = skf.smoother(data=normalized_data)

from src.data_visualization import determine_time
time = determine_time(data_processor, len(normalized_data["y"]))

# Remove the state estimates in training
remove_until_index = data_processor.validation_start
base_model_mu_prior = np.array(states.mu_prior[remove_until_index:])
base_model_var_prior = np.array(states.var_prior[remove_until_index:])
data = data_processor.normalize_data()
data = data[:, 0].flatten()
data = data[remove_until_index:]
# hsl_tsad_agent.base_model.states.mu_posterior = hsl_tsad_agent.base_model.states.mu_posterior[remove_until_index:]
# hsl_tsad_agent.base_model.states.var_posterior = hsl_tsad_agent.base_model.states.var_posterior[remove_until_index:]
# hsl_tsad_agent.base_model.states.cov_states = hsl_tsad_agent.base_model.states.cov_states[remove_until_index:]
# hsl_tsad_agent.base_model.states.mu_smooth = hsl_tsad_agent.base_model.states.mu_smooth[remove_until_index:]
# hsl_tsad_agent.base_model.states.var_smooth = hsl_tsad_agent.base_model.states.var_smooth[remove_until_index:]

# hsl_tsad_agent.drift_model.states.mu_prior = hsl_tsad_agent.drift_model.states.mu_prior[remove_until_index:]
# hsl_tsad_agent.drift_model.states.var_prior = hsl_tsad_agent.drift_model.states.var_prior[remove_until_index:]
# hsl_tsad_agent.drift_model.states.mu_posterior = hsl_tsad_agent.drift_model.states.mu_posterior[remove_until_index:]
# hsl_tsad_agent.drift_model.states.var_posterior = hsl_tsad_agent.drift_model.states.var_posterior[remove_until_index:]
# hsl_tsad_agent.drift_model.states.cov_states = hsl_tsad_agent.drift_model.states.cov_states[remove_until_index:]
# hsl_tsad_agent.drift_model.states.mu_smooth = hsl_tsad_agent.drift_model.states.mu_smooth[remove_until_index:]
# hsl_tsad_agent.drift_model.states.var_smooth = hsl_tsad_agent.drift_model.states.var_smooth[remove_until_index:]

p_anm_all = filter_marginal_abnorm_prob[remove_until_index:]

if (np.array(p_anm_all) > 0.5).any():
    anm_detected_index = np.where(np.array(p_anm_all) > 0.5)[0]
else:
    anm_detected_index = len(p_anm_all)

print(len(anm_detected_index))

time = time[remove_until_index:]
anm_start_index = anm_start_index - remove_until_index

#  Plot
state_type = "prior"
#  Plot states from pretrained model
fig = plt.figure(figsize=(5, 3))
gs = gridspec.GridSpec(3, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])

# plot_data(
#     data_processor=data_processor,
#     plot_train_data=False,
#     normalization=True,
#     plot_column=output_col,
#     validation_label="y",
#     sub_plot=ax0,
# )

ax0.plot(time, data, color='k')
ax0.plot(time, base_model_mu_prior[:,0,0])
ax0.fill_between(
    time,
    base_model_mu_prior[:,0,0] - np.sqrt(base_model_var_prior[:,0,0]),
    base_model_mu_prior[:,0,0] + np.sqrt(base_model_var_prior[:,0,0]),
    alpha=0.5,
    color="gray",
)
ax0.axvline(x=time[anm_start_index], color='r', linestyle='--', label="trend change")
for i in range(len(anm_detected_index)):
    if i == 0:
        ax0.axvline(x=time[anm_detected_index[i]], color='k', linestyle='--', label="detect")
    else:
        ax0.axvline(x=time[anm_detected_index[i]], color='k', linestyle='--')
ax0.set_ylabel('level')
ax0.set_yticks([0, 2.5])
ax0.set_xticks([time[int(len(time)*1/9)-1], time[int(len(time)*3/9)-1],time[int(len(time)*5/9)-1],time[int(len(time)*7/9)-1],time[int(-1)]])
ax0.legend(bbox_to_anchor=(1.02, 1.8), frameon=False, ncol=2)
ax0.set_xticklabels([])

ax1.plot(time, base_model_mu_prior[:,1,0], label="LT")
ax1.fill_between(
    time,
    base_model_mu_prior[:,1,0] - np.sqrt(base_model_var_prior[:,1,1]),
    base_model_mu_prior[:,1,0] + np.sqrt(base_model_var_prior[:,1,1]),
    alpha=0.5,
    color="gray",
)
ax1.set_ylabel('trend')
ax1.set_yticks([0, 0.01, 0.02])
ax1.set_xticks([time[int(len(time)*1/9)-1], time[int(len(time)*3/9)-1],time[int(len(time)*5/9)-1],time[int(len(time)*7/9)-1],time[int(-1)]])
ax1.set_xticklabels([])

ax2.plot(time, p_anm_all)
ax2.set_ylabel(r'$p_{\mathrm{anm}}$')
ax2.set_xlim(ax0.get_xlim())
# ax2.axvline(x=time[anm_start_index], color='r', linestyle='--')
ax2.set_ylim(-0.05, 1.05)
ax2.set_yticks([0, 1])
ax2.set_xticks([time[int(len(time)*1/9)-1], time[int(len(time)*3/9)-1],time[int(len(time)*5/9)-1],time[int(len(time)*7/9)-1],time[int(-1)]])
ax2.set_xticklabels(['2016', '2018', '2020', '2022', '2024'])
ax2.set_xlim(ax0.get_xlim())

plt.tight_layout(h_pad=0.5, w_pad=0.1)
plt.savefig('hsl.png', dpi=300)

plt.show()