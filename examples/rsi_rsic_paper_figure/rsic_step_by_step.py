import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from canari.component import LocalTrend, LstmNetwork, Autoregression
from canari import (
    DataProcess,
    Model,
    common,
    plot_data,
    plot_prediction,
    plot_states,
)
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
import pickle
from pytagi import Normalizer
from src.hsl_classification_2classes_rsic import hsl_classification

from matplotlib import ticker
import matplotlib.dates as mdates
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          'lines.linewidth' : 1,
          }
plt.rcParams.update(params)


# # # Read data
data_file = "./data/toy_time_series/syn_data_anmtype_simple_phi05.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

# LT anomaly
time_anomaly = 52*7
anm_mag = 10/52
anm_baseline = np.arange(len(df_raw)) * anm_mag
anm_baseline[time_anomaly:] -= anm_baseline[time_anomaly]
anm_baseline[:time_anomaly] = 0
df_raw = df_raw.add(anm_baseline, axis=0)


# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=0.3,
    validation_split=0.1,
    output_col=output_col,
)

# Normalize anm_mag
normed_anm_mag =  anm_mag / data_processor.scale_const_std[0]

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()
train_val_data = copy.deepcopy(normalized_data)
train_val_data["x"] = train_val_data["x"][0:data_processor.validation_end, :]
train_val_data["y"] = train_val_data["y"][0:data_processor.validation_end, :]

# Define AR model
AR_process_error_var_prior = 1e2
var_W2bar_prior = 1e2
AR = Autoregression(mu_states=[0, 0, 0, 0, 0, AR_process_error_var_prior],var_states=[1e-06, 0.01, 0, AR_process_error_var_prior, 0, var_W2bar_prior])
LSTM = LstmNetwork(
        look_back_len=52,
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
model.auto_initialize_baseline_states(train_data["y"][0 : 52 * 3])



# Load model_dict to local
import pickle
with open("saved_params/paper_example.pkl", "rb") as f:
    model_dict = pickle.load(f)

####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
with open("saved_params/ssm_ts_anmtype_simple_phi05.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=52,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

phi_index = model_dict["states_name"].index("phi")
W2bar_index = model_dict["states_name"].index("W2bar")
autoregression_index = model_dict["states_name"].index("autoregression")

print("phi_AR =", model_dict['states_optimal'].mu_prior[-1][phi_index].item())
print("sigma_AR =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()))
pretrained_model = Model(
    # LocalTrend(mu_states=model_dict["mu_states"][0:2].reshape(-1), var_states=np.diag(model_dict["var_states"][0:2, 0:2])),
    LocalTrend(mu_states=[0, 0], var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                   mu_states=[model_dict['states_optimal'].mu_prior[0][autoregression_index].item()], 
                   var_states=[model_dict['states_optimal'].var_prior[0][autoregression_index, autoregression_index].item()]),
)
gen_model = Model(
    # LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=np.diag(model_dict['states_optimal'].var_prior[0][0:2, 0:2])),
    LocalTrend(mu_states=[0, 0], var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(phi=model_dict['gen_phi_ar'], std_error=model_dict['gen_sigma_ar'],
                   mu_states=[model_dict['states_optimal'].mu_prior[0][autoregression_index].item()], 
                   var_states=[model_dict['states_optimal'].var_prior[0][autoregression_index, autoregression_index].item()]),
)

pretrained_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])
gen_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

ltd_error = 1e-5

hsl_tsad_agent = hsl_classification(base_model=pretrained_model, generate_model=gen_model, data_processor=data_processor, drift_model_process_error_std=ltd_error, y_std_scale = 1)

# Get flexible drift model from the beginning
hsl_tsad_agent_pre = hsl_classification(base_model=pretrained_model.load_dict(pretrained_model.get_dict()), generate_model=gen_model, data_processor=data_processor, drift_model_process_error_std=ltd_error)
hsl_tsad_agent_pre.filter(train_data)
hsl_tsad_agent_pre.filter(validation_data)
hsl_tsad_agent.drift_model.var_states = hsl_tsad_agent_pre.drift_model.var_states
hsl_tsad_agent.init_drift_model.var_states = hsl_tsad_agent_pre.drift_model.var_states

hsl_tsad_agent.mu_LTd = 2.83129300946429e-07
hsl_tsad_agent.LTd_std = 4.9551180011919054e-05
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std * 1)
hsl_tsad_agent.detection_threshold = 0.5545706309885293
hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class = 2.0454983e-05, 0.000387628
hsl_tsad_agent.mean_target_lt_model, hsl_tsad_agent.std_target_lt_model = np.array([0.00014448, 0.01961236]), np.array([0.00675291, 0.8995139])
hsl_tsad_agent.mean_target_ll_model, hsl_tsad_agent.std_target_ll_model = np.array([0.00261593]), np.array([0.6945869])

# mu_ar_preds_all, std_ar_preds_all = [], []
# mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(train_data, buffer_LTd=True)
# mu_ar_preds_all = np.hstack((mu_ar_preds_all, mu_ar_preds.flatten()))
# std_ar_preds_all = np.hstack((std_ar_preds_all, std_ar_preds.flatten()))
# mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(validation_data, buffer_LTd=True)
# mu_ar_preds_all = np.hstack((mu_ar_preds_all, mu_ar_preds.flatten()))
# std_ar_preds_all = np.hstack((std_ar_preds_all, std_ar_preds.flatten()))

# hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class = 2.0454983e-05, 0.000387628
# hsl_tsad_agent.mean_target_lt_model, hsl_tsad_agent.std_target_lt_model = np.array([0.00014448, 0.01961236]), np.array([0.00675291, 0.8995139])
# hsl_tsad_agent.mean_target_ll_model, hsl_tsad_agent.std_target_ll_model = np.array([0.00261593]), np.array([0.6945869])


mu_y_preds, std_y_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(train_data, buffer_LTd=True)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(validation_data, buffer_LTd=True)
mu_y_preds = np.append(mu_y_preds, mu_obs_preds)
std_y_preds = np.append(std_y_preds, std_obs_preds)

hsl_tsad_agent.learn_intervention(training_samples_path='data/anm_type_class_train_samples/classifier_learn_samples_syn_simple_phi05.csv', 
                                    load_lt_model_path='saved_params/NN_intervention_LT_model_syn_simple_phi05.pkl', 
                                    load_ll_model_path='saved_params/NN_intervention_LL_model_syn_simple_phi05.pkl', 
                                    max_training_epoch=50)
# # Regular detection
# mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data, apply_intervention=False)
# mu_y_preds = np.append(mu_y_preds, mu_obs_preds)
# std_y_preds = np.append(std_y_preds, std_obs_preds)

# # # Regular filter
# mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(test_data)
# mu_y_preds = np.append(mu_y_preds, mu_obs_preds)
# std_y_preds = np.append(std_y_preds, std_obs_preds)

# Manual intervention on the test set
intervention_time_global = 415
confidence_enough_step = 550
# True anomaly baseline
true_correction_trend = normed_anm_mag
true_correction_level = true_correction_trend * (intervention_time_global - time_anomaly)
dummy_level_correction = (true_correction_level + true_correction_trend * (confidence_enough_step - time_anomaly)) / 2 * 0.7

intervention_time = intervention_time_global - data_processor.validation_end
test_data_before_itv = copy.deepcopy(test_data)
test_data_after_itv = copy.deepcopy(test_data)
test_data_before_itv["x"] = test_data_before_itv["x"][:intervention_time, :]
test_data_before_itv["y"] = test_data_before_itv["y"][:intervention_time, :]
test_data_after_itv["x"] = test_data_after_itv["x"][intervention_time:, :]
test_data_after_itv["y"] = test_data_after_itv["y"][intervention_time:, :]

mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(test_data_before_itv)
mu_y_preds = np.append(mu_y_preds, mu_obs_preds)
std_y_preds = np.append(std_y_preds, std_obs_preds)

LL_index = hsl_tsad_agent.base_model.states_name.index("level")
LT_index = hsl_tsad_agent.base_model.states_name.index("trend")
LLd_index = hsl_tsad_agent.drift_model.states_name.index("level")
LTd_index = hsl_tsad_agent.drift_model.states_name.index("trend")

# # Correction with drift model
# hsl_tsad_agent.base_model.mu_states[LL_index] += hsl_tsad_agent.drift_model.mu_states[LLd_index]
# hsl_tsad_agent.base_model.mu_states[LT_index] += hsl_tsad_agent.drift_model.mu_states[LTd_index]
# hsl_tsad_agent.base_model.var_states[LL_index, LL_index] += hsl_tsad_agent.drift_model.var_states[LLd_index, LLd_index]
# hsl_tsad_agent.base_model.var_states[LT_index, LT_index] += hsl_tsad_agent.drift_model.var_states[LTd_index, LTd_index]

# # Correction with true anomaly baseline
# hsl_tsad_agent.base_model.mu_states[LL_index] += true_correction_level
# hsl_tsad_agent.base_model.mu_states[LT_index] += true_correction_trend

# Dummy correction on level only
hsl_tsad_agent.base_model.mu_states[LL_index] += dummy_level_correction

hsl_tsad_agent.drift_model.mu_states[LLd_index] = 0
hsl_tsad_agent.drift_model.mu_states[LTd_index] = hsl_tsad_agent.mu_LTd

mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(test_data_after_itv)
mu_y_preds = np.append(mu_y_preds, mu_obs_preds)
std_y_preds = np.append(std_y_preds, std_obs_preds)

# Deep copy hsl_tsad_agent.drift_model.states
drift_model_states_dummy = copy.deepcopy(hsl_tsad_agent.drift_model.states)
base_model_states_dummy = copy.deepcopy(hsl_tsad_agent.base_model.states)
print(drift_model_states_dummy.mu_prior)
# for i in range(intervention_time + 10, len(drift_model_states_dummy.mu_prior)):
# for i in range(-(len(drift_model_states_dummy.mu_prior)-intervention_time_global)+5, -1):
for i in range(-(len(drift_model_states_dummy.mu_prior)-confidence_enough_step)+5, -1):
    print(drift_model_states_dummy.mu_prior[i])
    drift_model_states_dummy.mu_prior[i] = np.full_like(drift_model_states_dummy.mu_prior[i], np.nan)
    base_model_states_dummy.mu_prior[i] = np.full_like(base_model_states_dummy.mu_prior[i], np.nan)
    mu_y_preds[i] = np.nan
    print(drift_model_states_dummy.mu_prior[i])
    print('-----------------------')


# # Freeze the mu_prior and var_prior of the drift model states to the values at the intervention time
# drift_model_states_dummy.mu_prior[intervention_time:] = drift_model_states_dummy.mu_prior[intervention_time]
# drift_model_states_dummy.var_prior[intervention_time:] = drift_model_states_dummy.var_prior[intervention_time]


# ##########################################################################################

# mu_y_preds, std_y_preds,_ = pretrained_model.filter(normalized_data,train_lstm=False)
# pretrained_model.smoother()

#  Plot
state_type = "prior"
# fig = plt.figure(figsize=(6, 2.5), constrained_layout=True)
fig = plt.figure(figsize=(3, 2.5), constrained_layout=True)
gs = gridspec.GridSpec(4, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])

plot_data(
    data_processor=data_processor,
    standardization=True,
    plot_column=output_col,
    sub_plot=ax0,
    color='k',
    test_label = 'Obs.'
)
time = data_processor.get_time(split="all")
ax0.plot(time, mu_y_preds, color='tab:grey', label='Predicted obs.')
ax0.fill_between(time, 
                 mu_y_preds - std_y_preds, 
                 mu_y_preds + std_y_preds, 
                 color='tab:grey', alpha=0.2)
plot_states(
    data_processor=data_processor,
    standardization=True,
    # states=hsl_tsad_agent.base_model.states,
    states=base_model_states_dummy,
    states_type=state_type,
    states_to_plot=['level'],
    sub_plot=ax0,
    color='tab:orange',
)
ax0.set_ylabel('$x^{\mathtt{LL}}$')
# ax0.axvline(x=time[time_anomaly], color='tab:red', linestyle='--', label='Anomaly')
ax0.set_xticklabels([])

# ax0.legend(
#     bbox_to_anchor=(0.5, 1.02),
#     loc='lower center',
#     bbox_transform=ax0.transAxes,
#     ncol=4,
#     borderaxespad=0.,
#     frameon=True
# )

plot_states(
    data_processor=data_processor,
    standardization=True,
    # states=hsl_tsad_agent.base_model.states,
    states=base_model_states_dummy,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax1,
    color='tab:orange',
)
ax1.set_ylabel('$x^{\mathtt{LT}}$')
# ax1.yaxis.offsetText.set_fontsize(6)
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    # states=hsl_tsad_agent.base_model.states,
    states=base_model_states_dummy,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax2,
    color='tab:orange',
)
ax2.set_ylabel('$x^{\mathtt{AR}}$')
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    # states=hsl_tsad_agent.drift_model.states,
    states=drift_model_states_dummy,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax3,
    color='tab:orange',
)
ax3.set_ylabel('$x^{\mathtt{LTd}}$')

ax0.axvline(x=time[time_anomaly], color='tab:red', linestyle='--', label='Anomaly')
ax1.axvline(x=time[time_anomaly], color='tab:red', linestyle='--', label='Anomaly')
ax2.axvline(x=time[time_anomaly], color='tab:red', linestyle='--', label='Anomaly')
ax3.axvline(x=time[time_anomaly], color='tab:red', linestyle='--', label='Anomaly')

# Only plot from the beginning of the test set
ax0.set_xlim(time[data_processor.validation_end], time[-1])
ax1.set_xlim(time[data_processor.validation_end], time[-1])
ax2.set_xlim(time[data_processor.validation_end], time[-1])
ax3.set_xlim(time[data_processor.validation_end], time[-1])

tick_positions = [pd.Timestamp(f'{y}-01-01') for y in [2018, 2020, 2022, 2024]]
for ax in [ax0, ax1, ax2, ax3]:
    ax.set_xticks(tick_positions)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax0.set_xticklabels([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax1.set_ylim(-0.00041018286422543604, 0.0012889237594637154)
ax3.set_ylim(-0.00041018286422543604, 0.0012889237594637154)
# ax1.set_ylim(ax3.get_ylim())

# # Set ax3 x ticks to every 3 years and only show year number
# ax3.xaxis.set_major_locator(ticker.MultipleLocator(52*12))
# ax2.xaxis.set_major_locator(ticker.MultipleLocator(52*12))
# ax1.xaxis.set_major_locator(ticker.MultipleLocator(52*12))
# ax0.xaxis.set_major_locator(ticker.MultipleLocator(52*12))
# ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/(52*12))}'))

# # Plot stationary AR
# phi_ar = model_dict['states_optimal'].mu_prior[-1][phi_index].item()
# sigma_ar = np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item())
# std_ar_stationary = sigma_ar / np.sqrt(1 - phi_ar**2)
# ax3.fill_between(time, 0 - std_ar_stationary, 0 + std_ar_stationary, color='tab:orange', alpha=0.2, label='Stationary AR std')

fig.align_ylabels([ax0, ax1, ax2])
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.4)
plt.savefig('rsic_step_by_step_1.png', dpi=300)
plt.show()