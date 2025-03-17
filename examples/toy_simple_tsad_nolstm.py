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
from src.hsl_detection import hsl_detection
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
import pickle
from src.model import load_model_dict


# # Read data
data_file = "./data/toy_time_series/synthetic_simple_autoregression_periodic_2.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# linear_space = np.linspace(0, 4, num=len(df_raw))
linear_space = np.arange(len(df_raw)) * 0.010416667/10
# Set the first 52*12 values in linear_space to be 0
anm_start_index = 52*10
linear_space[anm_start_index:] -= linear_space[anm_start_index]
linear_space[:anm_start_index] = 0

df_raw = df_raw.add(linear_space, axis=0)

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
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=train_split,
    validation_split=validation_split,
    output_col=output_col,
    normalization = False,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()


####################################################################
######################### True model #########################
####################################################################
pretrained_model = Model(
    LocalTrend(mu_states=[0, 0], var_states=[1e-12, 1e-12], std_error=0),
    Periodic(period=52, mu_states=[0.5, 1], var_states=[1e-12, 1e-12]),
    Autoregression(std_error=0.05, phi=0.8, mu_states=[0], var_states=[0.08]),
)

ltd_error = 1e-5

hsl_tsad_agent = hsl_detection(base_model=pretrained_model, data_processor=data_processor, drift_model_process_error_std=ltd_error)

# Get flexible drift model from the beginning
hsl_tsad_agent_pre = hsl_detection(base_model=load_model_dict(pretrained_model.save_model_dict()), data_processor=data_processor, drift_model_process_error_std=ltd_error)
hsl_tsad_agent_pre.filter(train_data)
hsl_tsad_agent_pre.filter(validation_data)
hsl_tsad_agent.drift_model.var_states = hsl_tsad_agent_pre.drift_model.var_states

mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(train_data, buffer_LTd=True)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(validation_data, buffer_LTd=True)
hsl_tsad_agent.estimate_LTd_dist()
# print('start collecting NN training samples')
# hsl_tsad_agent.collect_synthetic_samples(num_time_series=1000, save_to_path= 'data/hsl_tsad_training_samples/itv_learn_samples_different_anm_mag_simple_complet_1000_.csv')
hsl_tsad_agent.nn_train_with = 'tagiv'
hsl_tsad_agent.learn_intervention(training_samples_path='data/hsl_tsad_training_samples/itv_learn_samples_different_anm_mag_simple_complet_1000.csv', 
                                  save_model_path='saved_params/NN_detection_model_simpleTS_fourrier_1000.pkl', max_training_epoch=50)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data, apply_intervention=False)

anm_detected_index = np.where(np.array(hsl_tsad_agent.p_anm_all) > 0.5)[0][0]

# Plot to debug
# Delete in hsl_tsad_agent.LTd_history_all all the samples before and after anm_start_index and anm_detected_index
hsl_tsad_agent.LTd_history_all = np.array(hsl_tsad_agent.LTd_history_all[anm_start_index:anm_detected_index])
print(hsl_tsad_agent.LTd_history_all.shape)
grayscale_anm_dev_time = (hsl_tsad_agent.train_y[:, 2] - hsl_tsad_agent.train_y[:, 2].min()) / (hsl_tsad_agent.train_y[:, 2].max() - hsl_tsad_agent.train_y[:, 2].min())
# Plot all samples input
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
# ax.plot(samples_input.T, color='black', alpha=0.1)
# Plot samples_input with color based on grayscale_anm_dev_time
for i in range(1000):
    ax.plot(hsl_tsad_agent.train_X[i], color=plt.cm.viridis_r(grayscale_anm_dev_time[i]), alpha=0.5)
for i in range(len(hsl_tsad_agent.LTd_history_all)):
    ax.plot(hsl_tsad_agent.LTd_history_all[i], color='r', alpha=0.5)

ax.set_xlabel('Time')
ax.set_ylabel('LTd')
# Plot the color map
fig.colorbar(plt.cm.ScalarMappable(cmap='viridis_r'), ax=ax, orientation='horizontal', label='anm_develop_time')

#  Plot
state_type = "prior"
#  Plot states from pretrained model
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(11, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
ax4 = plt.subplot(gs[4])
ax5 = plt.subplot(gs[5])
ax6 = plt.subplot(gs[6])
ax7 = plt.subplot(gs[7])
ax8 = plt.subplot(gs[8])
ax9 = plt.subplot(gs[9])
ax10 = plt.subplot(gs[10])
from src.data_visualization import determine_time
time = determine_time(data_processor, len(normalized_data["y"]))
plot_data(
    data_processor=data_processor,
    normalization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax0,
)
plot_states(
    data_processor=data_processor,
    # states=pretrained_model.states,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['local level'],
    sub_plot=ax0,
)
ax0.axvline(x=time[anm_start_index], color='r', linestyle='--')
ax0.set_xticklabels([])
ax0.set_title("Hidden states likelihood")
plot_states(
    data_processor=data_processor,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['local trend'],
    sub_plot=ax1,
)
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['periodic 1'],
    sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax3,
)
ax3.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=hsl_tsad_agent.drift_model.states,
    states_type=state_type,
    states_to_plot=['local level'],
    sub_plot=ax4,
)
ax4.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=hsl_tsad_agent.drift_model.states,
    states_type=state_type,
    states_to_plot=['local trend'],
    sub_plot=ax5,
)
ax5.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=hsl_tsad_agent.drift_model.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax6,
)
ax6.set_xticklabels([])
ax7.plot(time, hsl_tsad_agent.p_anm_all)
ax7.set_ylabel("p_anm")
ax7.set_xlim(ax0.get_xlim())
ax7.axvline(x=time[anm_start_index], color='r', linestyle='--')
ax7.set_ylim(-0.05, 1.05)

mu_itv_all = np.array(hsl_tsad_agent.mu_itv_all)
std_itv_all = np.array(hsl_tsad_agent.std_itv_all)
# Set all the values before anm_start_index to be nan
mu_itv_all[:anm_start_index] = np.nan
std_itv_all[:anm_start_index] = np.nan
print('anm_detect_index:' , anm_detected_index)
# Set all the values after anm_detected to be nan
mu_itv_all[anm_detected_index:] = np.nan
std_itv_all[anm_detected_index:] = np.nan
# print(mu_itv_all.shape)
# print(time.shape)

true_anm_dev_time = np.zeros_like(mu_itv_all[:, 1])
true_anm_dev_time[anm_start_index:anm_detected_index] += np.arange(anm_detected_index - anm_start_index)
true_LL = true_anm_dev_time * 0.00104166666666666

ax8.plot(time, mu_itv_all[:, 0])
ax8.fill_between(time, mu_itv_all[:, 0] - std_itv_all[:, 0], mu_itv_all[:, 0] + std_itv_all[:, 0], alpha=0.5)
ax8.set_ylabel("itv_LT")
ax8.set_xlim(ax0.get_xlim())
ax8.axvline(x=time[anm_start_index], color='r', linestyle='--')
ax8.axhline(y=0.00104166666666666, color='r', linestyle='--')

ax9.plot(time, mu_itv_all[:, 1])
ax9.fill_between(time, mu_itv_all[:, 1] - std_itv_all[:, 1], mu_itv_all[:, 1] + std_itv_all[:, 1], alpha=0.5)
ax9.plot(time, true_LL, color='k', linestyle='--')
ax9.set_ylabel("itv_LL")
ax9.set_xlim(ax0.get_xlim())
ax9.axvline(x=time[anm_start_index], color='r', linestyle='--')

ax10.plot(time, mu_itv_all[:, 2])
ax10.fill_between(time, mu_itv_all[:, 2] - std_itv_all[:, 2], mu_itv_all[:, 2] + std_itv_all[:, 2], alpha=0.5)
ax10.plot(time, true_anm_dev_time, color='k', linestyle='--')
ax10.set_ylabel("itv_time")
ax10.set_xlim(ax0.get_xlim())
ax10.axvline(x=time[anm_start_index], color='r', linestyle='--')
plt.show()