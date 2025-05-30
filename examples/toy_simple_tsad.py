import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari.component import LocalTrend, LstmNetwork, Autoregression
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_states,
    common,
)
from src.hsl_detection import hsl_detection
import pytagi.metric as metric
from matplotlib import gridspec
import pickle


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

# Load model_dict from local
with open("saved_params/toy_simple_model_rebased.pkl", "rb") as f:
    model_dict = pickle.load(f)

# Get true baseline
norm_const_std = data_processor.std_const_std[data_processor.output_col]
anm_mag_normed = anm_mag / norm_const_std
LL_baseline_true = np.zeros_like(df_raw)
LT_baseline_true = np.zeros_like(df_raw)
for i in range(1, len(df_raw)):
    if i > anm_start_index:
        LL_baseline_true[i] += anm_mag_normed * (i - anm_start_index)
        LT_baseline_true[i] += anm_mag_normed

LL_baseline_true += model_dict['early_stop_init_mu_states'][0].item()
LL_baseline_true = LL_baseline_true.flatten()
LT_baseline_true = LT_baseline_true.flatten()

####################################################################
######################### Pretrained model #########################
####################################################################
LSTM = LstmNetwork(
        look_back_len=19,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

print("phi_AR =", model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item())
print("sigma_AR =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()))


pretrained_model = Model(
    LocalTrend(mu_states=model_dict['early_stop_init_mu_states'][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(), 
                   mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()], 
                   var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
)

pretrained_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

ltd_error = 1e-5

hsl_tsad_agent = hsl_detection(base_model=pretrained_model, data_processor=data_processor, drift_model_process_error_std=ltd_error)

# Get flexible drift model from the beginning
hsl_tsad_agent_pre = hsl_detection(base_model=pretrained_model.load_dict(pretrained_model.get_dict()), data_processor=data_processor, drift_model_process_error_std=ltd_error)
hsl_tsad_agent_pre.filter(train_data)
hsl_tsad_agent_pre.filter(validation_data)
hsl_tsad_agent.drift_model.var_states = hsl_tsad_agent_pre.drift_model.var_states


mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(train_data, buffer_LTd=True)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(validation_data, buffer_LTd=True)
# hsl_tsad_agent.estimate_LTd_dist()
hsl_tsad_agent.mu_LTd = -6.51818370462253e-06
hsl_tsad_agent.LTd_std = 3.676127606711578e-05
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std)

# hsl_tsad_agent.collect_synthetic_samples(num_time_series=1000, save_to_path= 'data/hsl_tsad_training_samples/itv_learn_samples_toy_simple_rebased.csv')
hsl_tsad_agent.nn_train_with = 'tagiv'
hsl_tsad_agent.mean_train, hsl_tsad_agent.std_train, hsl_tsad_agent.mean_target, hsl_tsad_agent.std_target = 4.6750658e-05, 0.0007661439, np.array([4.8646217e-04,5.3189050e-02,1.0734344e+02]), np.array([1.1217532e-02, 1.3954039e+00, 6.2539051e+01])
hsl_tsad_agent.learn_intervention(training_samples_path='data/hsl_tsad_training_samples/itv_learn_samples_toy_simple_rebased.csv', 
                                  load_model_path='saved_params/NN_detection_model_toy_simple_rebased.pkl', max_training_epoch=50)
                                #   load_model_path='saved_params/NN_detection_model_toy_simple_V2.pkl', max_training_epoch=50)
# hsl_tsad_agent.tune(decay_factor=0.95)
# hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std * 0.6634204312890623)
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data, apply_intervention=True)

if (np.array(hsl_tsad_agent.p_anm_all) > 0.5).any():
    anm_detected_index = np.where(np.array(hsl_tsad_agent.p_anm_all) > 0.5)[0][0]
else:
    anm_detected_index = len(hsl_tsad_agent.p_anm_all)

# Compute MSE for SKF baselines
mu_LL_states = hsl_tsad_agent.base_model.states.get_mean(states_type='prior', states_name=["level"])["level"]
mu_LT_states = hsl_tsad_agent.base_model.states.get_mean(states_type='prior', states_name=["trend"])["trend"]
mse_LL = metric.mse(
    mu_LL_states[anm_start_index:],
    LL_baseline_true[anm_start_index:],
)
mse_LT = metric.mse(
    mu_LT_states[anm_start_index:],
    LT_baseline_true[anm_start_index:],
)
mse = mse_LL + mse_LT

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
time = data_processor.get_time(split="all")
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
    # states=pretrained_model.states,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['level'],
    sub_plot=ax0,
)
ax0.plot(time, LL_baseline_true, color='k', linestyle='--')
ax0.axvline(x=time[anm_start_index], color='r', linestyle='--')
ax0.set_xticklabels([])
ax0.set_title(f"IL, mse_LL = {mse_LL:.3e}, mse_LT = {mse_LT:.3e}")
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax1,
)
# ax0.plot(time[anm_start_index:], mu_LL_states[anm_start_index:], color='r')
# ax1.plot(time[anm_start_index:], mu_LT_states[anm_start_index:], color='r')
ax1.plot(time, LT_baseline_true, color='k', linestyle='--')
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['lstm'],
    sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax3,
)
ax3.set_xticklabels([])

# ax4.plot(time, hsl_tsad_agent.p_anm_all, color='b')
# ax4.set_ylabel("p_anm")
# ax4.set_xlim(ax0.get_xlim())
# ax4.axvline(x=time[anm_start_index], color='r', linestyle='--')
# ax4.set_ylim(-0.05, 1.05)
# add_dynamic_grids(ax4, time)

plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.drift_model.states,
    states_type=state_type,
    states_to_plot=['level'],
    sub_plot=ax4,
)

ax4.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.drift_model.states,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax5,
)
ax5.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
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

true_anm_dev_time = np.zeros_like(mu_itv_all[:, 1])
true_anm_dev_time[anm_start_index:len(np.zeros_like(mu_itv_all[:, 1]))] += np.arange(len(np.zeros_like(mu_itv_all[:, 1])) - anm_start_index)
true_LL = true_anm_dev_time * anm_mag

ax8.plot(time, mu_itv_all[:, 0])
ax8.fill_between(time, mu_itv_all[:, 0] - std_itv_all[:, 0], mu_itv_all[:, 0] + std_itv_all[:, 0], alpha=0.5)
ax8.set_ylabel("itv_LT")
ax8.set_xlim(ax0.get_xlim())
ax8.axvline(x=time[anm_start_index], color='r', linestyle='--')
ax8.axhline(y=anm_mag, color='k', linestyle='--')

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