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
from src.hsl_classification_2classes_rsic_v2_w5 import hsl_classification
import pytagi.metric as metric
from matplotlib import gridspec
import pickle
from pytagi import Normalizer
import copy
from canari.data_visualization import _add_dynamic_grids


# # # Read data
data_file = "./data/benchmark_data/detrended_data/test_6_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

# # LT anomaly
# anm_type = 'LT'
# time_anomaly = 52*7
# anm_mag = 0.1/52
# anm_baseline = np.arange(len(df_raw)) * anm_mag
# # Set the first 52*12 values in anm_baseline to be 0
# anm_baseline[time_anomaly:] -= anm_baseline[time_anomaly]
# anm_baseline[:time_anomaly] = 0
# df_raw = df_raw.add(anm_baseline, axis=0)

# # LL anomaly
# anm_type = 'LL'
# time_anomaly = 52*7
# anm_mag = 17
# anm_baseline = np.ones(len(df_raw)) * anm_mag
# anm_baseline[:time_anomaly] = 0
# df_raw = df_raw.add(anm_baseline, axis=0)

# # Second anomaly
# anm2_type = 'LL'
# time_anomaly2 = 52*10
# anm2_mag = 0.4
# anm2_baseline = np.ones(len(df_raw)) * anm2_mag
# anm2_baseline[:time_anomaly2] = 0
# df_raw = df_raw.add(anm2_baseline, axis=0)

# # Second anomaly
# anm2_type = 'LT'
# time_anomaly2 = 52*10
# anm2_mag = 0.1/52
# anm2_baseline = np.arange(len(df_raw)) * anm2_mag
# anm2_baseline[time_anomaly2:] -= anm2_baseline[time_anomaly2]
# anm2_baseline[:time_anomaly2] = 0
# df_raw = df_raw.add(anm2_baseline, axis=0)

# Data pre-processing
output_col = [0]
train_split=0.3
validation_split=0.1
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=train_split,
    validation_split=validation_split,
    output_col=output_col,
)


train_data, validation_data, test_data, normalized_data = data_processor.get_splits()
train_val_data = copy.deepcopy(normalized_data)
train_val_data["x"] = train_val_data["x"][0:data_processor.validation_end, :]
train_val_data["y"] = train_val_data["y"][0:data_processor.validation_end, :]

####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
with open("saved_params/real_ts6_tsmodel_detrended.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=17,
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

mu_ar_preds_all, std_ar_preds_all = [], []
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(train_data, buffer_LTd=True)
mu_ar_preds_all = np.hstack((mu_ar_preds_all, mu_ar_preds.flatten()))
std_ar_preds_all = np.hstack((std_ar_preds_all, std_ar_preds.flatten()))
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(validation_data, buffer_LTd=True)
mu_ar_preds_all = np.hstack((mu_ar_preds_all, mu_ar_preds.flatten()))
std_ar_preds_all = np.hstack((std_ar_preds_all, std_ar_preds.flatten()))
# hsl_tsad_agent.estimate_LTd_dist()
hsl_tsad_agent.mu_LTd = 1.122958405704475e-05
hsl_tsad_agent.LTd_std = 9.393607875116122e-05
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std)
# hsl_tsad_agent.tune_panm_threshold(data=normalized_data)
hsl_tsad_agent.detection_threshold = 0.1

# hsl_tsad_agent.collect_anmtype_samples(num_time_series=1000, save_to_path='data/anm_type_class_train_samples/classifier_learn_samples_detrended_ts6.csv')
hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class = 3.154062e-05, 0.001513519
hsl_tsad_agent.mean_target_lt_model, hsl_tsad_agent.std_target_lt_model = np.array([-6.7769302e-05, -1.0898615e-02]), np.array([0.00661581, 0.8756936])
hsl_tsad_agent.mean_target_ll_model, hsl_tsad_agent.std_target_ll_model = np.array([0.00809346]), np.array([0.6842518])

hsl_tsad_agent.learn_intervention(training_samples_path='data/anm_type_class_train_samples/classifier_learn_samples_detrended_ts6.csv', 
                                    load_lt_model_path='saved_params/NN_intervention_LT_model_rsic_detrend_ts6.pkl', 
                                    load_ll_model_path='saved_params/NN_intervention_LL_model_rsic_detrend_ts6.pkl', 
                                    max_training_epoch=50)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data)
mu_ar_preds_all = np.hstack((mu_ar_preds_all, mu_ar_preds.flatten()))
std_ar_preds_all = np.hstack((std_ar_preds_all, std_ar_preds.flatten()))

# #  Plot
state_type = "posterior"
#  Plot states from pretrained model
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(7, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
ax4 = plt.subplot(gs[4])
ax5 = plt.subplot(gs[5])
# ax6 = plt.subplot(gs[6])
ax7 = plt.subplot(gs[6])
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
ax0.plot(time, np.array(hsl_tsad_agent.posterior_mu_states_no_itv)[:, 0, 0], color='grey', linestyle='--')
ax0.fill_between(time, ax0.get_ylim()[0], ax0.get_ylim()[1], where=~np.isnan(np.array(hsl_tsad_agent.posterior_mu_states_no_itv)[:, 0, 0]), color='lightgrey', alpha=0.5)
# ax0.axvline(x=time[anm_start_index], color='red', linestyle='--', label='Anomaly start')
ax0.set_xticklabels([])
# ax0.axhline(y=normed_anm_mag, color='purple', linestyle='--', label='Anomaly magnitude')
# ax0.plot(time, normed_anm_baseline, color='purple', linestyle='--', label='Anomaly magnitude')

############ Original plots #############
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax1,
)
ax1.plot(time, np.array(hsl_tsad_agent.posterior_mu_states_no_itv)[:, 1, 0], color='grey', linestyle='--')
ax1.fill_between(time, ax1.get_ylim()[0], ax1.get_ylim()[1], where=~np.isnan(np.array(hsl_tsad_agent.posterior_mu_states_no_itv)[:, 1, 0]), color='lightgrey', alpha=0.5)
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

ax4.plot(time, hsl_tsad_agent.p_anm_all)
detection_time = np.where(np.array(hsl_tsad_agent.p_anm_all) > hsl_tsad_agent.detection_threshold)[0]
for i in range(len(detection_time)):
    ax4.axvline(x=time[detection_time[i]], color='red', linestyle='--', label='Anomaly start')
ax4.set_ylabel("p_anm")
ax4.set_xlim(ax0.get_xlim())
ax4.set_ylim(-0.05, 1.05)
_add_dynamic_grids(ax4, time)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

final_class_log_probs = np.array(hsl_tsad_agent.class_prob_moments)[:, 0:2]
final_class_prob_stds = np.array(hsl_tsad_agent.class_prob_moments)[:, 2]

gen_ar_phi = model_dict['gen_phi_ar']
gen_ar_sigma =model_dict['gen_sigma_ar']
stationary_ar_std = np.sqrt(gen_ar_sigma**2 / (1 - gen_ar_phi**2))
ax5.fill_between(time, - 2 * stationary_ar_std, 2 * stationary_ar_std, color='gray', alpha=0.3, label='2-Sigma range')
ax5.plot(time, hsl_tsad_agent.ll_itv_all, label='LL itv', color='tab:blue')
ax5.plot(time, hsl_tsad_agent.lt_itv_all, label='LT itv', color='tab:orange')
ax5.set_ylabel("itv")

for class_idx in range(final_class_log_probs.shape[1]):
    ax7.plot(time, final_class_log_probs[:, class_idx], color=colors[class_idx])
    ax7.fill_between(time, final_class_log_probs[:, class_idx] - final_class_prob_stds, 
                     final_class_log_probs[:, class_idx] + final_class_prob_stds, color=colors[class_idx], alpha=0.3)
ax7.set_ylim(-0.05, 1.05)
ax7.set_ylabel("Pr(anm)")

plt.show()