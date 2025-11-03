import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from canari.component import LocalTrend, LstmNetwork, Periodic, Autoregression
from canari import (
    DataProcess,
    Model,
    common,
    plot_data,
    plot_prediction,
    plot_states,
)
from src.hsl_classification_mp2_2classes_2driftmodels import hsl_classification
from src.matrix_profile_functions import past_only_matrix_profile
import pytagi.metric as metric
import pickle
import ast
from tqdm import tqdm
from matplotlib import gridspec
from canari.data_visualization import _add_dynamic_grids


# # # Read data
data_file = "./data/toy_time_series/syn_data_anmtype_simple_phi05.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

# # LT anomaly
# time_anomaly = 52*7
# anm_mag = 12/52
# anm_baseline = np.arange(len(df_raw)) * anm_mag
# # Set the first 52*12 values in anm_baseline to be 0
# anm_baseline[time_anomaly:] -= anm_baseline[time_anomaly]
# anm_baseline[:time_anomaly] = 0
# df_raw = df_raw.add(anm_baseline, axis=0)

# # LL anomaly
# time_anomaly = 52*7
# anm_mag = 35
# anm_baseline = np.ones(len(df_raw)) * anm_mag
# anm_baseline[:time_anomaly] = 0
# df_raw = df_raw.add(anm_baseline, axis=0)

# # PD anomaly
# time_anomaly = 52*7
# anm_mag = 35
# sine_curve = anm_mag * np.sin(np.arange(len(df_raw)) * 2 * np.pi / 52)
# sine_curve[:time_anomaly] = 0
# df_raw = df_raw.add(sine_curve, axis=0)

# # # Outlier
# time_anomaly = 52*7
# df_raw.iloc[time_anomaly] += 50

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
with open("saved_params/paper_example.pkl", "rb") as f:
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
    LocalTrend(mu_states=[0, 0], var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                   mu_states=[model_dict["mu_states"][autoregression_index].item()], 
                   var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
)
print("phi_AR_gen =", model_dict['gen_phi_ar'])
print("sigma_AR_gen =", np.sqrt(model_dict['gen_sigma_ar']))
gen_model = Model(
    # LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=np.diag(model_dict['states_optimal'].var_prior[0][0:2, 0:2])),\
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
hsl_tsad_agent.mu_LTd = 8.881274575122074e-06
hsl_tsad_agent.LTd_std = 0.00010250327431417399
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std * 1)
# hsl_tsad_agent.tune_panm_threshold(data=train_val_data)
hsl_tsad_agent.detection_threshold = 0.1

# hsl_tsad_agent.collect_anmtype_samples(num_time_series=1000, save_to_path='data/anm_type_class_train_samples/classifier_learn_samples_syn_simple_ts_two_classes_dmodels.csv')
# hsl_tsad_agent.nn_train_with = 'tagiv'
# hsl_tsad_agent.mean_train, hsl_tsad_agent.std_train, hsl_tsad_agent.mean_target, hsl_tsad_agent.std_target = -3.7583715e-05, 0.0004518164, np.array([-4.0172847e-04, -4.7810923e-02, 1.0713673e+02]), np.array([1.1112380e-02, 1.3762859e+00, 6.2584328e+01])
# hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class, hsl_tsad_agent.mean_MP_class, hsl_tsad_agent.std_MP_class = -3.0772888e-05, 0.0004556137, 3.1387298, 1.321072
# hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class, hsl_tsad_agent.mean_MP_class, hsl_tsad_agent.std_MP_class = -2.4802439e-05, 0.000404261, 2.988104, 1.2404884    # V2 training
# MP2 models:
hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class,hsl_tsad_agent.mean_LTd2_class, hsl_tsad_agent.std_LTd2_class, hsl_tsad_agent.mean_MP_class, hsl_tsad_agent.std_MP_class = -2.1853131e-05, 0.00045217096, -5.1776482e-05, 0.003471423, 4.198408, 1.8218131
hsl_tsad_agent.learn_classification(training_samples_path='data/anm_type_class_train_samples/classifier_learn_samples_syn_simple_ts_two_classes_dmodels.csv', 
                                    load_model_path='saved_params/NN_classification_model_syn_simple_ts_mp2_2classes_2dmodels.pkl', max_training_epoch=50)
# hsl_tsad_agent.learn_intervention(training_samples_path='data/hsl_tsad_training_samples/itv_learn_samples_syn_simple_ts.csv', 
#                                   load_model_path='saved_params/NN_detection_model_syn_simple_ts.pkl', max_training_epoch=50)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data, apply_intervention=False)
mu_ar_preds_all = np.hstack((mu_ar_preds_all, mu_ar_preds.flatten()))
std_ar_preds_all = np.hstack((std_ar_preds_all, std_ar_preds.flatten()))

# #  Plot
state_type = "posterior"
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
# ax0.axvline(x=time[anm_start_index], color='red', linestyle='--', label='Anomaly start')
ax0.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax1,
)
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
ax4.plot(time, np.array(mu_ar_preds_all), label='obs', color='tab:red')
ax4.fill_between(time,
                np.array(mu_ar_preds_all) - np.array(std_ar_preds_all),
                np.array(mu_ar_preds_all) + np.array(std_ar_preds_all),
                color='tab:red',
                alpha=0.5)
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.drift_model.states,
    states_type=state_type,
    states_to_plot=['level'],
    sub_plot=ax4,
)
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.drift_model2.states,
    states_type=state_type,
    states_to_plot=['level'],
    sub_plot=ax4,
    color='tab:green',
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
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.drift_model2.states,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax5,
    color='tab:green',
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
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.drift_model2.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax6,
    color='tab:green',
)
ax6.set_xticklabels([])

ax7.plot(time, hsl_tsad_agent.mp_all, label="MP metric", color="blue")
# ax6.axvline(x=time[time_anomaly], color='tab:red', linestyle='--', label='Anomaly')
ax7.set_ylabel('MP')
ax7.set_xticklabels([])
_add_dynamic_grids(ax7, time)

ax8.plot(time, hsl_tsad_agent.p_anm_all)
detection_time = np.where(np.array(hsl_tsad_agent.p_anm_all) > hsl_tsad_agent.detection_threshold)[0]
for i in range(len(detection_time)):
    ax8.axvline(x=time[detection_time[i]], color='red', linestyle='--', label='Anomaly start')
ax8.set_ylabel("p_anm")
ax8.set_xlim(ax0.get_xlim())
ax8.set_ylim(-0.05, 1.05)
_add_dynamic_grids(ax8, time)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

m_logits = np.array(hsl_tsad_agent.pred_class_probs)
std_logits = np.sqrt(np.array(hsl_tsad_agent.pred_class_probs_var))
## ReMax(logits)
from src.convert_to_class import hierachical_softmax
m_probs, std_probs = [], []
for t in range(m_logits.shape[0]):
    pr_classes = hierachical_softmax(m_logits[t], std_logits[t])
    m_probs.append(pr_classes.tolist())
    # std_probs.append(std_p.tolist())
m_probs = np.array(m_probs)
# std_probs = np.array(std_probs)

# anm_prob_lower = np.array(hsl_tsad_agent.pred_class_probs) - np.sqrt(np.array(hsl_tsad_agent.pred_class_probs_var))
for class_idx in range(m_logits.shape[1]):
    ax9.plot(time, m_logits[:, class_idx], color=colors[class_idx])
    ax9.fill_between(time,
                    m_logits[:, class_idx] - std_logits[:, class_idx],
                    m_logits[:, class_idx] + std_logits[:, class_idx],
                    color=colors[class_idx],
                    alpha=0.3, label="_nolegend_")
# Set legend labels to ['LT', 'LL', 'PD']
ax9.legend(['gate1', 'gate2'],loc='upper left', ncol=2)
# ax9.legend = ['LT', 'LL', 'PD']
ax9.set_ylim(-5, 5)
# ax9.legend(loc='upper left', ncol=2)
ax9.set_ylabel("logits")

# Plot class probabilities with ReMax
for class_idx in range(m_probs.shape[1]):
    ax10.plot(time, m_probs[:, class_idx], color=colors[class_idx])
    # ax10.fill_between(time,
    #                 m_probs[:, class_idx] - std_probs[:, class_idx],
    #                 m_probs[:, class_idx] + std_probs[:, class_idx],
    #                 color=colors[class_idx],
    #                 alpha=0.3, label="_nolegend_")
# Set legend labels to ['LT', 'LL', 'PD']
ax10.legend(['LT', 'LL'],loc='upper left', ncol=2)
ax10.set_ylim(-0.05, 1.05)
ax10.set_ylabel("probs")

# Find the anm_flags that is 0.5 greater than the other two class flags
# anm_type_flags = []
# for t in range(anm_prob_lower.shape[0]):
#     max_flag = np.max(anm_prob_lower[t, :])
#     max_idx = np.argmax(anm_prob_lower[t, :])
#     other_anm = np.delete(anm_prob_lower[t, :], max_idx)
#     if max_flag - np.max(other_anm) < 0.5:
#         pass
#     else:
#         anm_type_flags.append((t, max_idx, anm_prob_lower[t, max_idx]))
# # Plot vertical lines for the anm_type_flags
# for flag in anm_type_flags:
#     ax9.axvline(x=time[flag[0]], color=colors[flag[1]], linestyle=':')

plt.show()