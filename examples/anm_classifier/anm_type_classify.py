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
from src.hsl_classification_2classes_2itvmodels import hsl_classification
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

# LT anomaly
anm_type = 'LT'
time_anomaly = 52*7
anm_mag = 6/52
anm_baseline = np.arange(len(df_raw)) * anm_mag
# Set the first 52*12 values in anm_baseline to be 0
anm_baseline[time_anomaly:] -= anm_baseline[time_anomaly]
anm_baseline[:time_anomaly] = 0
df_raw = df_raw.add(anm_baseline, axis=0)

# # LL anomaly
# anm_type = 'LL'
# time_anomaly = 52*7
# anm_mag = 17
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

# Normalize the trend anomaly
# normed_anm_mag =  (anm_mag - data_processor.scale_const_mean[0]) / data_processor.scale_const_std[0]
normed_anm_mag =  anm_mag / data_processor.scale_const_std[0]
print("normalied anomaly trend", normed_anm_mag)
normed_anm_baseline = np.arange(len(df_raw)) * normed_anm_mag
# Set the first 52*12 values in anm_baseline to be 0
normed_anm_baseline[time_anomaly:] -= normed_anm_baseline[time_anomaly]
normed_anm_baseline[:time_anomaly] = 0
# Get the normed lt baseline

# print("normalied anomaly trend", anm_mag / data_processor.scale_const_std[0])


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
print("sigma_AR_gen =", model_dict['gen_sigma_ar'])
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

# hsl_tsad_agent.collect_anmtype_samples(num_time_series=1000, save_to_path='data/anm_type_class_train_samples/classifier_learn_samples_syn_simple_ts_two_classes_dmodels_itv_newMP.csv')
# hsl_tsad_agent.nn_train_with = 'tagiv'
# hsl_tsad_agent.mean_train, hsl_tsad_agent.std_train, hsl_tsad_agent.mean_target, hsl_tsad_agent.std_target = -3.7583715e-05, 0.0004518164, np.array([-4.0172847e-04, -4.7810923e-02, 1.0713673e+02]), np.array([1.1112380e-02, 1.3762859e+00, 6.2584328e+01])
# hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class, hsl_tsad_agent.mean_MP_class, hsl_tsad_agent.std_MP_class = -3.0772888e-05, 0.0004556137, 3.1387298, 1.321072
# hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class, hsl_tsad_agent.mean_MP_class, hsl_tsad_agent.std_MP_class = -2.4802439e-05, 0.000404261, 2.988104, 1.2404884    # V2 training

# hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class,hsl_tsad_agent.mean_LTd2_class, hsl_tsad_agent.std_LTd2_class, hsl_tsad_agent.mean_MP_class, hsl_tsad_agent.std_MP_class = -2.1853131e-05, 0.00045217096, -5.1776482e-05, 0.003471423, 4.198408, 1.8218131

# # Classification + intervention models:
# hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class,hsl_tsad_agent.mean_LTd2_class, hsl_tsad_agent.std_LTd2_class, hsl_tsad_agent.mean_MP_class, hsl_tsad_agent.std_MP_class = -2.8887205e-05, 0.00045540265, -9.227837e-05, 0.0034822284, 4.2108073, 1.8548799
# hsl_tsad_agent.mean_target, hsl_tsad_agent.std_target = np.array([-2.3099042e-04, 1.1933503e-02, 5.7366203e+01]), np.array([5.7640807e-03, 5.9275675e-01, 7.4977921e+01])

# # Classification + intervention models + continuous history:
# hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class,hsl_tsad_agent.mean_LTd2_class, hsl_tsad_agent.std_LTd2_class, hsl_tsad_agent.mean_MP_class, hsl_tsad_agent.std_MP_class = 3.0531803e-06, 0.0003885695, -7.6621116e-05, 0.0034827138, 3.8428175, 2.1701705
# hsl_tsad_agent.mean_target, hsl_tsad_agent.std_target = np.array([2.2853521e-04, 2.1895172e-02, 5.9053360e+01]), np.array([5.3996309e-03, 6.2583315e-01, 7.5614799e+01])

# # # Classification + intervention models + new MP:
# hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class,hsl_tsad_agent.mean_LTd2_class, hsl_tsad_agent.std_LTd2_class, hsl_tsad_agent.mean_MP_class, hsl_tsad_agent.std_MP_class = -1.3441674e-05, 0.0004603353, -4.2705156e-05, 0.003506228, 5.272658, 3.560844
# hsl_tsad_agent.mean_target_lt_model, hsl_tsad_agent.std_target_lt_model = np.array([3.3949208e-04, 3.8289719e+01]), np.array([6.3677123e-03, 6.6973274e+01])
# hsl_tsad_agent.mean_target_ll_model, hsl_tsad_agent.std_target_ll_model = np.array([1.9944116e-02, 4.2935741e+01]), np.array([0.72300506, 69.76192])

# 2 intervention models:
hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class,hsl_tsad_agent.mean_LTd2_class, hsl_tsad_agent.std_LTd2_class, hsl_tsad_agent.mean_MP_class, hsl_tsad_agent.std_MP_class = -1.3276256e-05, 0.0004600634, -3.7790043e-05, 0.003501312, 5.2705355, 3.5586061
hsl_tsad_agent.mean_target_lt_model, hsl_tsad_agent.std_target_lt_model = np.array([2.518625e-04, 3.668254e+01]), np.array([6.1926572e-03, 6.5914955e+01])
hsl_tsad_agent.mean_target_ll_model, hsl_tsad_agent.std_target_ll_model = np.array([6.4476170e-03, 4.3150906e+01]), np.array([0.7179858, 69.91169])

hsl_tsad_agent.learn_classification(training_samples_path='data/anm_type_class_train_samples/classifier_learn_samples_syn_simple_ts_two_classes_dmodels_itv_newMP.csv', 
                                    load_model_path='saved_params/NN_classification_model_syn_simple_ts_datall_newMP.pkl', max_training_epoch=50)
hsl_tsad_agent.learn_intervention(training_samples_path='data/anm_type_class_train_samples/classifier_learn_samples_syn_simple_ts_two_classes_dmodels_itv_newMP.csv', 
                                    load_lt_model_path='saved_params/NN_intervention_LT_model_syn_simple_ts.pkl', load_ll_model_path='saved_params/NN_intervention_LL_model_syn_simple_ts.pkl', max_training_epoch=50)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data, apply_intervention=False)
mu_ar_preds_all = np.hstack((mu_ar_preds_all, mu_ar_preds.flatten()))
std_ar_preds_all = np.hstack((std_ar_preds_all, std_ar_preds.flatten()))

# #  Plot
state_type = "posterior"
#  Plot states from pretrained model
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(8, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
ax4 = plt.subplot(gs[4])
ax5 = plt.subplot(gs[5])
ax6 = plt.subplot(gs[6])
ax7 = plt.subplot(gs[7])
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
# ax0.axhline(y=normed_ll, color='purple', linestyle='--', label='Anomaly magnitude')
ax0.plot(time, normed_anm_baseline, color='purple', linestyle='--', label='Anomaly magnitude')

############ Original plots #############
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

ax4.plot(time, hsl_tsad_agent.p_anm_all)
detection_time = np.where(np.array(hsl_tsad_agent.p_anm_all) > hsl_tsad_agent.detection_threshold)[0]
for i in range(len(detection_time)):
    ax4.axvline(x=time[detection_time[i]], color='red', linestyle='--', label='Anomaly start')
ax4.set_ylabel("p_anm")
ax4.set_xlim(ax0.get_xlim())
ax4.set_ylim(-0.05, 1.05)
_add_dynamic_grids(ax4, time)

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

m_logits = np.array(hsl_tsad_agent.pred_class_probs)
std_logits = np.sqrt(np.array(hsl_tsad_agent.pred_class_probs_var))
## ReMax(logits)
from src.convert_to_class import hierachical_softmax
m_probs, std_probs = [], []
for t in range(m_logits.shape[0]):
    pr_classes = hierachical_softmax(m_logits[t], std_logits[t])
    m_probs.append(pr_classes.tolist())
m_probs = np.array(m_probs)

# # Combine the m_probs with self.data_loglikelihoods to get final class probabilities
final_class_log_probs = []
for t in range(len(hsl_tsad_agent.data_loglikelihoods)):
    if hsl_tsad_agent.data_loglikelihoods[t][0] is None:
        final_class_log_probs.append([0.5, 0.5])
    else:
        log_likelihoods = hsl_tsad_agent.data_loglikelihoods[t][0:2]
        log_likelihoods_op = hsl_tsad_agent.data_loglikelihoods[t][2:]

        probs = np.exp(log_likelihoods)
        # probs = np.array(log_likelihoods).astype(np.float64)
        probs /= np.sum(probs)
        final_class_log_probs.append(probs)        

final_class_log_probs = np.array(final_class_log_probs)
# Plot final class probabilities
for class_idx in range(final_class_log_probs.shape[1]):
    ax7.plot(time, final_class_log_probs[:, class_idx], color=colors[class_idx])
# Set legend labels to ['LT', 'LL', 'PD']
ax7.legend(['LT', 'LL'], loc='upper left', ncol=2)
ax7.set_ylim(-0.05, 1.05)
ax7.set_ylabel("posteriors")

gen_ar_phi = model_dict['gen_phi_ar']
gen_ar_sigma =model_dict['gen_sigma_ar']
stationary_ar_std = np.sqrt(gen_ar_sigma**2 / (1 - gen_ar_phi**2))
ax5.fill_between(time, - 2 * stationary_ar_std, 2 * stationary_ar_std, color='gray', alpha=0.3, label='2-Sigma range')
ax5.plot(time, hsl_tsad_agent.lt_itv_all, label='LT itv', color='tab:blue')
ax5.plot(time, hsl_tsad_agent.ll_itv_all, label='LL itv', color='tab:orange')
ax5.set_ylabel("itv")
# ax5.axhline(y=normed_ll, color='purple', linestyle='--', label='Anomaly magnitude')
ax5.plot(time, normed_anm_baseline, color='purple', linestyle='--', label='Anomaly magnitude')

# Look up the index where hsl_tsad_agent.ll_itv_all is equal to - 2 * stationary_ar_std or 2 * stationary_ar_std
masked_ll_itv = np.array(hsl_tsad_agent.ll_itv_all)
# Set the first 26 values greater than 0 to be 0
# Fine the values greater than 0
non_zero_start_indices = np.where(masked_ll_itv != 0)[0][0]
masked_ll_itv[non_zero_start_indices:non_zero_start_indices+26] = 0
if np.any(masked_ll_itv > 2 * stationary_ar_std):
    certain_zone_begin = np.where(masked_ll_itv > 2 * stationary_ar_std)[0][0]
    ax7.fill_between(time, -0.05, 1.05, where=(time < time[certain_zone_begin]), color='gray', alpha=0.3, label='Uncertain zone')
    ax7.fill_between(time, -0.05, 1.05, where=(time >= time[certain_zone_begin]), color='green', alpha=0.3, label='Certain zone')
else:
    ax7.fill_between(time, -0.05, 1.05, color='gray', alpha=0.3, label='Uncertain zone')



plt.show()