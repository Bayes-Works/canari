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
from pytagi import Normalizer
import copy

from matplotlib import ticker
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
data_file = "./data/toy_time_series/paper_data_decompose_delay_example.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

# LT anomaly
# anm_mag = 0.010416667/10
time_anomaly = 52*4
anm_mag = 15/52
# anm_baseline = np.linspace(0, 3, num=len(df_raw))
anm_baseline = np.arange(len(df_raw)) * anm_mag
# Set the first 52*12 values in anm_baseline to be 0
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
    # LocalTrend(mu_states=model_dict["mu_states"][0:2].reshape(-1), var_states=np.diag(model_dict["var_states"][0:2, 0:2])),
    LocalTrend(mu_states=[0, 0], var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                   mu_states=[model_dict["mu_states"][autoregression_index].item()], 
                   var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
)
gen_model = Model(
    # LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=np.diag(model_dict['states_optimal'].var_prior[0][0:2, 0:2])),\
    LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(phi=model_dict['gen_phi_ar'], std_error=model_dict['gen_sigma_ar'],
                   mu_states=[model_dict["mu_states"][autoregression_index].item()], 
                   var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
)

pretrained_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])
gen_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

ltd_error = 1e-5

hsl_tsad_agent = hsl_detection(base_model=pretrained_model, generate_model=gen_model, data_processor=data_processor, drift_model_process_error_std=ltd_error, y_std_scale = 1)

# Get flexible drift model from the beginning
hsl_tsad_agent_pre = hsl_detection(base_model=pretrained_model.load_dict(pretrained_model.get_dict()), generate_model=gen_model, data_processor=data_processor, drift_model_process_error_std=ltd_error)
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
hsl_tsad_agent.mu_LTd =  -1.1818004808627191e-05
hsl_tsad_agent.LTd_std = 9.836129975831529e-05
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std)
# hsl_tsad_agent.tune_panm_threshold(data=train_val_data)

# hsl_tsad_agent.collect_synthetic_samples(num_time_series=20, save_to_path='data/hsl_tsad_training_samples/itv_learn_samples_paper_example.csv')
hsl_tsad_agent.nn_train_with = 'tagiv'
hsl_tsad_agent.mean_train, hsl_tsad_agent.std_train, hsl_tsad_agent.mean_target, hsl_tsad_agent.std_target = -0.00017661595, 0.00059864233, np.array([-3.2626945e-03, -3.5036656e-01, 1.0573172e+02]), np.array([9.4544115e-03, 1.1906682e+00, 6.1778023e+01])

hsl_tsad_agent.learn_intervention(training_samples_path='data/hsl_tsad_training_samples/itv_learn_samples_paper_example.csv',
                                  load_model_path='saved_params/NN_detection_model_paper_example.pkl', max_training_epoch=20)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data, apply_intervention=False)
mu_ar_preds_all = np.hstack((mu_ar_preds_all, mu_ar_preds.flatten()))
std_ar_preds_all = np.hstack((std_ar_preds_all, std_ar_preds.flatten()))

# #  Plot
state_type = "prior"
#  Plot states from pretrained model
fig = plt.figure(figsize=(5.5, 1.9), constrained_layout=True)
gs = gridspec.GridSpec(3, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])

time = data_processor.get_time(split="all")
# plot_data(
#     data_processor=data_processor,
#     standardization=True,
#     plot_column=output_col,
#     validation_label="y",
#     sub_plot=ax0,
# )
# plot_states(
#     data_processor=data_processor,
#     standardization=True,
#     # states=pretrained_model.states,
#     states=hsl_tsad_agent.base_model.states,
#     states_type=state_type,
#     states_to_plot=['level'],
#     sub_plot=ax0,
# )
# # ax0.axvline(x=time[anm_start_index], color='red', linestyle='--', label='Anomaly start')
# ax0.set_xticklabels([])
# ax0.set_title("HSL Detection & Intervention agent")
# plot_states(
#     data_processor=data_processor,
#     standardization=True,
#     states=hsl_tsad_agent.base_model.states,
#     states_type=state_type,
#     states_to_plot=['trend'],
#     sub_plot=ax1,
# )
# ax1.set_xticklabels([])
# plot_states(
#     data_processor=data_processor,
#     standardization=True,
#     states=hsl_tsad_agent.base_model.states,
#     states_type=state_type,
#     states_to_plot=['lstm'],
#     sub_plot=ax2,
# )
# ax2.set_xticklabels([])
# plot_states(
#     data_processor=data_processor,
#     standardization=True,
#     states=hsl_tsad_agent.base_model.states,
#     states_type=state_type,
#     states_to_plot=['autoregression'],
#     sub_plot=ax3,
# )
# ax3.set_xticklabels([])
# ax4.plot(time, np.array(mu_ar_preds_all), label='obs', color='tab:red')
# ax4.fill_between(time,
#                 np.array(mu_ar_preds_all) - np.array(std_ar_preds_all),
#                 np.array(mu_ar_preds_all) + np.array(std_ar_preds_all),
#                 color='tab:red',
#                 alpha=0.5)
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.base_model.states,
    states_type='prior',
    states_to_plot=['autoregression'],
    sub_plot=ax0,
)
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.drift_model.states,
    states_type=state_type,
    states_to_plot=['level'],
    sub_plot=ax0,
    color='tab:orange',
)
ax0.axvline(x=time[time_anomaly], color='tab:red', linestyle='--', label='Anomaly')
ax0.set_ylabel('$x^{\mathtt{LLd}}$')
ax0.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.drift_model.states,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax1,
    color='tab:orange',
)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.yaxis.offsetText.set_fontsize(6)
ax1.axvline(x=time[time_anomaly], color='tab:red', linestyle='--', label='Anomaly')
ax1.set_ylabel('$x^{\mathtt{LTd}}$')
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.drift_model.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax2,
    color='tab:orange',
)
ax2.axvline(x=time[time_anomaly], color='tab:red', linestyle='--', label='Anomaly')
ax2.set_ylabel('$x^{\mathtt{ARd}}$')
# ax2.set_xticklabels([])
# ax7.plot(time, hsl_tsad_agent.p_anm_all)
# ax7.set_ylabel("p_anm")
# detection_time = np.where(np.array(hsl_tsad_agent.p_anm_all) > hsl_tsad_agent.detection_threshold)[0]
# for i in range(len(detection_time)):
#     ax7.axvline(x=time[detection_time[i]], color='red', linestyle='--', label='Anomaly start')
# ax7.set_xlim(ax0.get_xlim())
# ax7.set_ylim(-0.05, 1.05)

# mu_itv_all = np.array(hsl_tsad_agent.mu_itv_all)
# std_itv_all = np.array(hsl_tsad_agent.std_itv_all)

# ax8.plot(time, mu_itv_all[:, 0])
# ax8.fill_between(time, mu_itv_all[:, 0] - std_itv_all[:, 0], mu_itv_all[:, 0] + std_itv_all[:, 0], alpha=0.5)
# ax8.set_ylabel("itv_LT")
# ax8.set_xlim(ax0.get_xlim())

# ax9.plot(time, mu_itv_all[:, 1])
# ax9.fill_between(time, mu_itv_all[:, 1] - std_itv_all[:, 1], mu_itv_all[:, 1] + std_itv_all[:, 1], alpha=0.5)
# ax9.set_ylabel("itv_LL")
# ax9.set_xlim(ax0.get_xlim())

# ax10.plot(time, mu_itv_all[:, 2])
# ax10.fill_between(time, mu_itv_all[:, 2] - std_itv_all[:, 2], mu_itv_all[:, 2] + std_itv_all[:, 2], alpha=0.5)
# ax10.set_ylabel("itv_time")
# ax10.set_xlim(ax0.get_xlim())
fig.align_ylabels([ax0, ax1, ax2])
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.4)
plt.savefig('filtered_residuals.png', dpi=300)
plt.show()