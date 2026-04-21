import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari.component import LocalTrend, LstmNetwork, Autoregression
from canari import (
    DataProcess,
    Model,
    # plot_data,
    # plot_states,
    common,
)
from src.canari.data_visualization_nogrid import plot_data, plot_states
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

# # LL anomaly
# # anm_mag = 0.010416667/10
# time_anomaly = 52*4
# anm_mag = 50
# # anm_baseline = np.linspace(0, 3, num=len(df_raw))
# anm_baseline = np.ones(len(df_raw)) * anm_mag
# anm_baseline[:time_anomaly] = 0
# df_raw = df_raw.add(anm_baseline, axis=0)

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

# Normalize the anm_mag
anm_mag_normalized = anm_mag / data_processor.scale_const_std[0]
anm_baseline_normalized = anm_baseline / data_processor.scale_const_std[0]

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
hsl_tsad_agent.detection_threshold = 0.5


# hsl_tsad_agent.collect_synthetic_samples(num_time_series=10, save_to_path='data/hsl_tsad_training_samples/itv_learn_samples_paper_example_dummy.csv')
hsl_tsad_agent.nn_train_with = 'tagiv'
hsl_tsad_agent.mean_train, hsl_tsad_agent.std_train, hsl_tsad_agent.mean_target, hsl_tsad_agent.std_target = -5.740547e-05, 0.0006672608, np.array([-6.4706226e-04, -8.1444547e-02, 1.1747440e+02]), np.array([1.0015702e-02, 1.3998920e+00, 7.0780159e+01])

hsl_tsad_agent.learn_intervention(training_samples_path='data/hsl_tsad_training_samples/itv_learn_samples_paper_example_dummy.csv',
                                  load_model_path='saved_params/NN_detection_model_paper_example_dummy.pkl', max_training_epoch=5)
# Split test_data into two parts at 3/4
add_intervention_proportion = 0.02
test_data_part1 = {
    "x": test_data["x"][0:int(len(test_data["x"])*add_intervention_proportion), :],
    "y": test_data["y"][0:int(len(test_data["y"])*add_intervention_proportion), :],
}
test_data_part2 = {
    "x": test_data["x"][int(len(test_data["x"])*add_intervention_proportion):, :],
    "y": test_data["y"][int(len(test_data["y"])*add_intervention_proportion):, :],
}

# RSI: issue 1, illustration
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data_part1, apply_intervention=False)
LT_index = hsl_tsad_agent.base_model.states_name.index("trend")
LL_index = hsl_tsad_agent.base_model.states_name.index("level")
hsl_tsad_agent.base_model.mu_states[LT_index] += anm_mag_normalized
# hsl_tsad_agent.base_model.mu_states[LL_index] += anm_mag_normalized * ( hsl_tsad_agent.current_time_step - time_anomaly)
hsl_tsad_agent.base_model.var_states[LT_index, LT_index] += 0.000025
hsl_tsad_agent.base_model.var_states[LL_index, LL_index] += 0.0025
hsl_tsad_agent.drift_model.mu_states[LL_index] = 0
hsl_tsad_agent.drift_model.mu_states[LT_index] = 0
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data_part2, apply_intervention=False)
mu_ar_preds_all = np.hstack((mu_ar_preds_all, mu_ar_preds.flatten()))
std_ar_preds_all = np.hstack((std_ar_preds_all, std_ar_preds.flatten()))

# # RSI: issue 2, illustration
# mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data, apply_intervention=True)

# Get all the detection point
detection_points = np.where(np.array(hsl_tsad_agent.p_anm_all) > hsl_tsad_agent.detection_threshold)[0]

# #  Plot
state_type = "prior"
#  Plot states from pretrained model
fig = plt.figure(figsize=(6, 2), constrained_layout=True)
gs = gridspec.GridSpec(6, 1, figure=fig, height_ratios=[2, 1, 1, 1, 1, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
ax4 = plt.subplot(gs[4])
ax5 = plt.subplot(gs[5])
# ax6 = plt.subplot(gs[6])

time = data_processor.get_time(split="all")
plot_data(
    data_processor=data_processor,
    standardization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax0,
    color='k',
)
ax0.plot(time, anm_baseline_normalized, color='k', alpha=0.2)
# for dp in detection_points:
#     ax0.axvline(x=time[dp], color='r', linestyle='--')
# plot_states(
#     data_processor=data_processor,
#     standardization=True,
#     states=hsl_tsad_agent.base_model.states,
#     states_type=state_type,
#     states_to_plot=['level'],
#     sub_plot=ax0,
# )
ax0.axvline(x=time[time_anomaly], color='k', linestyle='--', label='Anomaly start')
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax1,
    color='tab:orange',
)
ax2.plot(time, hsl_tsad_agent.p_anm_all, color='tab:orange')
# plot_states(
#     data_processor=data_processor,
#     standardization=True,
#     states=hsl_tsad_agent.base_model.states,
#     states_type=state_type,
#     states_to_plot=['lstm'],
#     sub_plot=ax2,
#     color='tab:orange',
# )
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax3,
    color='tab:orange',
)
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.drift_model.states,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax4,
    color='tab:orange',
)
# ax4.axvline(x=time[time_anomaly], color='tab:red', linestyle='--', label='Anomaly')
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=hsl_tsad_agent.drift_model.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax5,
    color='tab:orange',
)

for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
    ax.grid(False)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.4)
plt.savefig('rsi_issue.png', dpi=300)
plt.show()