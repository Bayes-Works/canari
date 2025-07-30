import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari.component import LocalTrend, LstmNetwork, Autoregression, WhiteNoise
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_states,
    common,
)
from src.hsl_detection import hsl_detection
from src.hsl_generator import hsl_generator
import pytagi.metric as metric
from matplotlib import gridspec
import pickle


# # # Read data
data_file = "./data/benchmark_data/detrended_data/test_9_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

# # LT anomaly
# # anm_mag = 0.010416667/10
# # anm_start_index = 52*8
anm_start_index = 52*6
# anm_mag = 0.4/52
# # anm_baseline = np.linspace(0, 3, num=len(df_raw))
# anm_baseline = np.arange(len(df_raw)) * anm_mag
# # Set the first 52*12 values in anm_baseline to be 0
# anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
# anm_baseline[:anm_start_index] = 0
# df_raw = df_raw.add(anm_baseline, axis=0)

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=0.35354,
    validation_split=0.087542,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()


####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
with open("saved_params/real_detrended_ts9_wn_ssm_dist_update.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=33,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
    )
phi_ar = model_dict['phi_ar']
sigma_ar = model_dict['sigma_ar']

print("phi_AR =", phi_ar)
print("sigma_AR =", sigma_ar)

wn_model = Model(
    LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=np.diag(model_dict['states_optimal'].var_prior[0][0:2, 0:2])),
    LSTM,
    WhiteNoise(std_error=model_dict["sigma_v_optimal"]),
)

ar_model = Model(
    LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=np.diag(model_dict['states_optimal'].var_prior[0][0:2, 0:2])),
    LSTM,
    Autoregression(phi=phi_ar, std_error=sigma_ar),
)

wn_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])
ar_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

ltd_error = 1e-5

hsl_tsad_agent = hsl_detection(base_model=wn_model, data_processor=data_processor, drift_model_process_error_std=ltd_error, y_std_scale = 1,
                               phi_ar=phi_ar, sigma_ar=sigma_ar)
hsl_tsad_generator = hsl_generator(base_model=ar_model, data_processor=data_processor, drift_model_process_error_std=ltd_error, y_std_scale = 1,
                               phi_ar=phi_ar, sigma_ar=sigma_ar)

_,_,_,_ = hsl_tsad_generator.filter(train_data, buffer_LTd=False)
_,_,_,_ = hsl_tsad_generator.filter(validation_data, buffer_LTd=False)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(train_data, buffer_LTd=True)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(validation_data, buffer_LTd=True)
# hsl_tsad_agent.estimate_LTd_dist()
hsl_tsad_agent.mu_LTd = 2.7739820777270047e-06
hsl_tsad_agent.LTd_std = 1.4053257192829538e-05
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std * 1)

time_series_path = 'data/hsl_tsad_training_samples/itv_syn_ts_real_detrended_ts9_wn.csv'
# hsl_tsad_generator.generate_time_series(num_time_series=1000, save_to_path=time_series_path)
# hsl_tsad_agent.collect_synthetic_samples(time_series_path=time_series_path, save_to_path='data/hsl_tsad_training_samples/itv_learn_samples_real_ts9_detrended_wn.csv')


hsl_tsad_agent.nn_train_with = 'tagiv'
hsl_tsad_agent.mean_train, hsl_tsad_agent.std_train, hsl_tsad_agent.mean_target, hsl_tsad_agent.std_target = 1.2572588e-05, 9.5143805e-05, np.array([-1.0365549e-03, -1.6639595e-01, 1.0383029e+02]), np.array([1.0482573e-02, 1.2390215e+00, 6.2284431e+01])
# hsl_tsad_agent.tune(begin_std_LTd=2)
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std * 2.7)

hsl_tsad_agent.learn_intervention(training_samples_path='data/hsl_tsad_training_samples/itv_learn_samples_real_ts9_detrended_wn.csv', 
                                  load_model_path='saved_params/NN_detection_model_real_ts9_detrended_wn.pkl', max_training_epoch=50)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data, apply_intervention=True)



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
ax0.set_title("HSL Detection & Intervention agent")
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
    states_to_plot=['white noise'],
    sub_plot=ax3,
)
ax3.set_xticklabels([])
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
ax7.set_ylim(-0.05, 1.05)

mu_itv_all = np.array(hsl_tsad_agent.mu_itv_all)
std_itv_all = np.array(hsl_tsad_agent.std_itv_all)

ax8.plot(time, mu_itv_all[:, 0])
ax8.fill_between(time, mu_itv_all[:, 0] - std_itv_all[:, 0], mu_itv_all[:, 0] + std_itv_all[:, 0], alpha=0.5)
ax8.set_ylabel("itv_LT")
ax8.set_xlim(ax0.get_xlim())

ax9.plot(time, mu_itv_all[:, 1])
ax9.fill_between(time, mu_itv_all[:, 1] - std_itv_all[:, 1], mu_itv_all[:, 1] + std_itv_all[:, 1], alpha=0.5)
ax9.set_ylabel("itv_LL")
ax9.set_xlim(ax0.get_xlim())

ax10.plot(time, mu_itv_all[:, 2])
ax10.fill_between(time, mu_itv_all[:, 2] - std_itv_all[:, 2], mu_itv_all[:, 2] + std_itv_all[:, 2], alpha=0.5)
ax10.set_ylabel("itv_time")
ax10.set_xlim(ax0.get_xlim())
plt.show()