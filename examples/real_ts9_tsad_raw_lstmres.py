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
from src.hsl_detection_new import hsl_detection
from src.hsl_generator_new import hsl_generator
import pytagi.metric as metric
from matplotlib import gridspec
import pickle
from pytagi import Normalizer


# # # Read data
data_file = "./data/benchmark_data/test_9_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]

# Data pre-processing
output_col = [0]
# train_split=0.28
# validation_split=0.07
train_split=0.35354
validation_split=0.087542
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=train_split,
    validation_split=validation_split,
    output_col=output_col,
)
data_processor.scale_const_mean, data_processor.scale_const_std = Normalizer.compute_mean_std(
                data_processor.data.iloc[0 : 52].values
            )


train_data, validation_data, test_data, normalized_data = data_processor.get_splits()


####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
with open("saved_params/real_ts9_tsmodel_raw_lstmres.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=27,
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
    LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                   mu_states=[model_dict["mu_states"][autoregression_index].item()], 
                   var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
)

generate_model = Model(
    LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=model_dict['gen_sigma_ar'], 
                   phi=model_dict['gen_phi_ar']),
)

pretrained_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])
generate_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

ltd_error = 1e-5

hsl_tsad_agent = hsl_detection(base_model=pretrained_model, data_processor=data_processor, drift_model_process_error_std=ltd_error, y_std_scale = 1)
hsl_tsad_generator = hsl_generator(base_model=generate_model, data_processor=data_processor, drift_model_process_error_std=ltd_error, y_std_scale = 1)

# Get flexible drift model from the beginning
hsl_tsad_agent_pre = hsl_detection(base_model=pretrained_model.load_dict(pretrained_model.get_dict()), data_processor=data_processor, drift_model_process_error_std=ltd_error)
hsl_tsad_agent_pre.filter(train_data)
hsl_tsad_agent_pre.filter(validation_data)
hsl_tsad_agent.drift_model.var_states = hsl_tsad_agent_pre.drift_model.var_states

mu_ar_preds_all, std_ar_preds_all = [], []
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(train_data, buffer_LTd=True)
mu_ar_preds_all = np.hstack((mu_ar_preds_all, mu_ar_preds.flatten()))
std_ar_preds_all = np.hstack((std_ar_preds_all, std_ar_preds.flatten()))
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(validation_data, buffer_LTd=True)
mu_ar_preds_all = np.hstack((mu_ar_preds_all, mu_ar_preds.flatten()))
std_ar_preds_all = np.hstack((std_ar_preds_all, std_ar_preds.flatten()))
# hsl_tsad_agent.estimate_LTd_dist()

_, _, _, _ = hsl_tsad_generator.filter(train_data, buffer_LTd=True)
_, _, _, _ = hsl_tsad_generator.filter(validation_data, buffer_LTd=True)

time_series_path = 'data/hsl_tsad_training_samples/itv_syn_ts_real_ts9_lstmres.csv'
# hsl_tsad_generator.generate_time_series(num_time_series=1000, save_to_path=time_series_path)
# hsl_tsad_agent.collect_synthetic_samples(time_series_path=time_series_path, save_to_path='data/hsl_tsad_training_samples/itv_learn_samples_real_ts9_detrended_raw_lstmres.csv')

hsl_tsad_agent.mu_LTd = -0.0018000274853443186
hsl_tsad_agent.LTd_std = 0.0027839103718582665
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std * 1)

hsl_tsad_agent.nn_train_with = 'tagiv'
hsl_tsad_agent.mean_train, hsl_tsad_agent.std_train, hsl_tsad_agent.mean_target, hsl_tsad_agent.std_target = 0.00032299958, 0.003925998, np.array([-3.4595627e-04, -4.1475490e-02, 1.0706683e+02]), np.array([1.1326723e-02, 1.4040886e+00, 6.2480057e+01])
# hsl_tsad_agent.tune()

hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std * 0.28242953648100017)

hsl_tsad_agent.learn_intervention(training_samples_path='data/hsl_tsad_training_samples/itv_learn_samples_real_ts9_detrended_raw_lstmres.csv', 
                                  load_model_path='saved_params/NN_detection_model_real_ts9_detrended_raw_lstmres.pkl', max_training_epoch=50)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data, apply_intervention=True)
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