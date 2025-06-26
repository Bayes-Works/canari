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
from src.hsl_detection import hsl_detection
import pytagi.metric as metric
import pickle
import ast
from tqdm import tqdm
from matplotlib import gridspec
from canari.data_visualization import _add_dynamic_grids


# # # Read raw data
data_file = "./data/benchmark_data/detrended_data/test_5_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

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

# # # Read test data
df = pd.read_csv("data/prob_eva_syn_time_series/detrended_ts5_tsgen.csv")

# Containers for restored data
restored_data = []
for _, row in df.iterrows():
    values = np.array(eval(row["values"], {"nan": float("nan")}), dtype=float)
    anomaly_magnitude = float(row["anomaly_magnitude"])
    anomaly_start_index = int(row["anomaly_start_index"])
    
    restored_data.append((values, anomaly_magnitude, anomaly_start_index))

# # Plot generated time series
# fig = plt.figure(figsize=(10, 6))
# gs = gridspec.GridSpec(1, 1)
# ax0 = plt.subplot(gs[0])
# for j in range(len(restored_data)):
#     ax0.plot(restored_data[j][0])
#     ax0.axvline(x=restored_data[j][2]+len(df_raw)-len(test_data["y"]), color='r', linestyle='--')
# # ax0.axvline(x=len(self.data_processor.data.values[train_index, self.data_processor.output_col].reshape(-1))+len(self.data_processor.data.values[val_index, self.data_processor.output_col].reshape(-1)), color='r', linestyle='--')
# ax0.set_title("Data generation")
# plt.show()

####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
# with open("saved_params/real_ts5_detrend_tsmodel.pkl", "rb") as f:
with open("saved_params/real_ts5_detrend_tsmodel_better.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=16,
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
    LocalTrend(mu_states=model_dict["mu_states"][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                   mu_states=[model_dict["mu_states"][autoregression_index].item()], 
                   var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
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
# hsl_tsad_agent.mu_LTd = 3.312482141267754e-05
# hsl_tsad_agent.LTd_std = 3.881592825882659e-05
hsl_tsad_agent.mu_LTd = -1.1292506864266692e-05
hsl_tsad_agent.LTd_std = 6.462098601531065e-05
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std)

hsl_tsad_agent.nn_train_with = 'tagiv'
# hsl_tsad_agent.mean_train, hsl_tsad_agent.std_train, hsl_tsad_agent.mean_target, hsl_tsad_agent.std_target = -1.3397412e-05, 0.00061888463, np.array([-3.4437198e-04, -4.0980659e-02, 1.0753817e+02]), np.array([1.1108804e-02, 1.3843211e+00, 6.2692867e+01])
hsl_tsad_agent.mean_train, hsl_tsad_agent.std_train, hsl_tsad_agent.mean_target, hsl_tsad_agent.std_target = 1.5701287e-05, 0.0015430708, np.array([2.65250510e-05, 7.54264649e-03, 1.03558205e+02]), np.array([1.0685091e-02, 1.2410772e+00, 6.1538734e+01])
hsl_tsad_agent.learn_intervention(training_samples_path='data/hsl_tsad_training_samples/itv_learn_samples_real_ts5better_detrended.csv', 
                                  load_model_path='saved_params/NN_detection_model_real_ts5better_detrended.pkl', max_training_epoch=50)
# hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std * 0.6)
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std * 1.1)

# Store the states, mu_states, var_states, lstm_cell_states, and lstm_output_history of base_model
states_temp = copy.deepcopy(hsl_tsad_agent.base_model.states)
mu_states_temp = copy.deepcopy(hsl_tsad_agent.base_model.mu_states)
var_states_temp = copy.deepcopy(hsl_tsad_agent.base_model.var_states)
lstm_cell_states_temp = copy.deepcopy(hsl_tsad_agent.base_model.lstm_net.get_lstm_states())
lstm_output_history_temp = copy.deepcopy(hsl_tsad_agent.base_model.lstm_output_history)

drift_states_temp = copy.deepcopy(hsl_tsad_agent.drift_model.states)
mu_drift_states_temp = copy.deepcopy(hsl_tsad_agent.drift_model.mu_states)
var_drift_states_temp = copy.deepcopy(hsl_tsad_agent.drift_model.var_states)

current_time_step_temp = copy.deepcopy(hsl_tsad_agent.current_time_step)
p_anm_all_temp = copy.deepcopy(hsl_tsad_agent.p_anm_all)
mu_itv_all_temp = copy.deepcopy(hsl_tsad_agent.mu_itv_all)
std_itv_all_temp = copy.deepcopy(hsl_tsad_agent.std_itv_all)

results_all = []

for k in tqdm(range(len(restored_data))):
    df_k = copy.deepcopy(df_raw)
    # Replace the values in the dataframe with the restored_data[k][0]
    df_k.iloc[:, 0] = restored_data[k][0]

    data_processor_k = DataProcess(
        data=df_k,
        time_covariates=["week_of_year"],
        train_split=train_split,
        validation_split=validation_split,
        output_col=output_col,
    )
    _, _, test_data_k, _ = data_processor_k.get_splits()

    anm_start_index = restored_data[k][2]
    anm_mag = restored_data[k][1]
    anm_start_index_global = anm_start_index + len(df_k) - len(test_data_k["y"])

    mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data_k, apply_intervention=True)

    all_detection_points = str(np.where(np.array(hsl_tsad_agent.p_anm_all) > 0.5)[0].tolist())
    if (np.array(hsl_tsad_agent.p_anm_all) > 0.5).any():
        anm_detected_index = np.where(np.array(hsl_tsad_agent.p_anm_all) > 0.5)[0][0]
    else:
        anm_detected_index = len(hsl_tsad_agent.p_anm_all)

    # Get true baseline
    anm_mag_normed = anm_mag
    LL_baseline_true = np.zeros_like(df_raw)
    LT_baseline_true = np.zeros_like(df_raw)
    for i in range(1, len(df_raw)):
        if i > anm_start_index_global:
            LL_baseline_true[i] += anm_mag_normed * (i - anm_start_index_global)
            LT_baseline_true[i] += anm_mag_normed

    LL_baseline_true += model_dict['early_stop_init_mu_states'][0].item()
    LL_baseline_true = LL_baseline_true.flatten()
    LT_baseline_true = LT_baseline_true.flatten()

    # Compute MSE for SKF baselines
    mu_LL_states = hsl_tsad_agent.base_model.states.get_mean(states_type='prior', states_name="level")
    mu_LT_states = hsl_tsad_agent.base_model.states.get_mean(states_type='prior', states_name="trend")
    mse_LL = metric.mse(
        mu_LL_states[anm_start_index_global+1:],
        LL_baseline_true[anm_start_index_global+1:],
    )
    mse_LT = metric.mse(
        mu_LT_states[anm_start_index_global+1:],
        LT_baseline_true[anm_start_index_global+1:],
    )

    detection_time = anm_detected_index - anm_start_index_global
    results_all.append([anm_mag, anm_start_index_global, all_detection_points, mse_LL, mse_LT, detection_time])


    # # Plot all the baselines, true and estimated
    # time = data_processor_k.get_time(split="all")
    # plt.figure()
    # plt.plot(time, LL_baseline_true, label="True", color='blue')
    # plt.plot(time, mu_LL_states, label="Online", color='red')
    # plt.axvline(x=time[anm_start_index], color='k', linestyle='--')
    # plt.legend()
    # plt.ylabel('LL')

    # plt.figure()
    # plt.plot(time, LT_baseline_true, label="True", color='blue')
    # plt.plot(time, mu_LT_states, label="Online", color='red')
    # plt.axvline(x=time[anm_start_index], color='k', linestyle='--')
    # plt.legend()
    # plt.ylabel('LT')
    # plt.show()

    detection_time = anm_detected_index - anm_start_index_global

    results_all.append([anm_mag, anm_start_index_global, all_detection_points, mse_LL, mse_LT, detection_time])

    #  Plot
    state_type = "prior"
    #  Plot states from pretrained model
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(5, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    ax4 = plt.subplot(gs[4])
    time = data_processor_k.get_time(split="all")
    plot_data(
        data_processor=data_processor_k,
        standardization=True,
        plot_column=output_col,
        validation_label="y",
        sub_plot=ax0,
    )
    plot_states(
        data_processor=data_processor_k,
        standardization=True,
        # states=pretrained_model.states,
        states=hsl_tsad_agent.base_model.states,
        states_type=state_type,
        states_to_plot=['level'],
        sub_plot=ax0,
    )
    ax0.set_xticklabels([])
    ax0.plot(time, LL_baseline_true, color='k', linestyle='--')
    ax0.axvline(x=time[anm_start_index_global], color='r', linestyle='--')
    ax0.set_title(f"IL, mse_LL = {mse_LL:.3e}, mse_LT = {mse_LT:.3e}, detection_time = {detection_time}")
    plot_states(
        data_processor=data_processor_k,
        standardization=True,
        states=hsl_tsad_agent.base_model.states,
        states_type=state_type,
        states_to_plot=['trend'],
        sub_plot=ax1,
    )
    ax1.set_xticklabels([])
    ax1.plot(time, LT_baseline_true, color='k', linestyle='--')
    plot_states(
        data_processor=data_processor_k,
        standardization=True,
        states=hsl_tsad_agent.base_model.states,
        states_type=state_type,
        states_to_plot=['lstm'],
        sub_plot=ax2,
    )
    ax2.set_xticklabels([])
    plot_states(
        data_processor=data_processor_k,
        standardization=True,
        states=hsl_tsad_agent.base_model.states,
        states_type=state_type,
        states_to_plot=['autoregression'],
        sub_plot=ax3,
    )
    ax3.set_xticklabels([])

    ax4.plot(time, hsl_tsad_agent.p_anm_all, color='b')
    ax4.set_ylabel("p_anm")
    ax4.set_xlim(ax0.get_xlim())
    ax4.set_ylim(-0.05, 1.05)
    _add_dynamic_grids(ax4, time)
    plt.show()

    # Put back the states, mu_states, var_states, lstm_cell_states, and lstm_output_history of base_model
    hsl_tsad_agent.base_model.states = copy.deepcopy(states_temp)
    hsl_tsad_agent.base_model.mu_states = copy.deepcopy(mu_states_temp)
    hsl_tsad_agent.base_model.var_states = copy.deepcopy(var_states_temp)
    hsl_tsad_agent.base_model.lstm_net.set_lstm_states(lstm_cell_states_temp)
    hsl_tsad_agent.base_model.lstm_output_history = copy.deepcopy(lstm_output_history_temp)
    hsl_tsad_agent.drift_model.states = copy.deepcopy(drift_states_temp)
    hsl_tsad_agent.drift_model.mu_states = copy.deepcopy(mu_drift_states_temp)
    hsl_tsad_agent.drift_model.var_states = copy.deepcopy(var_drift_states_temp)
    hsl_tsad_agent.current_time_step = copy.deepcopy(current_time_step_temp)
    hsl_tsad_agent.p_anm_all = copy.deepcopy(p_anm_all_temp)
    hsl_tsad_agent.mu_itv_all = copy.deepcopy(mu_itv_all_temp)
    hsl_tsad_agent.std_itv_all = copy.deepcopy(std_itv_all_temp)

# Save the results to a CSV file
results_df = pd.DataFrame(results_all, columns=["anomaly_magnitude", "anomaly_start_index", "anomaly_detected_index", "mse_LL", "mse_LT",  "detection_time"])
results_df.to_csv("saved_results/prob_eva/detrended_ts5better_results_il.csv", index=False)