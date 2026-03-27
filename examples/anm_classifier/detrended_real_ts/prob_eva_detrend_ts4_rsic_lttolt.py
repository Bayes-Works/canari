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
from src.hsl_classification_2classes_rsic_v2_w5 import hsl_classification
import pytagi.metric as metric
import pickle
import ast
from tqdm import tqdm
from matplotlib import gridspec
from canari.data_visualization import _add_dynamic_grids


# # # Read data
data_file = "./data/benchmark_data/detrended_data/test_4_data_detrended.csv"
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
# Get scale constants
scale_const_mean = copy.deepcopy(data_processor.scale_const_mean)
scale_const_std = copy.deepcopy(data_processor.scale_const_std)
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

df = pd.read_csv("data/prob_eva_syn_time_series/detrend_rsic_simple_ts4_gen_lttolt.csv")

# Containers for restored data
restored_data = []
time_stamps = eval(df.iloc[0]["timestamp"], {"nan": float("nan")})
for _, row in df.iterrows():
    values = np.array(eval(row["values"], {"nan": float("nan")}), dtype=float)
    anomaly1_magnitude = float(row["anomaly1_magnitude"])
    anomaly2_magnitude = float(row["anomaly2_magnitude"])
    anomaly_start_index1 = int(row["anomaly_start_index1"])
    anomaly_start_index2 = int(row["anomaly_start_index2"])
    
    restored_data.append((values, anomaly1_magnitude, anomaly2_magnitude, anomaly_start_index1, anomaly_start_index2))

####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
with open("saved_params/real_ts4_tsmodel_detrended.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=51,
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

mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(train_data, buffer_LTd=True)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(validation_data, buffer_LTd=True)
# hsl_tsad_agent.estimate_LTd_dist()
hsl_tsad_agent.mu_LTd = 2.3102192245878576e-05
hsl_tsad_agent.LTd_std = 7.574552337904965e-05
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std)
# hsl_tsad_agent.tune_panm_threshold(data=train_val_data)
hsl_tsad_agent.detection_threshold = 0.1

hsl_tsad_agent.mean_LTd_class, hsl_tsad_agent.std_LTd_class = 2.2566534e-05, 0.00018387675
hsl_tsad_agent.mean_target_lt_model, hsl_tsad_agent.std_target_lt_model = np.array([-0.00010463, -0.01490419]), np.array([0.00673775, 0.8857056])
hsl_tsad_agent.mean_target_ll_model, hsl_tsad_agent.std_target_ll_model = np.array([0.03493336]), np.array([0.6830585])

hsl_tsad_agent.learn_intervention(training_samples_path='data/anm_type_class_train_samples/classifier_learn_samples_detrended_ts4.csv', 
                                    load_lt_model_path='saved_params/NN_intervention_LT_model_rsic_detrend_ts4.pkl', 
                                    load_ll_model_path='saved_params/NN_intervention_LL_model_rsic_detrend_ts4.pkl', 
                                    max_training_epoch=50)

# Store the states, mu_states, var_states, lstm_cell_states, and lstm_output_history of base_model
states_temp = copy.deepcopy(hsl_tsad_agent.base_model.states)
mu_states_temp = copy.deepcopy(hsl_tsad_agent.base_model.mu_states)
var_states_temp = copy.deepcopy(hsl_tsad_agent.base_model.var_states)
lstm_states_temp = copy.deepcopy(hsl_tsad_agent.base_model.lstm_net.get_lstm_states())
lstm_output_history_temp = copy.deepcopy(hsl_tsad_agent.base_model.lstm_output_history)
lstm_history_temp = copy.deepcopy(hsl_tsad_agent.lstm_history)
lstm_cell_states_temp = copy.deepcopy(hsl_tsad_agent.lstm_cell_states)

drift_states_temp = copy.deepcopy(hsl_tsad_agent.drift_model.states)
mu_drift_states_temp = copy.deepcopy(hsl_tsad_agent.drift_model.mu_states)
var_drift_states_temp = copy.deepcopy(hsl_tsad_agent.drift_model.var_states)

# drift2_states_temp = copy.deepcopy(hsl_tsad_agent.drift_model2.states)
# mu_drift2_states_temp = copy.deepcopy(hsl_tsad_agent.drift_model2.mu_states)
# var_drift2_states_temp = copy.deepcopy(hsl_tsad_agent.drift_model2.var_states)

current_time_step_temp = copy.deepcopy(hsl_tsad_agent.current_time_step)
p_anm_all_temp = copy.deepcopy(hsl_tsad_agent.p_anm_all)
mu_itv_all_temp = copy.deepcopy(hsl_tsad_agent.mu_itv_all)
std_itv_all_temp = copy.deepcopy(hsl_tsad_agent.std_itv_all)

class_prob_moments_temp = copy.deepcopy(hsl_tsad_agent.class_prob_moments)
ll_itv_all_temp = copy.deepcopy(hsl_tsad_agent.ll_itv_all)    # For visualization, could be removed
lt_itv_all_temp = copy.deepcopy(hsl_tsad_agent.lt_itv_all)    # For visualization, could be removed  
test_mu_obs_preds_temp = copy.deepcopy(hsl_tsad_agent.mu_obs_preds)     # To improve naming, hsl_tsad_agent.mu_obs_preds only store the test part
test_std_obs_preds_temp = copy.deepcopy(hsl_tsad_agent.std_obs_preds)   # To improve naming, hsl_tsad_agent.std_obs_preds only store the test part

results_all = []
for m in range(10):
    for n in tqdm(range(len(restored_data)//10)):
        # n += 10
        k = m + n * 10
        # Create a new pandas dataframe df_k, with one column filled with restored_data[k][0], and index as time_stamps
        df_k = pd.DataFrame()
        df_k["obs"] = restored_data[k][0]
        df_k.index = pd.to_datetime(time_stamps)

        # Anomaly info
        anm_mag1 = restored_data[k][1]
        anm_mag2 = restored_data[k][2]
        anm_start_index1 = restored_data[k][3]
        anm_start_index2 = restored_data[k][4]

        data_processor_k = DataProcess(
            data=df_k,
            time_covariates=["week_of_year"],
            train_split=train_split,
            validation_split=validation_split,
            output_col=output_col,
        )
        data_processor_k.train_start = data_processor.train_start
        data_processor_k.train_end = data_processor.train_end
        data_processor_k.validation_start = data_processor.validation_start
        data_processor_k.validation_end = data_processor.validation_end
        data_processor_k.test_start = data_processor.test_start
        data_processor_k.test_end = len(df_k)
        data_processor_k._compute_standardization_constants()
        _, _, test_data_k, _ = data_processor_k.get_splits()

        mu_obs_preds, std_obs_preds, itv_log, itv_applied_times = hsl_tsad_agent.detect(test_data_k, apply_intervention=False)

        all_detection_points = str(np.where(np.array(hsl_tsad_agent.p_anm_all) > hsl_tsad_agent.detection_threshold)[0].tolist())
        itv_log = str(itv_log.tolist())
        itv_applied_times = str(itv_applied_times.tolist())

        # Get baselines for comparison
        # Baseline estimation
        mu_LL_states = hsl_tsad_agent.base_model.states.get_mean(states_type='prior', states_name="level", standardization=True)
        mu_LT_states = hsl_tsad_agent.base_model.states.get_mean(states_type='prior', states_name="trend", standardization=True, scale_const_mean=scale_const_mean[0], scale_const_std=scale_const_std[0])
        # True baselines
        true_LL_baseline = np.zeros(len(df_k))
        true_LT_baseline = np.zeros(len(df_k))
        # LT to LT anomaly
        anm_mag1_perweek = anm_mag1 / 52
        anm_mag2_perweek = anm_mag2 / 52
        true_LL_baseline[anm_start_index1:] += np.arange(len(true_LL_baseline)-anm_start_index1) * anm_mag1_perweek
        true_LL_baseline[anm_start_index2:] += np.arange(len(true_LL_baseline)-anm_start_index2) * anm_mag2_perweek
        true_LT_baseline[anm_start_index1:] += anm_mag1_perweek
        true_LT_baseline[anm_start_index2:] += anm_mag2_perweek

        # Convert the baselines to strings and save to results_all
        true_LL_baseline_str = str(true_LL_baseline.tolist())
        true_LT_baseline_str = str(true_LT_baseline.tolist())
        estimate_LL_baseline_str = str(mu_LL_states.tolist())
        estimate_LT_baseline_str = str(mu_LT_states.tolist())
        
        results_all.append([anm_mag2, anm_start_index1, anm_start_index2, all_detection_points, itv_log, itv_applied_times, true_LL_baseline_str, true_LT_baseline_str, estimate_LL_baseline_str, estimate_LT_baseline_str])

        # # #  Plot
        # state_type = "posterior"
        # fig = plt.figure(figsize=(10, 8))
        # gs = gridspec.GridSpec(7, 1)
        # ax0 = plt.subplot(gs[0])
        # ax1 = plt.subplot(gs[1])
        # ax2 = plt.subplot(gs[2])
        # ax3 = plt.subplot(gs[3])
        # ax4 = plt.subplot(gs[4])
        # ax5 = plt.subplot(gs[5])
        # # ax6 = plt.subplot(gs[6])
        # ax7 = plt.subplot(gs[6])
        # time = data_processor_k.get_time(split="all")
        # plot_data(
        #     data_processor=data_processor_k,
        #     standardization=True,
        #     plot_column=output_col,
        #     validation_label="y",
        #     sub_plot=ax0,
        # )
        # plot_states(
        #     data_processor=data_processor_k,
        #     standardization=True,
        #     states=hsl_tsad_agent.base_model.states,
        #     states_type=state_type,
        #     states_to_plot=['level'],
        #     sub_plot=ax0,
        # )
        # ax0.plot(time, true_LL_baseline, 'k--')
        # ax0.set_xticklabels([])

        # ############ Original plots #############
        # plot_states(
        #     data_processor=data_processor_k,
        #     standardization=True,
        #     states=hsl_tsad_agent.base_model.states,
        #     states_type=state_type,
        #     states_to_plot=['trend'],
        #     sub_plot=ax1,
        # )
        # ax1.plot(time, true_LT_baseline, 'k--')
        # ax1.set_xticklabels([])
        # plot_states(
        #     data_processor=data_processor_k,
        #     standardization=True,
        #     states=hsl_tsad_agent.base_model.states,
        #     states_type=state_type,
        #     states_to_plot=['lstm'],
        #     sub_plot=ax2,
        # )
        # ax2.set_xticklabels([])
        # plot_states(
        #     data_processor=data_processor_k,
        #     standardization=True,
        #     states=hsl_tsad_agent.base_model.states,
        #     states_type=state_type,
        #     states_to_plot=['autoregression'],
        #     sub_plot=ax3,
        # )
        # ax3.set_xticklabels([])

        # ax4.plot(time, hsl_tsad_agent.p_anm_all)
        # detection_time = np.where(np.array(hsl_tsad_agent.p_anm_all) > hsl_tsad_agent.detection_threshold)[0]
        # for i in range(len(detection_time)):
        #     ax4.axvline(x=time[detection_time[i]], color='red', linestyle='--', label='Anomaly start')
        # ax4.set_ylabel("p_anm")
        # ax4.set_xlim(ax0.get_xlim())
        # ax4.set_ylim(-0.05, 1.05)
        # _add_dynamic_grids(ax4, time)

        # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        # final_class_log_probs = np.array(hsl_tsad_agent.class_prob_moments)[:, 0:2]
        # final_class_prob_stds = np.array(hsl_tsad_agent.class_prob_moments)[:, 2]

        # gen_ar_phi = model_dict['gen_phi_ar']
        # gen_ar_sigma =model_dict['gen_sigma_ar']
        # stationary_ar_std = np.sqrt(gen_ar_sigma**2 / (1 - gen_ar_phi**2))
        # ax5.fill_between(time, - 2 * stationary_ar_std, 2 * stationary_ar_std, color='gray', alpha=0.3, label='2-Sigma range')
        # ax5.plot(time, hsl_tsad_agent.ll_itv_all, label='LL itv', color='tab:blue')
        # ax5.plot(time, hsl_tsad_agent.lt_itv_all, label='LT itv', color='tab:orange')
        # ax5.set_ylabel("itv")

        # for class_idx in range(final_class_log_probs.shape[1]):
        #     ax7.plot(time, final_class_log_probs[:, class_idx], color=colors[class_idx])
        #     ax7.fill_between(time, final_class_log_probs[:, class_idx] - final_class_prob_stds, 
        #                     final_class_log_probs[:, class_idx] + final_class_prob_stds, color=colors[class_idx], alpha=0.3)
        # ax7.set_ylim(-0.05, 1.05)
        # ax7.set_ylabel("Pr(anm)")
        # plt.show()

        # Put back the states, mu_states, var_states, lstm_cell_states, and lstm_output_history of base_model
        hsl_tsad_agent.base_model.states = copy.deepcopy(states_temp)
        hsl_tsad_agent.base_model.mu_states = copy.deepcopy(mu_states_temp)
        hsl_tsad_agent.base_model.var_states = copy.deepcopy(var_states_temp)
        hsl_tsad_agent.base_model.lstm_net.set_lstm_states(lstm_states_temp)
        hsl_tsad_agent.base_model.lstm_output_history = copy.deepcopy(lstm_output_history_temp)
        hsl_tsad_agent.drift_model.states = copy.deepcopy(drift_states_temp)
        hsl_tsad_agent.drift_model.mu_states = copy.deepcopy(mu_drift_states_temp)
        hsl_tsad_agent.drift_model.var_states = copy.deepcopy(var_drift_states_temp)
        # hsl_tsad_agent.drift_model2.states = copy.deepcopy(drift2_states_temp)
        # hsl_tsad_agent.drift_model2.mu_states = copy.deepcopy(mu_drift2_states_temp)
        # hsl_tsad_agent.drift_model2.var_states = copy.deepcopy(var_drift2_states_temp)
        hsl_tsad_agent.current_time_step = copy.deepcopy(current_time_step_temp)
        hsl_tsad_agent.p_anm_all = copy.deepcopy(p_anm_all_temp)
        hsl_tsad_agent.mu_itv_all = copy.deepcopy(mu_itv_all_temp)
        hsl_tsad_agent.std_itv_all = copy.deepcopy(std_itv_all_temp)
        hsl_tsad_agent.class_prob_moments = copy.deepcopy(class_prob_moments_temp)
        hsl_tsad_agent.ll_itv_all = copy.deepcopy(ll_itv_all_temp)
        hsl_tsad_agent.lt_itv_all = copy.deepcopy(lt_itv_all_temp)
        hsl_tsad_agent.mu_obs_preds = copy.deepcopy(test_mu_obs_preds_temp)
        hsl_tsad_agent.std_obs_preds = copy.deepcopy(test_std_obs_preds_temp)
        hsl_tsad_agent.lstm_history = copy.deepcopy(lstm_history_temp)
        hsl_tsad_agent.lstm_cell_states = copy.deepcopy(lstm_cell_states_temp)

# plt.show()

# Save the results to a CSV file
results_df = pd.DataFrame(results_all, columns=["anomaly_magnitude", "anomaly_start_index1", "anomaly_start_index2", "anomaly_detected_index", "intervention_log", "intervention_applied_times", "true_LL_baseline", "true_LT_baseline", "estimated_LL_baseline", "estimated_LT_baseline"])
results_df.to_csv("saved_results/prob_eva/detrend_ts4_results_rsic_lttolt.csv", index=False)