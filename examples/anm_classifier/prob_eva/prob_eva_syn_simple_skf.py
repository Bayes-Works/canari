import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, Autoregression
import copy
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_data,
    plot_states,
)
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
import pickle

import ast
from tqdm import tqdm
import random

# # # Read data
data_file = "./data/toy_time_series/syn_data_anmtype_simple_phi05.csv"
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

scale_const_mean = copy.deepcopy(data_processor.scale_const_mean)
scale_const_std = copy.deepcopy(data_processor.scale_const_std)
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()
validation_start = df_raw.index[data_processor.validation_start]
test_start = df_raw.index[data_processor.test_start]

####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
with open("saved_params/ssm_ts_anmtype_simple_phi05.pkl", "rb") as f:
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

stdtrans_normtoab_probthred_combs = [
                                     [0.00058540561184052, 1.781048611155288e-05, 0.01856760759164357],
                                     [0.0005400179445756177, 0.0001968681383225995, 0.12707745623164831],
                                     [0.0005691735027308731, 7.291704860580095e-06, 0.19581591960636593],
                                     [0.00022547346182363125, 0.0001733910644685438, 0.21764965538565006],
                                     [0.0004224169073912224, 0.00020793011231319818, 0.12256566121836052],
                                     [0.00031510712466808344, 1.4627517849594637e-05, 0.19753506269426185],
                                     [0.0009462122086358204, 4.0916994522061656e-06, 0.033800877887952],
                                     [0.000221523179913848, 1.8046578036748165e-06, 0.02493252994353446],
                                     [0.0005077883621933604, 1.881122017025267e-05, 0.01812488763085486],
                                     [0.00012707683933180027, 0.0005275225946085323, 0.024338754772420652],
                                     ]

# # False alarms check
# for k in range(len(stdtrans_normtoab_probthred_combs)):
#     df_k = copy.deepcopy(df_raw)

#     data_processor_k = DataProcess(
#         data=df_k,
#         time_covariates=["week_of_year"],
#         train_split=0.3,
#         validation_split=0.1,
#         output_col=output_col,
#     )
#     _, _, test_data_k, normalized_data = data_processor_k.get_splits()

#     # Define SKF model
#     norm_model = Model(
#         LocalTrend(mu_states=model_dict["mu_states"][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
#         LSTM,
#         Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
#                    phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
#                    mu_states=[model_dict["mu_states"][autoregression_index].item()], 
#                    var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
#     )
#     norm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

#     abnorm_model = Model(
#         LocalAcceleration(mu_states=[model_dict["mu_states"][0].item(), model_dict["mu_states"][1].item(), 0], var_states=[1e-12, 1e-12, 1e-4]),
#         LSTM,
#         Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
#                    phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
#                    mu_states=[model_dict["mu_states"][autoregression_index].item()], 
#                    var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
#     )
#     abnorm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])
#     skf = SKF(
#             norm_model=norm_model,
#             abnorm_model=abnorm_model,
#             std_transition_error=0,     # To be replaced
#             norm_to_abnorm_prob=0,      # To be replaced
#             abnorm_to_norm_prob=1e-1,
#             norm_model_prior_prob=0.99,
#         )
#     skf.std_transition_error = stdtrans_normtoab_probthred_combs[k][0]
#     skf.norm_to_abnorm_prob = stdtrans_normtoab_probthred_combs[k][1]
#     prob_anm_threshold = stdtrans_normtoab_probthred_combs[k][2]

#     skf._initialize_attributes()
#     skf._initialize_model(norm_model, abnorm_model)

#     skf.filter_marginal_prob_history = skf._prob_history()
#     skf._set_same_states_transition_models()
#     skf.initialize_states_history()

#     filter_marginal_abnorm_prob, states = skf.filter(data=normalized_data)

#     p_anm_all = filter_marginal_abnorm_prob

#     if (np.array(p_anm_all) > prob_anm_threshold).any():
#         print(f"False alarm detected for combination {k}: {stdtrans_normtoab_probthred_combs[k]}")
# 1/0

norm_const_std = data_processor.scale_const_std[data_processor.output_col]

# # # Read test data
# df = pd.read_csv("data/prob_eva_syn_time_series/syn_simple_tsgen.csv")
df = pd.read_csv("data/prob_eva_syn_time_series/syn_rsic_simple_ts_gen_lltolt.csv")

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

results_all = []

for m in range(10):
    for n in tqdm(range(len(restored_data)//10)):
        k = m + n * 10

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
        _, _, test_data_k, normalized_data = data_processor_k.get_splits()

        # Define SKF model
        norm_model = Model(
            LocalTrend(mu_states=[0, 0], var_states=[1e-12, 1e-12]),
            LSTM,
            Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                    phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                    mu_states=[model_dict["mu_states"][autoregression_index].item()], 
                    var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
        )
        norm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

        abnorm_model = Model(
            LocalAcceleration(),
            LSTM,
            Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                    phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                    mu_states=[model_dict["mu_states"][autoregression_index].item()], 
                    var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
        )
        abnorm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])
        skf = SKF(
                norm_model=norm_model,
                abnorm_model=abnorm_model,
                std_transition_error=0,     # To be replaced
                norm_to_abnorm_prob=0,      # To be replaced
                abnorm_to_norm_prob=1e-1,
                norm_model_prior_prob=0.99,
            )
        # Random pick one of the std_transition_error_norm_to_abnorm_prob_combs
        stdtrans_normtoab_probthred_comb = random.choice(stdtrans_normtoab_probthred_combs)
        skf.std_transition_error = stdtrans_normtoab_probthred_comb[0]
        skf.norm_to_abnorm_prob = stdtrans_normtoab_probthred_comb[1]
        prob_anm_threshold = stdtrans_normtoab_probthred_comb[2]
        # skf.std_transition_error = std_trans_error_norm_to_abnorm_prob_combs[0][0]
        # skf.norm_to_abnorm_prob = std_trans_error_norm_to_abnorm_prob_combs[0][1]
        skf._initialize_attributes()
        skf._initialize_model(norm_model, abnorm_model)

        skf.filter_marginal_prob_history = skf._prob_history()
        skf._set_same_states_transition_models()
        skf.initialize_states_history()

        filter_marginal_abnorm_prob, states = skf.filter(data=normalized_data)

        time = data_processor_k.get_time(split="all")

        p_anm_all = filter_marginal_abnorm_prob

        all_detection_points = str(np.where(np.array(p_anm_all) > prob_anm_threshold)[0].tolist())
        itv_log = []
        itv_applied_times = []

        # Get baselines for comparison
        # Baseline estimation
        mu_LL_states = states.get_mean(states_type='prior', states_name="level", standardization=True)
        mu_LT_states = states.get_mean(states_type='prior', states_name="trend", standardization=True, scale_const_mean=scale_const_mean[0], scale_const_std=scale_const_std[0])
        # True baselines
        true_LL_baseline = np.zeros(len(df_k))
        true_LT_baseline = np.zeros(len(df_k))
        anm_LL_baseline = np.zeros(len(df_k))
        anm_LT_baseline = np.zeros(len(df_k))
        anm_mag2_perweek = anm_mag2 / 52
        # LL to LT anomaly
        true_LL_baseline[anm_start_index1:] = anm_mag1
        true_LL_baseline[anm_start_index2:] += np.arange(len(true_LL_baseline)-anm_start_index2) * anm_mag2_perweek
        true_LT_baseline[anm_start_index2:] = anm_mag2_perweek

        # # # LL to LL anomaly
        # true_LL_baseline[anm_start_index1:] = anm_mag1
        # true_LL_baseline[anm_start_index2:] += np.ones(len(true_LL_baseline)-anm_start_index2) * anm_mag2

        # # LT to LL anomaly
        # anm_mag1_perweek = anm_mag1 / 52
        # true_LL_baseline[anm_start_index1:] += np.arange(len(true_LL_baseline)-anm_start_index1) * anm_mag1_perweek
        # true_LL_baseline[anm_start_index2:] += anm_mag2
        # true_LT_baseline[anm_start_index1:] += anm_mag1_perweek

        # # LT to LT anomaly
        # anm_mag1_perweek = anm_mag1 / 52
        # anm_mag2_perweek = anm_mag2 / 52
        # true_LL_baseline[anm_start_index1:] += np.arange(len(true_LL_baseline)-anm_start_index1) * anm_mag1_perweek
        # true_LL_baseline[anm_start_index2:] += np.arange(len(true_LL_baseline)-anm_start_index2) * anm_mag2_perweek
        # true_LT_baseline[anm_start_index1:] += anm_mag1_perweek
        # true_LT_baseline[anm_start_index2:] += anm_mag2_perweek

        # Convert the baselines to strings and save to results_all
        true_LL_baseline_str = str(true_LL_baseline.tolist())
        true_LT_baseline_str = str(true_LT_baseline.tolist())
        estimate_LL_baseline_str = str(mu_LL_states.tolist())
        estimate_LT_baseline_str = str(mu_LT_states.tolist())

        results_all.append([anm_mag2, anm_start_index1, anm_start_index2, all_detection_points, itv_log, itv_applied_times, true_LL_baseline_str, true_LT_baseline_str, estimate_LL_baseline_str, estimate_LT_baseline_str])


        if (np.array(p_anm_all) > prob_anm_threshold).any():
            anm_detected_index = np.where(np.array(p_anm_all) > prob_anm_threshold)[0][0]
        else:
            anm_detected_index = len(p_anm_all) - 1

        # #  Plot
        # state_type = "prior"
        # #  Plot states from pretrained model
        # fig = plt.figure(figsize=(10, 8))
        # gs = gridspec.GridSpec(5, 1)
        # ax0 = plt.subplot(gs[0])
        # ax1 = plt.subplot(gs[1])
        # ax2 = plt.subplot(gs[2])
        # ax3 = plt.subplot(gs[3])
        # ax4 = plt.subplot(gs[4])

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
        #     # states=pretrained_model.states,
        #     states=states,
        #     states_type=state_type,
        #     states_to_plot=['level'],
        #     sub_plot=ax0,
        # )
        # # ax0.axvline(x=time[anm_start_index_global], color='r', linestyle='--')
        # # ax0.set_xticklabels([])
        # # ax0.set_title(f"SKF, mse_LL = {mse_LL:.3e}, mse_LT = {mse_LT:.3e}, detection_time = {detection_time}")
        # all_detection_points_list = np.where(np.array(p_anm_all) > prob_anm_threshold)[0].tolist()
        # for detection_point in all_detection_points_list:
        #     ax0.axvline(x=time[int(detection_point)], color='r', linestyle='--')
        #     ax4.axvline(x=time[int(detection_point)], color='r', linestyle='--')
        # ax0.plot(time, true_LL_baseline, color='k', linestyle='--')
        # ax1.plot(time, true_LT_baseline, color='k', linestyle='--')
        # plot_states(
        #     data_processor=data_processor_k,
        #     standardization=True,
        #     states=states,
        #     states_type=state_type,
        #     states_to_plot=['trend'],
        #     sub_plot=ax1,
        # )
        # ax1.set_xticklabels([])
        # # ax1.set_ylim(-0.002, 0.005)

        # plot_states(
        #     data_processor=data_processor_k,
        #     standardization=True,
        #     states=states,
        #     states_type=state_type,
        #     states_to_plot=['lstm'],
        #     sub_plot=ax2,
        # )
        # ax2.set_xticklabels([])
        # plot_states(
        #     data_processor=data_processor_k,
        #     standardization=True,
        #     states=states,
        #     states_type=state_type,
        #     states_to_plot=['autoregression'],
        #     sub_plot=ax3,
        # )
        # ax3.set_xticklabels([])

        # ax4.plot(time, p_anm_all, color='b')
        # ax4.set_ylabel(r'$p_{\mathrm{anm}}$')
        # ax4.set_xlim(ax0.get_xlim())
        # # ax4.axvline(x=time[anm_start_index], color='r', linestyle='--')
        # ax4.set_ylim(-0.05, 1.05)
        # ax4.set_yticks([0, 1])
        # # ax4.set_xticks([time[int(len(time)*1/9)-1], time[int(len(time)*3/9)-1],time[int(len(time)*5/9)-1],time[int(len(time)*7/9)-1],time[int(-1)]])
        # # ax4.set_xticklabels(['2016', '2018', '2020', '2022', '2024'])
        # ax4.set_xlim(ax0.get_xlim())
        # ax4.axvline(x=time[anm_detected_index], color='r', linestyle='--', label='Anomaly start')
        # # for n in np.where(np.array(p_anm_all) > prob_anm_threshold)[0].tolist():
        # #     ax4.axvline(x=time[n], color='r', linestyle='--', label='Anomaly start')

        # plt.show()

# Save the results to a CSV file
results_df = pd.DataFrame(results_all, columns=["anomaly_magnitude", "anomaly_start_index1", "anomaly_start_index2", "anomaly_detected_index", "intervention_log", "intervention_applied_times", "true_LL_baseline", "true_LT_baseline", "estimated_LL_baseline", "estimated_LT_baseline"])
results_df.to_csv("saved_results/prob_eva/syn_simple_ts_results_skf_lltolt.csv", index=False)