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
data_file = "./data/benchmark_data/detrended_data/test_5_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

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
validation_start = df_raw.index[data_processor.validation_start]
test_start = df_raw.index[data_processor.test_start]

####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
with open("saved_params/real_ts5_tsmodel_detrended.pkl", "rb") as f:
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

stdtrans_normtoab_probthred_combs = [
                                     [4.340903491877591e-05, 3.671739603343575e-05, 0.11329189839252522],
                                     [6.473287303640981e-05, 4.614167194485501e-05, 0.09299104724138803],
                                     [0.00026796992585526014, 6.027770448488271e-06, 0.051412275498759624],
                                     [0.00018053791740468424, 3.739378363196771e-06, 0.05851662867119357],
                                     [0.001063915446121119, 1.4206283433387677e-06, 0.57867948853488685],
                                     [8.154575994462701e-05, 0.00029158245388583536, 0.024216502833571716],
                                     [0.0006324923533305021, 4.5945292824902795e-06, 0.411312430270560824],
                                     [9.104845894631987e-05, 9.815230725761424e-05, 0.03547701899139441],
                                     [0.0002981362182249163, 0.0008466062189607952, 0.51314359921126627],
                                     [0.00021924602792182794, 1.1127650680361712e-05, 0.02069157909740321],
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
df = pd.read_csv("data/prob_eva_syn_time_series/detrended_ts5_tsgen.csv")

# Containers for restored data
restored_data = []
for _, row in df.iterrows():
    values = np.array(eval(row["values"], {"nan": float("nan")}), dtype=float)
    anomaly_magnitude = float(row["anomaly_magnitude"])
    anomaly_start_index = int(row["anomaly_start_index"])
    
    restored_data.append((values, anomaly_magnitude, anomaly_start_index))

results_all = []

for k in tqdm(range(len(restored_data))):
# for k in tqdm(range(2)):
#     k += 150

    df_k = copy.deepcopy(df_raw)
    # Replace the values in the dataframe with the restored_data[k][0]
    df_k.iloc[:, 0] = restored_data[k][0]

    data_processor_k = DataProcess(
        data=df_k,
        time_covariates=["week_of_year"],
        train_split=0.3,
        validation_split=0.1,
        output_col=output_col,
    )
    _, _, test_data_k, normalized_data = data_processor_k.get_splits()

    anm_start_index = restored_data[k][2]
    anm_mag = restored_data[k][1]
    anm_start_index_global = anm_start_index + len(df_k) - len(test_data_k["y"])

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

    if (np.array(p_anm_all) > prob_anm_threshold).any():
        anm_detected_index = np.where(np.array(p_anm_all) > prob_anm_threshold)[0][0]
    else:
        anm_detected_index = len(p_anm_all) - 1

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
    mu_LL_states = states.get_mean(states_type='prior', states_name="level")
    mu_LT_states = states.get_mean(states_type='prior', states_name="trend")
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
    # ax0.axvline(x=time[anm_start_index_global], color='r', linestyle='--')
    # ax0.set_xticklabels([])
    # ax0.set_title(f"SKF, mse_LL = {mse_LL:.3e}, mse_LT = {mse_LT:.3e}, detection_time = {detection_time}")
    # ax0.plot(time, LL_baseline_true, color='k', linestyle='--')
    # ax1.plot(time, LT_baseline_true, color='k', linestyle='--')
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

# Save the results to a CSV file
results_df = pd.DataFrame(results_all, columns=["anomaly_magnitude", "anomaly_start_index", "anomaly_detected_index", "mse_LL", "mse_LT", "detection_time"])
results_df.to_csv("saved_results/prob_eva/detrended_ts5_results_skf.csv", index=False)