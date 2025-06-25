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

# # Read data
data_file = "./data/benchmark_data/test_5_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]

# Data pre-processing
output_col = [0]
train_split=0.289
validation_split=0.0693*2

data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=train_split,
    validation_split=validation_split,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()
validation_start = df_raw.index[data_processor.validation_start]
test_start = df_raw.index[data_processor.test_start]

####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
with open("saved_params/real_ts5_model_rebased.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=19,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

print("phi_AR =", model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item())
print("sigma_AR =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()))


# No detection: 3, 7
std_trans_error_norm_to_abnorm_prob_combs = [[2.338684239873693e-05, 0.0002962674875429233],
                                             [1.8375154282368036e-05, 8.20479620349539e-05],
                                             [2.9738634349295472e-05, 2.3642786293939953e-05],
                                             [0.00016745810435491771, 0.00013015428669050342],
                                             [0.00014488199573833057, 6.354177957238564e-05],
                                             [0.00016885502675486832, 2.9963728999873942e-06],
                                             [0.005250989305736423, 1.1007776480374377e-06],
                                             [0.00015258287393320067, 1.3185932704980428e-06],
                                             [0.00011103072861924886, 8.214369712669897e-06],
                                             [0.0014337531657054882, 2.454952839595189e-05]
                                            ]

norm_const_std = data_processor.std_const_std[data_processor.output_col]

# Load the CSV
df = pd.read_csv("data/prob_eva_syn_time_series/toy_real_ts5gen.csv")
# Containers for restored data
restored_data = []
for _, row in df.iterrows():
    # Convert string to list, then to desired type
    timestamps = pd.to_datetime(ast.literal_eval(row["timestamps"]))
    values = np.array(ast.literal_eval(row["values"]), dtype=float)
    anomaly_magnitude = float(row["anomaly_magnitude"])
    anomaly_start_index = int(row["anomaly_start_index"])
    
    restored_data.append((timestamps, values, anomaly_magnitude, anomaly_start_index))

results_all = []

for k in tqdm(range(len(restored_data))):

    anm_start_index = restored_data[k][3]
    anm_mag = restored_data[k][2]

    anm_start_index_global = anm_start_index + len(df_raw) - len(test_data["y"])

    # gen_time_series = gen_time_series[0]
    gen_time_series = restored_data[k][1]

    # Remove the last len(test_data["y"]) rows in df_raw
    df_raw = df_raw[:-len(test_data["y"])]

    new_df = pd.DataFrame({'values': gen_time_series}, index=restored_data[k][0])
    df_raw = pd.concat([df_raw, new_df])

    # df_raw["values"].iloc[-len(test_data["y"]):] = gen_time_series
    data_processor = DataProcess(
        data=df_raw,
        time_covariates=["week_of_year"],
        validation_start = validation_start,
        test_start = test_start,
        output_col=output_col,
    )
    train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

    # Define SKF model
    norm_model = Model(
        LocalTrend(mu_states=model_dict['early_stop_init_mu_states'][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
        LSTM,
        Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()), 
                    phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(), 
                    mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()], 
                    var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
    )
    norm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

    abnorm_model = Model(
        LocalAcceleration(mu_states=[model_dict['early_stop_init_mu_states'][0].item(), model_dict['early_stop_init_mu_states'][1].item(), 0], var_states=[1e-12, 1e-12, 1e-4]),
        LSTM,
        Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()),
                    phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(),
                    mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()],
                    var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
    )
    abnorm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])
    skf = SKF(
            norm_model=norm_model,
            abnorm_model=abnorm_model,
            std_transition_error=0.0008005024870053932,
            norm_to_abnorm_prob=1.6648750624135252e-06,
            abnorm_to_norm_prob=1e-1,
            norm_model_prior_prob=0.99,
        )
    # Random pick one of the std_transition_error_norm_to_abnorm_prob_combs
    std_trans_error_norm_to_abnorm_prob_comb = random.choice(std_trans_error_norm_to_abnorm_prob_combs)
    skf.std_transition_error = std_trans_error_norm_to_abnorm_prob_comb[0]
    skf.norm_to_abnorm_prob = std_trans_error_norm_to_abnorm_prob_comb[1]
    skf._initialize_attributes()
    skf._initialize_model(norm_model, abnorm_model)

    skf.filter_marginal_prob_history = skf._prob_history()
    skf._set_same_states_transition_models()
    skf.initialize_states_history()

    filter_marginal_abnorm_prob, states = skf.filter(data=normalized_data)

    time = data_processor.get_time(split="all")

    p_anm_all = filter_marginal_abnorm_prob

    all_detection_points = str(np.where(np.array(p_anm_all) > 0.5)[0].tolist())

    if (np.array(p_anm_all) > 0.5).any():
        anm_detected_index = np.where(np.array(p_anm_all) > 0.5)[0][0]
    else:
        anm_detected_index = len(p_anm_all)

    # Get true baseline
    anm_mag_normed = anm_mag / norm_const_std
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
    mu_LL_states = states.get_mean(states_type='prior', states_name=["level"])["level"]
    mu_LT_states = states.get_mean(states_type='prior', states_name=["trend"])["trend"]
    mse_LL = metric.mse(
        mu_LL_states[anm_start_index_global+1:],
        LL_baseline_true[anm_start_index_global+1:],
    )
    mse_LT = metric.mse(
        mu_LT_states[anm_start_index_global+1:],
        LT_baseline_true[anm_start_index_global+1:],
    )
    # # Compute MAPE for LL and LT
    # mape_LL = metric.mape(
    #     mu_LL_states[anm_start_index_global+1:],
    #     LL_baseline_true[anm_start_index_global+1:],
    # )
    # mape_LT = metric.mape(
    #     mu_LT_states[anm_start_index_global+1:],
    #     LT_baseline_true[anm_start_index_global+1:],
    # )
    mape_LL = None
    mape_LT = None

    detection_time = anm_detected_index - anm_start_index_global

    results_all.append([anm_mag, anm_start_index_global, all_detection_points, mse_LL, mse_LT, mape_LL, mape_LT, detection_time])

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
    #     data_processor=data_processor,
    #     standardization=True,
    #     states=states,
    #     states_type=state_type,
    #     states_to_plot=['trend'],
    #     sub_plot=ax1,
    # )
    # ax1.set_xticklabels([])
    # ax1.set_ylim(-0.002, 0.005)

    # plot_states(
    #     data_processor=data_processor,
    #     standardization=True,
    #     states=states,
    #     states_type=state_type,
    #     states_to_plot=['lstm'],
    #     sub_plot=ax2,
    # )
    # ax2.set_xticklabels([])
    # plot_states(
    #     data_processor=data_processor,
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

    # plt.show()

# Save the results to a CSV file
results_df = pd.DataFrame(results_all, columns=["anomaly_magnitude", "anomaly_start_index", "anomaly_detected_index", "mse_LL", "mse_LT", "mape_LL", "mape_LT", "detection_time"])
results_df.to_csv("saved_results/prob_eva/real_ts5_results_skf.csv", index=False)