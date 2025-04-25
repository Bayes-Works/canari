import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from src import (
    LocalLevel,
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    Periodic,
    Autoregression,
    WhiteNoise,
    Model,
    SKF,
    plot_data,
    plot_prediction,
    plot_states,
)
from src.hsl_detection import hsl_detection
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
import pickle
import src.common as common

import ast
from tqdm import tqdm
import random

# # Read data
data_file = "./data/toy_time_series/synthetic_simple_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

anm_start_index = 52*10

# LT anomaly
# anm_mag = 0.010416667/10
anm_mag = 0.3/52
# anm_baseline = np.linspace(0, 3, num=len(df_raw))
anm_baseline = np.arange(len(df_raw)) * anm_mag
# Set the first 52*12 values in anm_baseline to be 0
anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
anm_baseline[:anm_start_index] = 0

# # LL anomaly
# anm_mag = 0.5
# anm_baseline = np.zeros_like(df_raw)
# anm_baseline[anm_start_index:] += anm_mag

df_raw = df_raw.add(anm_baseline, axis=0)

data_file_time = "./data/toy_time_series/synthetic_simple_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Data pre-processing
output_col = [0]
train_split=0.289
validation_split=0.0693*2

# Remove the last 52*5 rows in df_raw
train_split = train_split * len(df_raw) / len(df_raw[:-52*5])
validation_split = validation_split * len(df_raw) / len(df_raw[:-52*5])
df_raw = df_raw[:-52*5]

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
with open("saved_params/toy_simple_model.pkl", "rb") as f:
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

# No detection: 3, 7
std_trans_error_norm_to_abnorm_prob_combs = [[0.0003123022418182095, 1.2129283607371455e-06],
                                            [5.6033733839347894e-05, 1.7559948785393028e-06],
                                            [6.564963793526337e-05, 7.712919602901581e-05],
                                            [0.00032847278347743723, 3.9947717600623924e-05],
                                            [0.0004800598353304185, 3.3254051926052085e-05],
                                            [3.400413281349548e-05, 6.282990924767272e-05],
                                            [0.0012194568427483132, 1.2514669352313052e-05],
                                            [0.0009316055756732739, 3.5848341660084674e-06],
                                            [0.00017739592835080213, 0.0013276774530535473],
                                            [0.0008005024870053932, 1.6648750624135252e-06],
                                            ]

skf = SKF(
    norm_model=norm_model,
    abnorm_model=abnorm_model,
    std_transition_error=0.0008005024870053932,
    norm_to_abnorm_prob=1.6648750624135252e-06,
    abnorm_to_norm_prob=1e-1,
    norm_model_prior_prob=0.99,
)

# # Anomaly Detection
# filter_marginal_abnorm_prob, states = skf.filter(data=normalized_data)
# smooth_marginal_abnorm_prob, states = skf.smoother(data=normalized_data)

skf.filter_marginal_prob_history = skf.prob_history()
skf.set_same_states_transition_models()
skf.initialize_states_history()

filter_marginal_abnorm_prob, states = skf.filter(data=train_data)
filter_marginal_abnorm_prob, states = skf.filter(data=validation_data)

states_temp = copy.deepcopy(states)
abnorm_prob_temp = copy.deepcopy(skf.filter_marginal_prob_history)
marginal_prob_current_temp = copy.deepcopy(skf.marginal_prob_current)
lstm_net_temp = copy.deepcopy(skf.model["norm_norm"].lstm_net.get_lstm_states())

norm_const_std = data_processor.norm_const_std[data_processor.output_col]

# Load the CSV
df = pd.read_csv("data/prob_eva_syn_time_series/toy_simple_tsgen.csv")
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

    # Random pick one of the std_transition_error_norm_to_abnorm_prob_combs
    std_trans_error_norm_to_abnorm_prob_comb = random.choice(std_trans_error_norm_to_abnorm_prob_combs)
    skf.std_transition_error = std_trans_error_norm_to_abnorm_prob_comb[0]
    skf.norm_to_abnorm_prob = std_trans_error_norm_to_abnorm_prob_comb[1]

    skf.states = copy.deepcopy(states_temp)
    skf.filter_marginal_prob_history = copy.deepcopy(abnorm_prob_temp)
    skf.marginal_prob_current = copy.deepcopy(marginal_prob_current_temp)
    current_time_step = len(train_data["y"]) + len(validation_data["y"])
    skf.model["norm_norm"].set_memory(states=skf.model["norm_norm"].states, time_step=current_time_step)
    skf.model["abnorm_abnorm"].set_memory(states=skf.model["abnorm_abnorm"].states, time_step=current_time_step)
    skf.model["abnorm_norm"].set_memory(states=skf.model["abnorm_norm"].states, time_step=current_time_step)
    skf.model["norm_abnorm"].set_memory(states=skf.model["norm_abnorm"].states, time_step=current_time_step)
    skf.model["norm_norm"].lstm_net.set_lstm_states(lstm_net_temp)

    filter_marginal_abnorm_prob, states = skf.filter(data=test_data)

    from src.data_visualization import determine_time
    time = determine_time(data_processor, len(normalized_data["y"]))

    p_anm_all = filter_marginal_abnorm_prob

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
    mu_LL_states = states.get_mean(states_type='prior', states_name=["local level"])["local level"]
    mu_LT_states = states.get_mean(states_type='prior', states_name=["local trend"])["local trend"]
    mse_LL = metric.mse(
        mu_LL_states[anm_start_index_global:],
        LL_baseline_true[anm_start_index_global:],
    )
    mse_LT = metric.mse(
        mu_LT_states[anm_start_index_global:],
        LT_baseline_true[anm_start_index_global:],
    )
    detection_time = anm_detected_index - anm_start_index_global

    results_all.append([anm_mag, anm_start_index_global, anm_detected_index, mse_LL, mse_LT, detection_time])

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
    #     normalization=True,
    #     plot_column=output_col,
    #     validation_label="y",
    #     sub_plot=ax0,
    # )
    # plot_states(
    #     data_processor=data_processor,
    #     normalization=True,
    #     # states=pretrained_model.states,
    #     states=states,
    #     states_type=state_type,
    #     states_to_plot=['local level'],
    #     sub_plot=ax0,
    # )
    # ax0.axvline(x=time[anm_start_index_global], color='r', linestyle='--')
    # ax0.set_xticklabels([])
    # ax0.set_title(f"SKF, mse_LL = {mse_LL:.3e}, mse_LT = {mse_LT:.3e}, detection_time = {detection_time}")
    # ax0.plot(time, LL_baseline_true, color='k', linestyle='--')
    # ax1.plot(time, LT_baseline_true, color='k', linestyle='--')
    # plot_states(
    #     data_processor=data_processor,
    #     normalization=True,
    #     states=states,
    #     states_type=state_type,
    #     states_to_plot=['local trend'],
    #     sub_plot=ax1,
    # )
    # ax1.set_xticklabels([])

    # plot_states(
    #     data_processor=data_processor,
    #     normalization=True,
    #     states=states,
    #     states_type=state_type,
    #     states_to_plot=['lstm'],
    #     sub_plot=ax2,
    # )
    # ax2.set_xticklabels([])
    # plot_states(
    #     data_processor=data_processor,
    #     normalization=True,
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

# # Save the results to a CSV file
# results_df = pd.DataFrame(results_all, columns=["anomaly_magnitude", "anomaly_start_index", "anomaly_detected_index", "mse_LL", "mse_LT", "detection_time"])
# results_df.to_csv("saved_results/prob_eva/toy_simple_results_skf.csv", index=False)