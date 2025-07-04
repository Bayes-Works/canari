import fire
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytagi import metric
from pytagi import Normalizer
from canari import (
    DataProcess,
    Model,
    ModelOptimizer,
    SKF,
    SKFOptimizer,
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise
import pickle
from tqdm import tqdm

# Parameter optimization for SKF
def initialize_skf(skf_param_space, model_param: dict):
    norm_model = Model.load_dict(model_param)
    abnorm_model = Model(
        LocalAcceleration(),
        LstmNetwork(),
        WhiteNoise(),
    )
    skf = SKF(
        norm_model=norm_model,
        abnorm_model=abnorm_model,
        std_transition_error=skf_param_space["std_transition_error"],
        norm_to_abnorm_prob=skf_param_space["norm_to_abnorm_prob"],
    )
    return skf

# Read data
data_file = "./data/benchmark_data/detrended_data/test_4_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# # Add synthetic anomaly to data
# # LT anomaly
# # anm_mag = 0.010416667/10
# anm_start_index = 52*8
# anm_mag = 0.2/52
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
    train_split=0.3,
    validation_split=0.1,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

num_models = 10

# Tune the threshold of p_anm again to make sure no triggers through the whole time series
for k in range(1, 1+num_models):
    print('Tuning threshold for model', k)
    # Load model_dict from local
    with open(f"saved_params/ts4_whitenoise_models/skf_param{k}.pkl", "rb") as f:
        skf_param = pickle.load(f)
    with open(f"saved_params/ts4_whitenoise_models/model_optim_dict{k}.pkl", "rb") as f:
        model_optim_dict = pickle.load(f)

    skf_optim = initialize_skf(skf_param, model_optim_dict)
    prob_anm_threshold = skf_param["threshold_anm_prob"]
    print(prob_anm_threshold)

    p_anm_all, states = skf_optim.filter(data=all_data)

    while (np.array(p_anm_all) > prob_anm_threshold).any():
        skf_param["threshold_anm_prob"] += 0.1
        prob_anm_threshold = skf_param["threshold_anm_prob"]
        print(prob_anm_threshold)

        skf_optim = initialize_skf(skf_param, model_optim_dict)
        p_anm_all, states = skf_optim.filter(data=all_data)

    if prob_anm_threshold > 1:
        print(f"Model {k} has a threshold of {prob_anm_threshold} > 1. Remove this model!")

    # Save the re-tuned skf_param
    with open(f"saved_params/ts4_whitenoise_models/skf_param_retuned{k}.pkl", "wb") as f:
        pickle.dump(skf_param, f)

    print('-----------------------------------------------------')


# # # Read test data
df = pd.read_csv("data/prob_eva_syn_time_series/detrended_ts4_tsgen.csv")

# Containers for restored data
restored_data = []
for _, row in df.iterrows():
    values = np.array(eval(row["values"], {"nan": float("nan")}), dtype=float)
    anomaly_magnitude = float(row["anomaly_magnitude"])
    anomaly_start_index = int(row["anomaly_start_index"])
    
    restored_data.append((values, anomaly_magnitude, anomaly_start_index))

results_all = []

for k in tqdm(range(len(restored_data))):
# for k in tqdm(range(10)):
#     k += 120

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

    model_index = np.random.randint(1, 1+num_models)
    # Load model_dict from local, note that the skf_param need to use the re-tuned one
    with open(f"saved_params/ts4_whitenoise_models/skf_param_retuned{model_index}.pkl", "rb") as f:
        skf_param = pickle.load(f)
    with open(f"saved_params/ts4_whitenoise_models/model_optim_dict{model_index}.pkl", "rb") as f:
        model_optim_dict = pickle.load(f)

    skf_optim = initialize_skf(skf_param, model_optim_dict)
    prob_anm_threshold = skf_param["threshold_anm_prob"]
    init_level = copy.deepcopy(model_optim_dict["mu_states"][0].item())

    # Detect anomaly
    filter_marginal_abnorm_prob, states = skf_optim.filter(data=normalized_data)

    time = data_processor_k.get_time(split="all")

    p_anm_all = filter_marginal_abnorm_prob

    all_detection_points = str(np.where(np.array(p_anm_all) > prob_anm_threshold)[0].tolist())

    if (np.array(p_anm_all) > prob_anm_threshold).any():
        anm_detected_index = np.where(np.array(p_anm_all) > prob_anm_threshold)[0][0]
    else:
        anm_detected_index = len(p_anm_all) - 1

    # Get true baseline
    # anm_mag_normed = anm_mag / norm_const_std
    anm_mag_normed = anm_mag
    LL_baseline_true = np.zeros_like(df_raw)
    LT_baseline_true = np.zeros_like(df_raw)
    for i in range(1, len(df_raw)):
        if i > anm_start_index_global:
            LL_baseline_true[i] += anm_mag_normed * (i - anm_start_index_global)
            LT_baseline_true[i] += anm_mag_normed

    LL_baseline_true += init_level
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


    # # Plotting SKF results
    # fig, ax = plot_skf_states(
    #     data_processor=data_processor_k,
    #     states=states,
    #     states_type="smooth",
    #     states_to_plot=["level", "trend", "lstm", "white noise"],
    #     model_prob=filter_marginal_abnorm_prob,
    #     standardization=False,
    # )
    # ax[0].axvline(
    #     x=data_processor.data.index[anm_start_index_global],
    #     color="r",
    #     linestyle="--",
    # )

    # ax[4].axvline(
    #     x=data_processor.data.index[anm_detected_index],
    #     color="r",
    #     linestyle="--",
    # )
    # fig.suptitle(f"IL, mse_LL = {mse_LL:.3e}, mse_LT = {mse_LT:.3e}, detection_time = {detection_time}")
    # plt.show()

# Save the results to a CSV file
results_df = pd.DataFrame(results_all, columns=["anomaly_magnitude", "anomaly_start_index", "anomaly_detected_index", "mse_LL", "mse_LT",  "detection_time"])
results_df.to_csv("saved_results/prob_eva/detrended_ts4_whitenoise_skf.csv", index=False)
