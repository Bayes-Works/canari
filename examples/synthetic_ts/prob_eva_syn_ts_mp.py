import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, Autoregression
import copy
from canari import (
    DataProcess,
    plot_data,
)
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
import pickle

from tqdm import tqdm
from src.matrix_profile_functions import past_only_matrix_profile

########################### Calibrate anomaly score ###########################
# # # Read data
# data_file = "./data/benchmark_data/detrended_data/test_11_data_detrended.csv"
# df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df_raw.iloc[:, 0])
# df_raw = df_raw.iloc[:, 1:]
# df_raw.index = time_series
# df_raw.index.name = "date_time"
# df_raw.columns = ["obs"]

# # # Read data
data_file = "./data/toy_time_series/syn_data_complex_phi09.csv"
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
train_data, validation_data, test_data, all_data = data_processor.get_splits()

m = 52
start_index = int(0.4*len(df_raw))
mp, mpi = past_only_matrix_profile(np.array(df_raw["obs"]).flatten().astype("float64"), m, start_idx=start_index, normalize=False)

#  Plot
from matplotlib import gridspec
state_type = "prior"
time = data_processor.get_time(split="all")
#  Plot states from pretrained model
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

plot_data(
    data_processor=data_processor,
    standardization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax0,
)
ax1.plot(time[:len(mp)], mp, label="MP metric", color="C1")
plt.show()

threshold = np.max(mp[np.isfinite(mp)]) * 1.1
print(f"Threshold for anomaly detection: {threshold}")

# # # Read test data
df = pd.read_csv("data/prob_eva_syn_time_series/syn_complex_ts_regen.csv")

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
#     k += 80

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

    start_index = int(0.4*len(df_k))
    mp, mpi = past_only_matrix_profile(np.array(df_k["obs"]).flatten().astype("float64"), m, start_idx=start_index, normalize=False)
    # Set infinite values to NaN
    mp[np.isinf(mp)] = np.nan

    if (mp > threshold).any():
        anm_detected_index = np.where(mp > threshold)[0][0]
    else:
        anm_detected_index = len(mp) - 1

    detection_time = anm_detected_index - anm_start_index_global
    all_detection_points = str(np.where(mp > threshold)[0].tolist())

    results_all.append([anm_mag, anm_start_index_global, all_detection_points, detection_time])

    # # Plot
    # time = data_processor_k.get_time(split="all")
    # fig = plt.figure(figsize=(10, 8))
    # gs = gridspec.GridSpec(2, 1)
    # ax0 = plt.subplot(gs[0])
    # ax1 = plt.subplot(gs[1])
    # plot_data(
    #     data_processor=data_processor_k,
    #     standardization=True,
    #     plot_column=output_col,
    #     validation_label="y",
    #     sub_plot=ax0,
    # )
    # ax0.axvline(x=time[anm_start_index_global], color='r', linestyle='--')
    # ax1.plot(time[:len(mp)], mp, label="MP metric", color="C1")
    # ax1.axvline(x=time[anm_start_index_global], color='r', linestyle='--')
    # all_detection_plot = np.where(mp > threshold)[0].tolist()
    # if len(all_detection_plot) > 0:
    #     for idx in all_detection_plot:
    #         ax1.axvline(x=time[idx], color='g', linestyle='--', alpha=0.3)
    # plt.show()

# Save the results to a CSV file
results_df = pd.DataFrame(results_all, columns=["anomaly_magnitude", "anomaly_start_index", "anomaly_detected_index", "detection_time"])
results_df.to_csv("saved_results/prob_eva/syn_complex_regen_ts_results_mp.csv", index=False)