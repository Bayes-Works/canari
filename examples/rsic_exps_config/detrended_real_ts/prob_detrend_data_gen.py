import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from canari.component import LocalTrend, LstmNetwork, Autoregression, Periodic
from tqdm import tqdm

# Set numpy seeds
np.random.seed(965)


########### Get the train + validation data from the parent time series ############
########### (keep them the same across all synthetic time series)       ############

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
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()
train_val_data_raw = df_raw["obs"].values[0:data_processor.validation_end]

# Get scale constants
scale_const_mean = copy.deepcopy(data_processor.scale_const_mean)
scale_const_std = copy.deepcopy(data_processor.scale_const_std)

num_test_ts = 10
anm_mag_all = np.concatenate([np.arange(0.1, 2.01, 0.1)])

# Generate timestamps beginning from 2011-02-06  12:00:00 AM, interval one week, and same num_time_steps
time_stamps = [str(x) for x in df_raw.index]

########### Apply anomalies ############
val_end_len = len(train_val_data_raw)

time_series_all = []

first_anm_type = 'll'
second_anm_type = 'lt'

if second_anm_type == 'll':
    anm_mag_all = np.concatenate([np.arange(0.1, 3.01, 0.2)])
    # anm_mag_all = np.concatenate([np.arange(1.1, 3.01, 0.1)])
    print(anm_mag_all)
elif second_anm_type == 'lt':
    # anm_mag_all = np.concatenate([np.arange(0.01, 0.11, 0.02), np.arange(0.1, 1.01, 0.2), np.arange(1.1, 2.01, 0.2)])
    anm_mag_all = np.concatenate([np.arange(0.1, 3.01, 0.2)])
    print(anm_mag_all)

for i, anm_mag in tqdm(enumerate(anm_mag_all)):
    for k in range(num_test_ts):
        # copy the raw obs data
        gen_anm_ts = copy.deepcopy(df_raw["obs"].values)

        # First anomaly
        # time_anomaly1 = np.random.randint(52, 52 * 2) + val_end_len
        time_anomaly1 = np.random.randint(26, 52) + val_end_len
        if first_anm_type == 'll':
            anm1_mag_fixed = 2        # LL anomaly
        elif first_anm_type == 'lt':
            anm1_mag_fixed = 0.5      # LT anomaly

        sign = -1. if np.random.rand() < 0.5 else 1. # Randomly assign positive and negative anomalies
        anm1_mag_fixed *= sign
        if first_anm_type == 'll':
            anm1_mag_unstandardize = anm1_mag_fixed * (scale_const_std[0] + 1e-10) # Unstandardize the anomaly magnitude
        elif first_anm_type == 'lt':
            anm1_mag_perweek = anm1_mag_fixed / 52
            anm1_mag_unstandardize = anm1_mag_perweek * (scale_const_std[0] + 1e-10)  # Unstandardize the anomaly magnitude

        if first_anm_type == 'll':
            anm1_baseline = np.ones(len(gen_anm_ts)) * anm1_mag_unstandardize
        else:
            anm1_baseline = np.arange(len(gen_anm_ts)) * anm1_mag_unstandardize
            anm1_baseline[time_anomaly1:] -= anm1_baseline[time_anomaly1]
        anm1_baseline[:time_anomaly1] = 0
        gen_anm_ts = gen_anm_ts + anm1_baseline

        # Second anomaly
        time_anomaly2 = time_anomaly1 + (len(gen_anm_ts)-time_anomaly1) // 2

        if second_anm_type == 'll':
            # LL anomaly
            anm2_mag = copy.deepcopy(anm_mag)
            sign = -1. if np.random.rand() < 0.5 else 1. # Randomly assign positive and negative anomalies
            anm2_mag *= sign
            anm2_mag_unstandardize = anm2_mag * (scale_const_std[0] + 1e-10)
            anm2_baseline = np.ones(len(gen_anm_ts)) * anm2_mag_unstandardize
            anm2_baseline[:time_anomaly2] = 0
        elif second_anm_type == 'lt':
            # LT anomaly
            anm2_mag = anm_mag/52
            sign = -1. if np.random.rand() < 0.5 else 1. # Randomly assign positive and negative anomalies
            anm2_mag *= sign
            anm2_mag_unstandardize = anm2_mag * (scale_const_std[0] + 1e-10)  # Unstandardize the anomaly magnitude
            anm2_baseline = np.arange(len(gen_anm_ts)) * anm2_mag_unstandardize
            anm2_baseline[time_anomaly2:] -= anm2_baseline[time_anomaly2]
        anm2_baseline[:time_anomaly2] = 0
        gen_anm_ts += anm2_baseline

        values_str = str(list(gen_anm_ts))
        time_series_all.append([values_str, anm1_mag_fixed, time_anomaly1, anm_mag*sign, time_anomaly2])


# Save to CSV
saved_path = "data/prob_eva_syn_time_series/detrend_rsic_simple_ts4_gen_"+first_anm_type+"to"+second_anm_type+".csv"
df_time_series_all = pd.DataFrame(time_series_all, columns=["values", "anomaly1_magnitude", "anomaly_start_index1", "anomaly2_magnitude", "anomaly_start_index2"])

# Add one column 'timestamp': time_stamps, only for the first row
df_time_series_all.insert(0, 'timestamp', [str(list(time_stamps))] + ['']*(df_time_series_all.shape[0]-1))
df_time_series_all.to_csv(saved_path, index=False)

# # # Read data
df = pd.read_csv(saved_path)

restored_data = []
time_stamps = eval(df.iloc[0]["timestamp"], {"nan": float("nan")})
for _, row in df.iterrows():
    values = np.array(eval(row["values"], {"nan": float("nan")}), dtype=float)
    anomaly1_magnitude = float(row["anomaly1_magnitude"])
    anomaly2_magnitude = float(row["anomaly2_magnitude"])
    anomaly_start_index1 = int(row["anomaly_start_index1"])
    anomaly_start_index2 = int(row["anomaly_start_index2"])
    
    restored_data.append((values, anomaly1_magnitude, anomaly2_magnitude, anomaly_start_index1, anomaly_start_index2))

# Plot generated time series
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 1)
ax0 = plt.subplot(gs[0])
# # Randomly plot samples time series
# random_indices = np.random.choice(len(restored_data), size=2, replace=False)
# for j in random_indices:
for j in range(len(restored_data)):
    ax0.plot(time_stamps, restored_data[j][0])
    ax0.axvline(x=restored_data[j][3], color='g', linestyle='--')
    ax0.axvline(x=restored_data[j][4], color='r', linestyle='--')
# ax0.axvline(x=len(self.data_processor.data.values[train_index, self.data_processor.output_col].reshape(-1))+len(self.data_processor.data.values[val_index, self.data_processor.output_col].reshape(-1)), color='r', linestyle='--')
ax0.set_title("Data generation")
# Only show x ticks for every 52 * 4 weeks
ax0.set_xticks(ax0.get_xticks()[::52*4])
plt.show()