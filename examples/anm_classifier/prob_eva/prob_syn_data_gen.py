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
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()
train_val_data_raw = df_raw["obs"].values[0:data_processor.validation_end]

# Get scale constants
scale_const_mean = copy.deepcopy(data_processor.scale_const_mean)
scale_const_std = copy.deepcopy(data_processor.scale_const_std)


########### Generate time series without anomalies ############
# Define SSM
model = Model(
    LocalTrend(
        mu_states=[0, 0],
        var_states=[1e-12, 1e-12],
        std_error=0,
    ),  
    Periodic(period=52, mu_states=[0, 5 * 5], var_states=[1e-12, 1e-12]),
    Periodic(period=13, mu_states=[0, 10], var_states=[1e-12, 1e-12]),
    Autoregression(
        std_error=5, phi=0.5, mu_states=[-0.0621], var_states=[6.36e-05]
    ),
)

num_test_ts = 10
# LL anomaly magnitude
anm_mag_all = np.concatenate([np.arange(0.01, 0.11, 0.01), np.arange(0.2, 1.01, 0.1)])
num_time_steps = 52 * 19
gen_ts, _, _, _ = model.generate_time_series(num_time_series=num_test_ts*len(anm_mag_all),
                                            num_time_steps=num_time_steps)

# Generate timestamps beginning from 2011-02-06  12:00:00 AM, interval one week, and same num_time_steps
start_time = pd.Timestamp('2000-02-06 00:00:00')
# time_stamps = pd.date_range(start=start_time, periods=num_time_steps, freq='W')
# Generate time_stamps with format yyyy-mm-dd hh:mm:ss
time_stamps = pd.date_range(start=start_time, periods=num_time_steps, freq='W').strftime('%Y-%m-%d %H:%M:%S')

########### Apply anomalies ############
val_end_len = len(train_val_data_raw)
for i in range(gen_ts.shape[0]):
    gen_ts[i, :val_end_len] = train_val_data_raw

time_series_all = []
for i, anm_mag in tqdm(enumerate(anm_mag_all)):
    for k in range(num_test_ts):
        # First anomaly
        time_anomaly1 = np.random.randint(52, 52 * 2) + val_end_len
        anm1_mag_fixed = 0.5

        sign = -1. if np.random.rand() < 0.5 else 1. # Randomly assign positive and negative anomalies
        anm1_mag_fixed *= sign
        anm1_mag_unstandardize = anm1_mag_fixed * (scale_const_std[0] + 1e-10)  # Unstandardize the anomaly magnitude

        anm1_baseline = np.ones(num_time_steps) * anm1_mag_unstandardize
        anm1_baseline[:time_anomaly1] = 0
        gen_anm_ts = gen_ts[i*num_test_ts + k, :] + anm1_baseline

        # Second anomaly
        time_anomaly2 = time_anomaly1 + 52 * 6
        anm2_mag = anm_mag

        sign = -1. if np.random.rand() < 0.5 else 1. # Randomly assign positive and negative anomalies
        anm2_mag *= sign
        anm2_mag_unstandardize = anm2_mag * (scale_const_std[0] + 1e-10)  # Unstandardize the anomaly magnitude

        anm2_baseline = np.ones(num_time_steps) * anm2_mag_unstandardize
        anm2_baseline[:time_anomaly2] = 0
        gen_anm_ts += anm2_baseline

        values_str = str(list(gen_anm_ts))
        time_series_all.append([values_str, anm_mag, time_anomaly1, time_anomaly2])


# Save to CSV
saved_path = "data/prob_eva_syn_time_series/syn_rsic_simple_ts_gen.csv"
df_time_series_all = pd.DataFrame(time_series_all, columns=["values", "anomaly_magnitude", "anomaly_start_index1", "anomaly_start_index2"])

# Add one column 'timestamp': time_stamps, only for the first row
df_time_series_all.insert(0, 'timestamp', [str(list(time_stamps))] + ['']*(df_time_series_all.shape[0]-1))
df_time_series_all.to_csv(saved_path, index=False)

# # # Read data
df = pd.read_csv(saved_path)

restored_data = []
time_stamps = eval(df.iloc[0]["timestamp"], {"nan": float("nan")})
for _, row in df.iterrows():
    values = np.array(eval(row["values"], {"nan": float("nan")}), dtype=float)
    anomaly_magnitude = float(row["anomaly_magnitude"])
    anomaly_start_index1 = int(row["anomaly_start_index1"])
    anomaly_start_index2 = int(row["anomaly_start_index2"])
    
    restored_data.append((values, anomaly_magnitude, anomaly_start_index1, anomaly_start_index2))

# Plot generated time series
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 1)
ax0 = plt.subplot(gs[0])
# Randomly plot 5 time series
random_indices = np.random.choice(len(restored_data), size=2, replace=False)
for j in random_indices:
# for j in range(len(restored_data)):
    ax0.plot(time_stamps, restored_data[j][0])
    ax0.axvline(x=restored_data[j][2], color='g', linestyle='--')
    ax0.axvline(x=restored_data[j][3], color='r', linestyle='--')
# ax0.axvline(x=len(self.data_processor.data.values[train_index, self.data_processor.output_col].reshape(-1))+len(self.data_processor.data.values[val_index, self.data_processor.output_col].reshape(-1)), color='r', linestyle='--')
ax0.set_title("Data generation")
# Only show x ticks for every 52 * 4 weeks
ax0.set_xticks(ax0.get_xticks()[::52*4])
plt.show()