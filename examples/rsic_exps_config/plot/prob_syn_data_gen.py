import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec, ticker
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

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          'lines.linewidth' : 1,
          }
plt.rcParams.update(params)
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'


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

num_test_ts = 1
# LT anomaly magnitude
anm_mag_all = np.concatenate([np.arange(0.01, 0.11, 0.01), np.arange(0.2, 1.01, 0.1)])
# LL anomaly magnitude
# anm_mag_all = np.concatenate([np.arange(0.1, 2.01, 0.1)])
# anm_mag_all = np.array([1.5])
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
        anm1_mag_fixed = 1        # LL anomaly
        # anm1_mag_fixed = 0.2        # LT anomaly
        

        sign = -1. if np.random.rand() < 0.5 else 1. # Randomly assign positive and negative anomalies
        anm1_mag_fixed *= sign
        # LL anomaly
        anm1_mag_unstandardize = anm1_mag_fixed * (scale_const_std[0] + 1e-10)  # Unstandardize the anomaly magnitude
        # # LT anomaly
        # anm1_mag_perweek = anm1_mag_fixed / 52
        # anm1_mag_unstandardize = anm1_mag_perweek * (scale_const_std[0] + 1e-10)  # Unstandardize the anomaly magnitude

        anm1_baseline = np.ones(num_time_steps) * anm1_mag_unstandardize
        # anm1_baseline = np.arange(num_time_steps) * anm1_mag_unstandardize
        # anm1_baseline[time_anomaly1:] -= anm1_baseline[time_anomaly1]
        anm1_baseline[:time_anomaly1] = 0
        gen_anm_ts = gen_ts[i*num_test_ts + k, :] + anm1_baseline

        # Second anomaly
        time_anomaly2 = time_anomaly1 + 52 * 6

        # LT anomaly
        anm2_mag = anm_mag/52
        sign = -1. if np.random.rand() < 0.5 else 1. # Randomly assign positive and negative anomalies
        anm2_mag *= sign
        anm2_mag_unstandardize = anm2_mag * (scale_const_std[0] + 1e-10)  # Unstandardize the anomaly magnitude
        anm2_baseline = np.arange(num_time_steps) * anm2_mag_unstandardize
        anm2_baseline[time_anomaly2:] -= anm2_baseline[time_anomaly2]
        anm2_baseline[:time_anomaly2] = 0
        gen_anm_ts += anm2_baseline

        # # LL anomaly
        # anm2_mag = copy.deepcopy(anm_mag)
        # sign = -1. if np.random.rand() < 0.5 else 1. # Randomly assign positive and negative anomalies
        # anm2_mag *= sign
        # anm2_mag_unstandardize = anm2_mag * (scale_const_std[0] + 1e-10)
        # anm2_baseline = np.ones(num_time_steps) * anm2_mag_unstandardize
        # anm2_baseline[:time_anomaly2] = 0
        # gen_anm_ts += anm2_baseline

        values_str = str(list(gen_anm_ts))
        time_series_all.append([values_str, anm1_mag_fixed, time_anomaly1, anm_mag*sign, time_anomaly2])


# Save to CSV
# saved_path = "data/prob_eva_syn_time_series/syn_rsic_simple_ts_gen_lttoll.csv"
df_time_series_all = pd.DataFrame(time_series_all, columns=["values", "anomaly1_magnitude", "anomaly_start_index1", "anomaly2_magnitude", "anomaly_start_index2"])
df_time_series_all.insert(0, 'timestamp', [str(list(time_stamps))] + ['']*(df_time_series_all.shape[0]-1))

# # Add one column 'timestamp': time_stamps, only for the first row
# df_time_series_all.insert(0, 'timestamp', [str(list(time_stamps))] + ['']*(df_time_series_all.shape[0]-1))
# df_time_series_all.to_csv(saved_path, index=False)

# # # # Read data
# df = pd.read_csv(saved_path)

# restored_data = []
# time_stamps = eval(df.iloc[0]["timestamp"], {"nan": float("nan")})
# for _, row in df.iterrows():
#     values = np.array(eval(row["values"], {"nan": float("nan")}), dtype=float)
#     anomaly1_magnitude = float(row["anomaly1_magnitude"])
#     anomaly2_magnitude = float(row["anomaly2_magnitude"])
#     anomaly_start_index1 = int(row["anomaly_start_index1"])
#     anomaly_start_index2 = int(row["anomaly_start_index2"])
    
#     restored_data.append((values, anomaly1_magnitude, anomaly2_magnitude, anomaly_start_index1, anomaly_start_index2))

df = df_time_series_all

restored_data = []
time_stamps = eval(df.iloc[0]["timestamp"], {"nan": float("nan")})
for _, row in df.iterrows():
    values = np.array(eval(row["values"], {"nan": float("nan")}), dtype=float)
    anomaly1_magnitude = float(row["anomaly1_magnitude"])
    anomaly2_magnitude = float(row["anomaly2_magnitude"])
    anomaly_start_index1 = int(row["anomaly_start_index1"])
    anomaly_start_index2 = int(row["anomaly_start_index2"])
    
    restored_data.append((values, anomaly1_magnitude, anomaly2_magnitude, anomaly_start_index1, anomaly_start_index2))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(3, 1))
gs = gridspec.GridSpec(1, 1)
ax0 = plt.subplot(gs[0])

for j in range(len(restored_data)):
    x = np.arange(len(time_stamps))  # use indices instead of timestamps
    y = np.array(restored_data[j][0])

    t1 = restored_data[j][3]
    t2 = restored_data[j][4]

    mask_before = x < t1
    mask_between = (x >= t1) & (x <= t2)
    mask_after = x > t2

    ax0.plot(x[mask_before], y[mask_before], 0.1, color='k')
    ax0.plot(x[mask_between], y[mask_between], 0.1, color='tab:blue')
    ax0.plot(x[mask_after], y[mask_after], 0.1, color='tab:orange')
    ax0.set_xlim(0, len(time_stamps))

# Remove all frame, ticks, and labels
ax0.set_frame_on(False)
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_xlabel('')
ax0.set_ylabel('')
plt.tight_layout()
plt.subplots_adjust(left=0.06, top=0.88, bottom=0.15)
plt.savefig('ts_example.png', dpi=500)
plt.show()