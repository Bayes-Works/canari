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
data_file = "./data/benchmark_data/test_2_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=";", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 4])
df_raw = df_raw.iloc[:, 6].to_frame()
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]
df_raw = df_raw.resample("W").mean()
df_raw = df_raw.iloc[30:, :]

# Data pre-processing
output_col = [0]
train_split = 0.25
validation_split = 0.08
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=train_split,
    validation_split=validation_split,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

m = 52
start_index = int((train_split)*len(df_raw))
mp, mpi = past_only_matrix_profile(np.array(df_raw["obs"][:int((train_split+validation_split)*len(df_raw))]).flatten().astype("float64"), m, start_idx=start_index, normalize=True)

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

# Run on the whole time series
start_index = int((train_split)*len(df_raw))
mp, mpi = past_only_matrix_profile(np.array(df_raw["obs"]).flatten().astype("float64"), m, start_idx=start_index, normalize=True)
# Set infinite values to NaN
mp[np.isinf(mp)] = np.nan

all_detection_points = str(np.where(mp > threshold)[0].tolist())

# Plot
time = data_processor.get_time(split="all")
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
all_detection_plot = np.where(mp > threshold)[0].tolist()
if len(all_detection_plot) > 0:
    for idx in all_detection_plot:
        ax0.axvline(x=time[idx], color='g', linestyle='--', alpha=0.3)
        ax1.axvline(x=time[idx], color='g', linestyle='--', alpha=0.3)
plt.show()
