import copy
import pandas as pd
from pytagi import Normalizer as normalizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytagi.metric as metric
from src.matrix_profile_functions import past_only_matrix_profile
from canari import DataProcess, plot_data

# # # Read data
data_file = "./data/benchmark_data/detrended_data/test_5_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

# LT anomaly
# anm_mag = 0.010416667/10
anm_start_index = 52*8
anm_mag = 0.1/52
# anm_mag = 0.

# anm_baseline = np.linspace(0, 3, num=len(df_raw))
anm_baseline = np.arange(len(df_raw)) * anm_mag
# Set the first 52*12 values in anm_baseline to be 0
anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
anm_baseline[:anm_start_index] = 0
df_raw = df_raw.add(anm_baseline, axis=0)

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

# Example usage
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
ax0.axvline(x=time[anm_start_index], color='r', linestyle='--')
ax1.plot(time[:len(mp)], mp, label="MP metric", color="C1")
ax1.axvline(x=time[anm_start_index], color='r', linestyle='--')
plt.show()
