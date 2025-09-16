import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from canari import (
    DataProcess,
)
from matplotlib import gridspec
import matplotlib.dates as mdates

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          'lines.linewidth' : 1,
          }
plt.rcParams.update(params)

####################################### Data 1 #######################################
data_file = "./data/benchmark_data/detrended_data/test_1_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]
datetime1 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0.,
    output_col=output_col,
)

_, _, _, data1 = data_processor.get_splits()

####################################### Data 2 #######################################
data_file = "./data/benchmark_data/detrended_data/test_2_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]
datetime2 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0.,
    output_col=output_col,
)

_, _, _, data2 = data_processor.get_splits()

####################################### Data 3 #######################################
data_file = "./data/benchmark_data/detrended_data/test_11_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]
datetime3 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0.,
    output_col=output_col,
)

_, _, _, data3 = data_processor.get_splits()

####################################### Data 4 #######################################
data_file = "./data/benchmark_data/detrended_data/test_4_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]
datetime4 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0.,
    output_col=output_col,
)
_, _, _, data4 = data_processor.get_splits()

####################################### Data 5 #######################################
data_file = "./data/benchmark_data/detrended_data/test_5_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]
datetime5 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0.,
    output_col=output_col,
)
_, _, _, data5 = data_processor.get_splits()

####################################### Data 6 #######################################
data_file = "./data/benchmark_data/detrended_data/test_6_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]
datetime6 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0.,
    output_col=output_col,
)
_, _, _, data6 = data_processor.get_splits()

####################################### Data 7 #######################################
data_file = "./data/benchmark_data/detrended_data/test_7_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]
datetime7 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0.,
    output_col=output_col,
)
_, _, _, data7 = data_processor.get_splits()

####################################### Data 8 #######################################
data_file = "./data/benchmark_data/detrended_data/test_8_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]
datetime8 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0.,
    output_col=output_col,
)
_, _, _, data8 = data_processor.get_splits()

####################################### Data 9 #######################################
data_file = "./data/benchmark_data/detrended_data/test_9_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]
datetime9 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0.,
    output_col=output_col,
)
_, _, _, data9 = data_processor.get_splits()

####################################### Data 10 #######################################
data_file = "./data/benchmark_data/detrended_data/test_10_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]
datetime10 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0.,
    output_col=output_col,
)
_, _, _, data10 = data_processor.get_splits()

# Plot real time series
fig = plt.figure(figsize=(8.7, 2.2), constrained_layout=True)
gs = gridspec.GridSpec(2, 5)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
ax4 = plt.subplot(gs[4])
ax5 = plt.subplot(gs[5])
ax6 = plt.subplot(gs[6])
ax7 = plt.subplot(gs[7])
ax8 = plt.subplot(gs[8])
ax9 = plt.subplot(gs[9])

ymin = -2.
ymax = 2.
yticks = [-2, 0, 2]

ax0.plot(datetime1, data1["y"], label="Observed", color='C0')
ax0.xaxis.set_major_locator(mdates.YearLocator(base=4, month=12))
ax0.set_ylim([ymin, ymax])
ax0.set_yticks(yticks)

ax1.plot(datetime2, data2["y"], label="Observed", color='C0')
ax1.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax1.set_ylim([ymin, ymax])
ax1.set_yticks(yticks)
ax1.set_yticklabels([])

ax2.plot(datetime3, data3["y"], label="Observed", color='C0')
ax2.xaxis.set_major_locator(mdates.YearLocator(base=6))
ax2.set_ylim([ymin, ymax])
ax2.set_yticks(yticks)
ax2.set_yticklabels([])

ax3.plot(datetime4, data4["y"], label="Observed", color='C0')
ax3.xaxis.set_major_locator(mdates.YearLocator(base=4))
ax3.set_ylim([ymin, ymax])
ax3.set_yticks(yticks)
ax3.set_yticklabels([])

ax4.plot(datetime5, data5["y"], label="Observed", color='C0')
ax4.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax4.set_ylim([ymin, ymax])
ax4.set_yticks(yticks)
ax4.set_yticklabels([])

ax5.plot(datetime6, data6["y"], label="Observed", color='C0')
ax5.xaxis.set_major_locator(mdates.YearLocator(base=4))
ax5.set_ylim([ymin, ymax])
ax5.set_yticks(yticks)

ax6.plot(datetime7, data7["y"], label="Observed", color='C0')
ax6.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax6.set_ylim([ymin, ymax])
ax6.set_yticks(yticks)
ax6.set_yticklabels([])

ax7.plot(datetime8, data8["y"], label="Observed", color='C0')
ax7.xaxis.set_major_locator(mdates.YearLocator(base=4))
ax7.set_ylim([ymin, ymax])
ax7.set_yticks(yticks)
ax7.set_yticklabels([])

ax8.plot(datetime9, data9["y"], label="Observed", color='C0')
ax8.xaxis.set_major_locator(mdates.YearLocator(base=4, month=12))
ax8.set_ylim([ymin, ymax])
ax8.set_yticks(yticks)
ax8.set_yticklabels([])

ax9.plot(datetime10, data10["y"], label="Observed", color='C0')
ax9.xaxis.set_major_locator(mdates.YearLocator(base=4))
ax9.set_ylim([ymin, ymax])
ax9.set_yticks(yticks)
ax9.set_yticklabels([])

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.6, wspace=0.15)
plt.savefig('detrend_ts_norm_demo.png', dpi=300)
plt.show()