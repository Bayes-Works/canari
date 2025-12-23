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
data_file = "./data/benchmark_data/test_1_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
datetime1 = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = datetime1
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

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
data_file = "./data/benchmark_data/test_2_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=";", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 4])
df_raw = df_raw.iloc[:, 6].to_frame()
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]
df_raw = df_raw.resample("W").mean()
df_raw = df_raw.iloc[30:, :]
datetime2 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0,
    output_col=output_col,
)
_, _, _, data2 = data_processor.get_splits()

####################################### Data 3 #######################################
data_file = "./data/benchmark_data/test_11_data.csv"
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
    validation_split=0,
    output_col=output_col,
)
_, _, _, data3 = data_processor.get_splits()

####################################### Data 4 #######################################
data_file = "./data/benchmark_data/test_4_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]
datetime4 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0,
    output_col=output_col,
)
_, _, _, data4 = data_processor.get_splits()

####################################### Data 5 #######################################
data_file = "./data/benchmark_data/test_5_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]
datetime5 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0,
    output_col=output_col,
)
_, _, _, data5 = data_processor.get_splits()

####################################### Data 6 #######################################
data_file = "./data/benchmark_data/test_6_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
# Remove the first 52 rows
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]
df_raw = df_raw.iloc[52:, :]
datetime6 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0,
    output_col=output_col,
)
_, _, _, data6 = data_processor.get_splits()

####################################### Data 7 #######################################
data_file = "./data/benchmark_data/test_7_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]
datetime7 = df_raw.index

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=1,
    validation_split=0,
    output_col=output_col,
)
_, _, _, data7 = data_processor.get_splits()

####################################### Data 8 #######################################
data_file = "./data/benchmark_data/test_8_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
datetime8 = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = datetime8
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]

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
data_file = "./data/benchmark_data/test_9_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]
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
data_file = "./data/benchmark_data/test_10_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]
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

####################################### Plot shaded training and validation regions #######################################
training_splits = [0.25, 0.25, 0.25, 0.23, 0.29, 0.29, 0.29, 0.30, 0.35, 0.28]
validation_splits = [0.07, 0.08, 0.10, 0.07, 0.07, 0.07, 0.07, 0.07, 0.09, 0.05]

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

ymin = -2.5
ymax = 2.5
yticks = [-2, 0, 2]

ax0.plot(datetime3, data3["y"], label="Observed", color='C0')
ax0.xaxis.set_major_locator(mdates.YearLocator(base=6))
ax0.set_ylim([ymin, ymax])
ax0.set_yticks(yticks)
ax0.axvspan(datetime3[0],
            datetime3[int(len(data3["y"])*(training_splits[2]))],
            color="red",
            alpha=0.1,
            edgecolor=None,
        )
ax0.axvspan(datetime3[int(len(data3["y"])*training_splits[2]+1)],
            datetime3[int(len(data3["y"])*(training_splits[2]+validation_splits[2]))],
            color="green",
            alpha=0.1,
            edgecolor=None,
        )

ax1.plot(datetime1, data1["y"], label="Observed", color='C0')
ax1.xaxis.set_major_locator(mdates.YearLocator(base=4, month=12))
ax1.set_ylim([ymin, ymax])
ax1.set_yticks(yticks)
ax1.set_yticklabels([])
ax1.axvspan(datetime1[0],
            datetime1[int(len(data1["y"])*(training_splits[0]))],
            color="red",
            alpha=0.1,
            edgecolor=None,
        )
ax1.axvspan(datetime1[int(len(data1["y"])*training_splits[0]+1)],
            datetime1[int(len(data1["y"])*(training_splits[0]+validation_splits[0]))],
            color="green",
            alpha=0.1,
            edgecolor=None,
        )

ax2.plot(datetime2, data2["y"], label="Observed", color='C0')
ax2.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax2.set_ylim([ymin, ymax])
ax2.set_yticks(yticks)
ax2.set_yticklabels([])
ax2.axvspan(datetime2[0],
            datetime2[int(len(data2["y"])*(training_splits[1]))],
            color="red",
            alpha=0.1,
            edgecolor=None,
        )
ax2.axvspan(datetime2[int(len(data2["y"])*training_splits[1]+1)],
            datetime2[int(len(data2["y"])*(training_splits[1]+validation_splits[1]))],
            color="green",
            alpha=0.1,
            edgecolor=None,
        )

ax3.plot(datetime4, data4["y"], label="Observed", color='C0')
ax3.xaxis.set_major_locator(mdates.YearLocator(base=4))
ax3.set_ylim([ymin, ymax])
ax3.set_yticks(yticks)
ax3.set_yticklabels([])
ax3.axvspan(datetime4[0],
            datetime4[int(len(data4["y"])*(training_splits[3]))],
            color="red",
            alpha=0.1,
            edgecolor=None,
        )
ax3.axvspan(datetime4[int(len(data4["y"])*training_splits[3]+1)],
            datetime4[int(len(data4["y"])*(training_splits[3]+validation_splits[3]))],
            color="green",
            alpha=0.1,
            edgecolor=None,
        )

ax4.plot(datetime5, data5["y"], label="Observed", color='C0')
ax4.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax4.set_ylim([ymin, ymax])
ax4.set_yticks(yticks)
ax4.set_yticklabels([])
ax4.axvspan(datetime5[0],
            datetime5[int(len(data5["y"])*(training_splits[4]))],
            color="red",
            alpha=0.1,
            edgecolor=None,
        )
ax4.axvspan(datetime5[int(len(data5["y"])*training_splits[4]+1)],
            datetime5[int(len(data5["y"])*(training_splits[4]+validation_splits[4]))],
            color="green",
            alpha=0.1,
            edgecolor=None,
        )

ax5.plot(datetime6, data6["y"], label="Observed", color='C0')
ax5.xaxis.set_major_locator(mdates.YearLocator(base=4))
ax5.set_ylim([ymin, ymax])
ax5.set_yticks(yticks)
ax5.axvspan(datetime6[0],
            datetime6[int(len(data6["y"])*(training_splits[5]))],
            color="red",
            alpha=0.1,
            edgecolor=None,
        )
ax5.axvspan(datetime6[int(len(data6["y"])*training_splits[5]+1)],
            datetime6[int(len(data6["y"])*(training_splits[5]+validation_splits[5]))],
            color="green",
            alpha=0.1,
            edgecolor=None,
        )

ax6.plot(datetime7, data7["y"], label="Observed", color='C0')
ax6.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax6.set_ylim([ymin, ymax])
ax6.set_yticks(yticks)
ax6.set_yticklabels([])
ax6.axvspan(datetime7[0],
            datetime7[int(len(data7["y"])*(training_splits[6]))],
            color="red",
            alpha=0.1,
            edgecolor=None,
        )
ax6.axvspan(datetime7[int(len(data7["y"])*training_splits[6]+1)],
            datetime7[int(len(data7["y"])*(training_splits[6]+validation_splits[6]))],
            color="green",
            alpha=0.1,
            edgecolor=None,
        )

ax7.plot(datetime8, data8["y"], label="Observed", color='C0')
ax7.xaxis.set_major_locator(mdates.YearLocator(base=4))
ax7.set_ylim([ymin, ymax])
ax7.set_yticks(yticks)
ax7.set_yticklabels([])
ax7.axvspan(datetime8[0],
            datetime8[int(len(data8["y"])*(training_splits[7]))],
            color="red",
            alpha=0.1,
            edgecolor=None,
        )
ax7.axvspan(datetime8[int(len(data8["y"])*training_splits[7]+1)],
            datetime8[int(len(data8["y"])*(training_splits[7]+validation_splits[7]))],
            color="green",
            alpha=0.1,
            edgecolor=None,
        )

ax8.plot(datetime9, data9["y"], label="Observed", color='C0')
ax8.xaxis.set_major_locator(mdates.YearLocator(base=4, month=12))
ax8.set_ylim([ymin, ymax])
ax8.set_yticks(yticks)
ax8.set_yticklabels([])
ax8.axvspan(datetime9[0],
            datetime9[int(len(data9["y"])*(training_splits[8]))],
            color="red",
            alpha=0.1,
            edgecolor=None,
        )
ax8.axvspan(datetime9[int(len(data9["y"])*training_splits[8]+1)],
            datetime9[int(len(data9["y"])*(training_splits[8]+validation_splits[8]))],
            color="green",
            alpha=0.1,
            edgecolor=None,
        )

ax9.plot(datetime10, data10["y"], label="Observed", color='C0')
ax9.xaxis.set_major_locator(mdates.YearLocator(base=4))
ax9.set_ylim([ymin, ymax])
ax9.set_yticks(yticks)
ax9.set_yticklabels([])
ax9.axvspan(datetime10[0],
            datetime10[int(len(data10["y"])*(training_splits[9]))],
            color="red",
            alpha=0.1,
            edgecolor=None,
        )
ax9.axvspan(datetime10[int(len(data10["y"])*training_splits[9]+1)],
            datetime10[int(len(data10["y"])*(training_splits[9]+validation_splits[9]))],
            color="green",
            alpha=0.1,
            edgecolor=None,
        )

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.6, wspace=0.15)
plt.savefig('real_ts_norm_demo.png', dpi=300)
plt.show()