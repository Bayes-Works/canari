import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari import (
    DataProcess,
)
from matplotlib import gridspec
import matplotlib.dates as mdates

# # # Read data
data_file = "./data/toy_time_series/syn_data_simple_phi05.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
date_time = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = date_time
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

# # # Read test data
df = pd.read_csv("data/prob_eva_syn_time_series/syn_simple_tsgen.csv")

# Containers for restored data
restored_data = []
for _, row in df.iterrows():
    values = np.array(eval(row["values"], {"nan": float("nan")}), dtype=float)
    anomaly_magnitude = float(row["anomaly_magnitude"])
    anomaly_start_index = int(row["anomaly_start_index"])
    
    restored_data.append((values, anomaly_magnitude, anomaly_start_index))

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

# Plot generated time series
fig = plt.figure(figsize=(5.5, 1.8), constrained_layout=True)
gs = gridspec.GridSpec(1, 1)
ax0 = plt.subplot(gs[0])
for j in range(int(len(restored_data)/10)):
    j = int(j*10)
    ax0.plot(date_time, restored_data[j][0])
    ax0.axvline(x=date_time[restored_data[j][2]+len(df_raw)-len(test_data["y"])], color='r', linestyle='--', alpha=0.3)
# ax0.axvline(x=len(self.data_processor.data.values[train_index, self.data_processor.output_col].reshape(-1))+len(self.data_processor.data.values[val_index, self.data_processor.output_col].reshape(-1)), color='r', linestyle='--')
# ax0.set_title("Data generation")

ax0.xaxis.set_major_locator(mdates.YearLocator(base=3))

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.6)
plt.savefig('syn_anm_demo.png', dpi=300)
plt.show()