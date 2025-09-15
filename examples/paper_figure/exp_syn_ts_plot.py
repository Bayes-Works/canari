import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


data_file = "./data/toy_time_series/syn_data_simple_phi05.csv"
df_simple = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_simple.iloc[:, 0])
# Set the first column name to "ds"
df_simple.columns = ['ds', 'y']
df_simple['ds'] = pd.to_datetime(df_simple['ds']).dt.date

data_file = "./data/toy_time_series/syn_data_complex_phi09.csv"
df_complex = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_complex.iloc[:, 0])
# Set the first column name to "ds"
df_complex.columns = ['ds', 'y']
df_complex['ds'] = pd.to_datetime(df_complex['ds']).dt.date


# #  Plot df_simple and df_complex
#  Plot states from pretrained model
fig = plt.figure(figsize=(5.5, 2.3), constrained_layout=True)
gs = gridspec.GridSpec(2, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.plot(df_simple['ds'], df_simple['y'], label="Simple TS", color='C0')
ax0.set_xticklabels([])
ax0.set_title('(a) Simple time series: $\phi^{\mathtt{AR}}=0.5$', loc='center', fontsize=12)
ax1.plot(df_complex['ds'], df_complex['y'], label="Complex TS", color='C1')
ax1.set_xticks(df_simple['ds'][::52*4])
ax1.set_title('(b) Complex time series: $\phi^{\mathtt{AR}}=0.9$', loc='center', fontsize=12)
# ax1.legend()

# Format x-axis ticks as year only
# ax0.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax0.xaxis.set_major_locator(mdates.YearLocator(base=3))
ax1.xaxis.set_major_locator(mdates.YearLocator(base=3))

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.6)
plt.savefig('syn_ts_demo.png', dpi=300)
plt.show()