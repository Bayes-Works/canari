import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
import numpy as np

import os
os.environ['OMP_NUM_THREADS'] = '1'

# Read another csv file and store it in df["ds"]
data_file_time = "./data/toy_time_series/synthetic_simple_autoregression_periodic_datetime.csv"
df = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
df[0] = pd.to_datetime(df[0]).dt.strftime('%Y-%m-%d')
df.columns = ["ds"]

data_file = "./data/toy_time_series/synthetic_simple_autoregression_periodic.csv"
values = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
df["y"] = values
df.head()

# LT anomaly
# anm_mag = 0.010416667/10
anm_start_index = 52*10
anm_mag = 0.3/52
# anm_baseline = np.linspace(0, 3, num=len(df_raw))
anm_baseline = np.arange(len(df)) * anm_mag
# Set the first 52*12 values in anm_baseline to be 0
anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
anm_baseline[:anm_start_index] = 0

# # LL anomaly
# anm_mag = 0.5
# anm_baseline = np.zeros_like(df_raw)
# anm_baseline[anm_start_index:] += anm_mag

df["y"] = df["y"].add(anm_baseline, axis=0)
# Remove the last 52*5 rows in df_raw
df = df[:-52*5]

df = df.iloc[:int(len(df) * 1)]

# m = Prophet(changepoint_range=1, n_changepoints=int(len(df) * 0.1), changepoint_prior_scale=0.003, growth='linear')
# m = Prophet(changepoint_range=1, n_changepoints=int(len(df)), changepoint_prior_scale=0.009, growth='linear')
# m = Prophet(changepoint_range=1, n_changepoints=int(len(df)), changepoint_prior_scale=0.007, growth='linear')
m = Prophet(changepoint_range=1)


m.fit(df)

# future = m.make_future_dataframe(periods=365)
# print(future.tail())

# forecast = m.predict(future)
forecast = m.predict(df)
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Multiply forecast['yhat_lower'] and forecast['yhat_upper'] by a scale
forecast['yhat_lower'] = forecast['yhat'] - (forecast['yhat'] - forecast['yhat_lower']) * 1
forecast['yhat_upper'] = forecast['yhat'] + (forecast['yhat_upper'] - forecast['yhat']) * 1

forecast['anomaly'] = ((df['y'] < forecast['yhat_lower']) | (df['y'] > forecast['yhat_upper'])).astype(int)

# print(forecast)
# Plot the anomalies in figure 1
threshold=0.5

fig1 = m.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), m, forecast, threshold=threshold)
# for i in range(len(forecast)):
#     if forecast['anomaly'][i] == 1:
#         plt.plot(forecast['ds'][i], df['y'][i], 'r.')

# Plot anomaly as vline
plt.axvline(x=m.history['ds'][anm_start_index], color='k', linestyle='--')


fig2 = m.plot_components(forecast)


# ############################### Plot for presentation ################################
# remove_until_index = int((0.2+0.0693)*len(df))

# level = forecast['trend'].to_numpy()
# level = level[remove_until_index:]

# signif_changepoints = m.changepoints[
#     np.abs(np.nanmean(m.params['delta'], axis=0)) >= threshold
# ] if len(m.changepoints) > 0 else []

# anm_detected_index = signif_changepoints.index
# # print(signif_changepoints.index)

# time = pd.to_datetime(df["ds"]).dt.strftime('%Y-%m-%d')
# time = np.arange(len(df))
# data = df["y"].values

# time = time[remove_until_index:]
# data = data[remove_until_index:]
# anm_start_index = anm_start_index - remove_until_index
# anm_detected_index = anm_detected_index - remove_until_index

# from matplotlib import ticker
# formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_scientific(True) 
# formatter.set_powerlimits((-1,1)) 
# params = {'text.usetex' : True,
#           'font.size' : 12,
#           'font.family' : 'lmodern',
#           }
# plt.rcParams.update(params)
# from matplotlib import gridspec
# fig = plt.figure(figsize=(5, 1.5))
# gs = gridspec.GridSpec(1, 1)
# ax0 = plt.subplot(gs[0])

# ax0.plot(time, data, color='k')
# ax0.plot(time, level)
# # ax0.fill_between(
# #     time,
# #     base_model_mu_prior[:,0,0] - np.sqrt(base_model_var_prior[:,0,0]),
# #     base_model_mu_prior[:,0,0] + np.sqrt(base_model_var_prior[:,0,0]),
# #     alpha=0.5,
# #     color="gray",
# # )
# ax0.axvline(x=time[anm_start_index], color='k', linestyle='--', label="trend change")
# for i in range(len(anm_detected_index)):
#     if i == 0:
#         ax0.axvline(x=time[anm_detected_index[i]], color='r', linestyle='--', label="detect")
#     else:
#         ax0.axvline(x=time[anm_detected_index[i]], color='r', linestyle='--')
# ax0.set_ylabel('obs')
# # ax0.set_yticks([0, 2.5])
# ax0.set_yticklabels([])
# # print(ax0.get_ylim())
# ax0.set_xticks([time[int(len(time)*1/9)-1], time[int(len(time)*3/9)-1],time[int(len(time)*5/9)-1],time[int(len(time)*7/9)-1],time[int(-1)]])
# ax0.set_xticklabels(['2016', '2018', '2020', '2022', '2024'])
# plt.tight_layout(h_pad=0.5, w_pad=0.1)
# plt.savefig('hsl.png', dpi=300)

plt.show()