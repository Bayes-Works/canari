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
anm_mag = 0.1/52
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
m = Prophet(changepoint_range=1, n_changepoints=int(len(df)/52*12), growth='linear')


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
fig1 = m.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), m, forecast, threshold=0.045)
# for i in range(len(forecast)):
#     if forecast['anomaly'][i] == 1:
#         plt.plot(forecast['ds'][i], df['y'][i], 'r.')

# Plot anomaly as vline
plt.axvline(x=m.history['ds'][anm_start_index], color='k', linestyle='--')


fig2 = m.plot_components(forecast)
plt.show()