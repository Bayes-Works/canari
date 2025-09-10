import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
import numpy as np

import os
os.environ['OMP_NUM_THREADS'] = '1'

data_file = "./data/toy_time_series/syn_data_complex_phi09.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
# Set the first column name to "ds"
df_raw.columns = ['ds', 'y']

# print(df_raw)


# # LT anomaly
# # anm_mag = 0.010416667/10
# anm_start_index = 52*10
# anm_mag = 0.3/52
# # anm_baseline = np.linspace(0, 3, num=len(df_raw))
# anm_baseline = np.arange(len(df_raw)) * anm_mag
# # Set the first 52*12 values in anm_baseline to be 0
# anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
# anm_baseline[:anm_start_index] = 0

# # LL anomaly
# anm_mag = 0.5
# anm_baseline = np.zeros_like(df_raw)
# anm_baseline[anm_start_index:] += anm_mag


df_raw = df_raw.iloc[:int(len(df_raw) * 1)]

# Genetrate percentages_check from 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, ... , 1
# percentages_check = [i / 100 for i in range(10, 101, 1)]
begin_idx = int(len(df_raw) * 0.6)

# changepoint_prior_scale = 0.01
threshold = 1.1

anm_detect_point = None

for i in range(len(df_raw)-begin_idx):
    current_idx = begin_idx + i
    df = df_raw.iloc[:current_idx]

    # m = Prophet(changepoint_range=1, n_changepoints=int(len(df)/52*12), changepoint_prior_scale=changepoint_prior_scale, growth='linear')
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
    fig1 = m.plot(forecast)
    a = add_changepoints_to_plot(fig1.gca(), m, forecast, threshold=threshold)
    # fig1.gca().set_ylim(-1.508410797906945, 2.650273557375443)
    # fig1.gca().set_xlim(14384.85, 20152.15)

    # Plot the anomalies in figure 1
    # for i in range(len(forecast)):
    #     if forecast['anomaly'][i] == 1:
    #         plt.plot(forecast['ds'][i], df['y'][i], 'r.')

    # fig2 = m.plot_components(forecast)
    # Show the figure automatically, and close for the next loop
    # plt.show()

    # # Plot the trend rate of change
    # # Trend change point threshold: m.params['delta'][0] > 0.01
    # plt.figure()
    # plt.plot(m.params['delta'][0])
    # plt.axhline(0, color='black', linestyle='--')
    # plt.title('Trend Rate of Change')
    # plt.xlabel('Date')
    # plt.ylabel('Rate of Trend Change')
    
    if i != len(df_raw) - begin_idx - 1:
        plt.pause(0.5)
        plt.close(fig1)
    elif i == len(df_raw) - begin_idx - 1:
        fig2 = m.plot_components(forecast)
        print("--------finished--------")
        plt.show()
        # # Get the ylimit of the first figure
        print(fig1.gca().get_ylim())
        print(fig1.gca().get_xlim())

    signif_changepoints = m.changepoints[
        np.abs(np.nanmean(m.params['delta'], axis=0)) >= threshold
    ] if len(m.changepoints) > 0 else []
    if len(signif_changepoints) > 0:
        anm_detect_point = current_idx
        break

# m = Prophet(changepoint_range=1, n_changepoints=int(len(df)/52*12), changepoint_prior_scale=changepoint_prior_scale, growth='linear')
m = Prophet(changepoint_range=1, changepoints=signif_changepoints)
m.fit(df_raw)
forecast = m.predict(df_raw)
fig1 = m.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), m, forecast, threshold=threshold)
print(fig1.gca().get_xlim())
print(fig1.gca().get_ylim())
# plt.axvline(x=m.history['ds'][anm_start_index], color='k', linestyle='--')
if anm_detect_point is not None:
    plt.axvline(x=m.history['ds'][anm_detect_point], color='r', linestyle='--')
    for cp in signif_changepoints:
        plt.axvline(x=cp, color='g', linestyle='--')
fig2 = m.plot_components(forecast)
plt.show()