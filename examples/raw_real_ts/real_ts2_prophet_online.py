import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
import numpy as np

import os
os.environ['OMP_NUM_THREADS'] = '1'

# data_file = "./data/benchmark_data/test_2_data.csv"
# df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df_raw.iloc[:, 0])
# # Set the first column name to "ds"
# df_raw.columns = ['ds', 'y']

data_file = "./data/benchmark_data/test_2_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=";", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 4])
df_raw = df_raw.iloc[:, 6].to_frame()
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["y"]
df_raw = df_raw.resample("W").mean()
df_raw = df_raw.iloc[30:, :]
df_raw = df_raw.reset_index().rename(columns={"date_time": "ds"})

print(df_raw)

train_split=0.25
validation_split=0.08


# df_raw = df_raw.iloc[:int(len(df_raw) * (train_split + validation_split))]
# begin_idx = int(52 * 3)

df_raw = df_raw.iloc[:int(len(df_raw) * 1)]
begin_idx = int(len(df_raw) * (train_split + validation_split))

threshold = 0.4

anm_detect_points = []
change_points_predicted = []
latest_changepoint = None

for i in range(len(df_raw) - 2 - begin_idx):
    current_idx = begin_idx + i

    df = df_raw.iloc[:current_idx]

    # m = Prophet(changepoint_range=1, n_changepoints=int(len(df)/52*12), changepoint_prior_scale=changepoint_prior_scale, growth='linear')
    m = Prophet(changepoint_range=1)
    m.fit(df)
    changepoint_grid_width = m.changepoints.index[1]- m.changepoints.index[0]

    forecast = m.predict(df)

    # Multiply forecast['yhat_lower'] and forecast['yhat_upper'] by a scale
    forecast['yhat_lower'] = forecast['yhat'] - (forecast['yhat'] - forecast['yhat_lower']) * 1
    forecast['yhat_upper'] = forecast['yhat'] + (forecast['yhat_upper'] - forecast['yhat']) * 1

    forecast['anomaly'] = ((df['y'] < forecast['yhat_lower']) | (df['y'] > forecast['yhat_upper'])).astype(int)

    # print(forecast)
    fig1 = m.plot(forecast)
    a = add_changepoints_to_plot(fig1.gca(), m, forecast, threshold=threshold)

    # # Plot the anomalies in figure 1
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

    # Get online changepoint detection
    signif_changepoints = m.changepoints[
        np.abs(np.nanmean(m.params['delta'], axis=0)) >= threshold
    ] if len(m.changepoints) > 0 else []
    if len(signif_changepoints) > 0:
        signif_changepoints = signif_changepoints.tolist()
        if latest_changepoint is None:
            latest_changepoint = signif_changepoints[-1]
            change_points_predicted = change_points_predicted + signif_changepoints
            anm_detect_points.append(current_idx)
        else:
            for cp in signif_changepoints:
                changepoint_increase = False
                if cp - pd.Timedelta(weeks=changepoint_grid_width) > latest_changepoint:
                    latest_changepoint = cp
                    change_points_predicted.append(cp)
                    changepoint_increase = True
            if changepoint_increase:
                anm_detect_points.append(current_idx)

m = Prophet(changepoint_range=1, changepoints=change_points_predicted)
m.fit(df_raw)
forecast = m.predict(df_raw)

fig1 = m.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), m, forecast, threshold=threshold)
# plt.axvline(x=m.history['ds'][anm_start_index], color='k', linestyle='--')
print(anm_detect_points)
if len(anm_detect_points)>0:
    for anm_detect_point in anm_detect_points:
        plt.axvline(x=m.history['ds'][anm_detect_point], color='g', linestyle='--')
fig2 = m.plot_components(forecast)
plt.show()