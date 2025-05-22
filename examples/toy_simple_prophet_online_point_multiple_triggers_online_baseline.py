import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
import numpy as np
import pytagi.metric as metric

import os
os.environ['OMP_NUM_THREADS'] = '1'

# Read another csv file and store it in df["ds"]
data_file_time = "./data/toy_time_series/synthetic_simple_autoregression_periodic_datetime.csv"
df_raw = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
df_raw[0] = pd.to_datetime(df_raw[0]).dt.strftime('%Y-%m-%d')
df_raw.columns = ["ds"]

data_file = "./data/toy_time_series/synthetic_simple_autoregression_periodic.csv"
values = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
df_raw["y"] = values
df_raw.head()

# LT anomaly
# anm_mag = 0.010416667/10
anm_start_index = 52*10
anm_mag = 0.3/52
# anm_baseline = np.linspace(0, 3, num=len(df_raw))
anm_baseline = np.arange(len(df_raw)) * anm_mag
# Set the first 52*12 values in anm_baseline to be 0
anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
anm_baseline[:anm_start_index] = 0

# # LL anomaly
# anm_mag = 0.5
# anm_baseline = np.zeros_like(df_raw)
# anm_baseline[anm_start_index:] += anm_mag

df_raw["y"] = df_raw["y"].add(anm_baseline, axis=0)
# Remove the last 52*5 rows in df_raw
df_raw = df_raw[:-52*5]

df_raw = df_raw.iloc[:int(len(df_raw) * 1)]

# Get true baseline
anm_mag_normed = anm_mag
LL_baseline_true = np.zeros(len(df_raw))
LT_baseline_true = np.zeros(len(df_raw))
for i in range(1, len(df_raw)):
    if i > anm_start_index:
        LL_baseline_true[i] += anm_mag_normed * (i - anm_start_index)
        LT_baseline_true[i] += anm_mag_normed

# LL_baseline_true += model_dict['early_stop_init_mu_states'][0].item()
LL_baseline_true = LL_baseline_true.flatten()
LT_baseline_true = LT_baseline_true.flatten()

# Genetrate percentages_check from 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, ... , 1
# percentages_check = [i / 100 for i in range(10, 101, 1)]
begin_idx = int(len(df_raw) * 0.9)

# changepoint_prior_scale = 0.01
threshold = 0.5

anm_detect_points = []
change_points_predicted = []
latest_changepoint = None

online_LL = np.full((begin_idx,), np.nan)
online_LT = np.full((begin_idx,), np.nan)

for i in range(len(df_raw)-begin_idx):
    current_idx = begin_idx + i
    df = df_raw.iloc[:current_idx]

    m = Prophet(changepoint_range=1)
    m.fit(df)
    changepoint_grid_width = m.changepoints.index[1]- m.changepoints.index[0]

    # future = m.make_future_dataframe(periods=365)
    # print(future.tail())

    # forecast = m.predict(future)
    forecast = m.predict(df)
    # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # print(forecast)
    fig1 = m.plot(forecast)
    a = add_changepoints_to_plot(fig1.gca(), m, forecast, threshold=threshold)
    fig1.gca().set_ylim(-1.508410797906945, 2.650273557375443)
    fig1.gca().set_xlim(10684.35, 16682.65)

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

# m = Prophet(changepoint_range=1, n_changepoints=int(len(df)/52*12), changepoint_prior_scale=changepoint_prior_scale, growth='linear')
m = Prophet(changepoint_range=1, changepoints=change_points_predicted)
m.fit(df_raw)
forecast = m.predict(df_raw)

LL_baseline_estimate = forecast["trend"]
LT_baseline_estimate = LL_baseline_estimate.diff()
# Convert LL_baseline_estimate to numpy array
LL_baseline_estimate = LL_baseline_estimate.to_numpy().flatten()
LT_baseline_estimate = LT_baseline_estimate.to_numpy().flatten()

# Plot all the baselines, true and estimated
plt.figure()
plt.plot(m.history['ds'], LL_baseline_true, label="LL Baseline True", color='blue')
plt.plot(m.history['ds'], LL_baseline_estimate, label="LL Baseline Estimate", color='green')
plt.axvline(x=m.history['ds'][anm_start_index], color='k', linestyle='--')
plt.ylabel('LL')

plt.figure()
plt.plot(m.history['ds'], LT_baseline_true, label="LT Baseline True", color='blue')
plt.plot(m.history['ds'], LT_baseline_estimate, label="LT Baseline Estimate", color='green')
plt.axvline(x=m.history['ds'][anm_start_index], color='k', linestyle='--')
plt.ylabel('LT')

mse_LL = metric.mse(
    LL_baseline_estimate[anm_start_index+1:],
    LL_baseline_true[anm_start_index+1:],
)
mse_LT = metric.mse(
    LT_baseline_estimate[anm_start_index+1:],
    LT_baseline_true[anm_start_index+1:],
)
print("MSE LL: ", mse_LL)
print("MSE LT: ", mse_LT)

fig1 = m.plot(forecast)
plt.axvline(x=m.history['ds'][anm_start_index], color='k', linestyle='--')
if len(anm_detect_points)>0:
    for anm_detect_point in anm_detect_points:
        plt.axvline(x=m.history['ds'][anm_detect_point], color='r', linestyle='--')
    for cp in change_points_predicted:
        plt.axvline(x=cp, color='g', linestyle='--')
fig2 = m.plot_components(forecast)
plt.show()