import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
import numpy as np
import pytagi.metric as metric
import ast
from tqdm import tqdm
import copy
from pytagi import Normalizer

import os
os.environ['OMP_NUM_THREADS'] = '1'

data_file = "./data/benchmark_data/detrended_data/test_7_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
# Set the first column name to "ds"
df_raw.columns = ['ds', 'y']

train_split=0.3
validation_split=0.1
train_end = int(np.floor(train_split * len(df_raw)))

# Get the train and validation set
validation_start = int(np.floor(train_split * len(df_raw)))
test_start = validation_start + int(
    np.ceil(validation_split * len(df_raw))
)

# # # Read test data
df = pd.read_csv("data/prob_eva_syn_time_series/detrended_ts7_tsgen.csv")

# Containers for restored data
restored_data = []
for _, row in df.iterrows():
    values = np.array(eval(row["values"], {"nan": float("nan")}), dtype=float)
    anomaly_magnitude = float(row["anomaly_magnitude"])
    anomaly_start_index = int(row["anomaly_start_index"])
    restored_data.append((values, anomaly_magnitude, anomaly_start_index))

begin_idx = int(len(df_raw) * 0.4)
threshold = 0.3
results_all = []

for ts_index in tqdm(range(len(restored_data))):
# for ts_index in tqdm(range(2)):
#     ts_index += 150

    df_k = copy.deepcopy(df_raw)
    raw_data_k = restored_data[ts_index][0]
    # Replace the values in the dataframe with the restored_data[k][0]
    # Normalize data
    scale_const_mean, scale_const_std = Normalizer.compute_mean_std(
                raw_data_k[0 : train_end]
            )
    norm_data = Normalizer.standardize(
                data=raw_data_k,
                mu=scale_const_mean,
                std=scale_const_std,
            )
    df_k.iloc[:, 1] = norm_data

    # generate_dates = restored_data[ts_index][0]
    anm_mag = restored_data[ts_index][1]
    anm_start_index = restored_data[ts_index][2]
    anm_start_index_global = anm_start_index + test_start

    # Get true baseline
    anm_mag_normed = anm_mag
    LL_baseline_true = np.zeros(len(df_raw))
    LT_baseline_true = np.zeros(len(df_raw))
    for i in range(1, len(df_raw)):
        if i > anm_start_index_global:
            LL_baseline_true[i] += anm_mag_normed * (i - anm_start_index_global)
            LT_baseline_true[i] += anm_mag_normed

    # LL_baseline_true += model_dict['early_stop_init_mu_states'][0].item()
    LL_baseline_true = LL_baseline_true.flatten()
    LT_baseline_true = LT_baseline_true.flatten()

    anm_detect_points = []
    change_points_predicted = []
    latest_changepoint = None

    online_LL = np.full((begin_idx,), np.nan).tolist()
    online_LT = np.full((begin_idx,), np.nan).tolist()

    for i in range(len(df_k)-begin_idx):
        current_idx = begin_idx + i
        df_ki = df_k.iloc[:current_idx]

        # m = Prophet(changepoint_range=1, n_changepoints=int(len(df_k)/52*12), changepoint_prior_scale=changepoint_prior_scale, growth='linear')
        m = Prophet(changepoint_range=1)
        m.fit(df_ki)
        changepoint_grid_width = m.changepoints.index[1]- m.changepoints.index[0]

        forecast = m.predict(df_ki)

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

        # Get online LL and LT
        LL_baseline_temp = forecast["trend"]
        LT_baseline_temp = LL_baseline_temp.diff()
        LT_pred = LT_baseline_temp.iloc[-1]
        LL_pred = LL_baseline_temp.iloc[-1] + LT_pred
        online_LL.append(LL_pred)
        online_LT.append(LT_pred)

    # m = Prophet(changepoint_range=1, n_changepoints=int(len(df)/52*12), changepoint_prior_scale=changepoint_prior_scale, growth='linear')
    m = Prophet(changepoint_range=1, changepoints=change_points_predicted)
    m.fit(df_k)
    forecast = m.predict(df_k)

    # fig1 = m.plot(forecast)
    # a = add_changepoints_to_plot(fig1.gca(), m, forecast, threshold=threshold)
    # # plt.axvline(x=m.history['ds'][anm_start_index], color='k', linestyle='--')
    # if len(anm_detect_points)>0:
    #     for anm_detect_point in anm_detect_points:
    #         plt.axvline(x=m.history['ds'][anm_detect_point], color='k', linestyle='--')
    #     for cp in signif_changepoints:
    #         plt.axvline(x=cp, color='g', linestyle='--')
    # fig2 = m.plot_components(forecast)
    # plt.show()

    LL_baseline_estimate = forecast["trend"]
    LT_baseline_estimate = LL_baseline_estimate.diff()
    # Convert LL_baseline_estimate to numpy array
    LL_baseline_estimate = LL_baseline_estimate.to_numpy().flatten()
    LT_baseline_estimate = LT_baseline_estimate.to_numpy().flatten()

    online_LL = np.array(online_LL).flatten()
    online_LT = np.array(online_LT).flatten()

    mse_LL = metric.mse(
        # LL_baseline_estimate[anm_start_index_global+1:],
        online_LL[anm_start_index_global+1:],
        LL_baseline_true[anm_start_index_global+1:],
    )
    mse_LT = metric.mse(
        # LT_baseline_estimate[anm_start_index_global+1:],
        online_LT[anm_start_index_global+1:],
        LT_baseline_true[anm_start_index_global+1:],
    )

    # # # Plot all the baselines, online_LL vs LL_baseline_true, online_LT vs LT_baseline_true
    # fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    # ax[0].plot(df_k['ds'], online_LL, label='LL Online Estimate', color='blue')
    # ax[0].plot(df_k['ds'], LL_baseline_true, label='LL True', color='orange')
    # ax[0].set_title('LL Baseline Estimate vs True')
    # ax[0].set_xlabel('Date')
    # ax[0].set_ylabel('LL Value')
    # ax[0].legend()  
    # ax[1].plot(df_k['ds'], online_LT, label='LT Online Estimate', color='blue')
    # ax[1].plot(df_k['ds'], LT_baseline_true, label='LT True', color='orange')
    # ax[1].set_title('LT Baseline Estimate vs True')
    # ax[1].set_xlabel('Date')
    # ax[1].set_ylabel('LT Value')
    # ax[1].legend()
    # plt.tight_layout()
    # plt.show()

    # Check if anm_detect_points is empty
    if len(anm_detect_points) == 0:
        detection_time = len(df_k) - anm_start_index_global
    else:
        detection_time = anm_detect_points[0] - anm_start_index_global

    anm_detected_index = str(anm_detect_points)

    results_all.append([anm_mag, anm_start_index_global, anm_detected_index, mse_LL, mse_LT, detection_time])

# Save the results to a CSV file
results_df = pd.DataFrame(results_all, columns=["anomaly_magnitude", "anomaly_start_index", "anomaly_detected_index", "mse_LL", "mse_LT", "detection_time"])
results_df.to_csv("saved_results/prob_eva/detrended_ts7_results_prophet_online.csv", index=False)