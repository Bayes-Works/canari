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

data_file = "./data/toy_time_series/syn_data_anmtype_simple_phi05.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
# Set the first column name to "ds"
df_raw.columns = ['ds', 'y']

train_split=0.3
validation_split=0.1
train_end = int(np.floor(train_split * len(df_raw)))

# Normalize data
scale_const_mean, scale_const_std = Normalizer.compute_mean_std(
            df_raw["y"].values[0 : train_end]
        )

# Get the train and validation set
validation_start = int(np.floor(train_split * len(df_raw)))
test_start = validation_start + int(
    np.ceil(validation_split * len(df_raw))
)

# # # Read test data
df = pd.read_csv("data/prob_eva_syn_time_series/syn_rsic_simple_ts_gen_lltolt.csv")

# Containers for restored data
restored_data = []
time_stamps = eval(df.iloc[0]["timestamp"], {"nan": float("nan")})
for _, row in df.iterrows():
    values = np.array(eval(row["values"], {"nan": float("nan")}), dtype=float)
    anomaly1_magnitude = float(row["anomaly1_magnitude"])
    anomaly2_magnitude = float(row["anomaly2_magnitude"])
    anomaly_start_index1 = int(row["anomaly_start_index1"])
    anomaly_start_index2 = int(row["anomaly_start_index2"])
    
    restored_data.append((values, anomaly1_magnitude, anomaly2_magnitude, anomaly_start_index1, anomaly_start_index2))

begin_idx = int(len(df_raw) * 0.4)
threshold = 0.2
results_all = []

for p in range(10):
    for q in tqdm(range(len(restored_data)//10)):
# for p in range(1):
#     for q in np.array([7, 8]):
        ts_index = p + q * 10

        df_k = pd.DataFrame()
        print(df_k)
        df_k["ds"] = pd.to_datetime(time_stamps)
        df_k.index = np.arange(len(df_k))

        raw_data_k = restored_data[ts_index][0]
        # Replace the values in the dataframe with the restored_data[k][0]
        norm_data = Normalizer.standardize(
                    data=raw_data_k,
                    mu=scale_const_mean,
                    std=scale_const_std,
                )
        df_k["y"] = norm_data

        # anm_start_index = restored_data[ts_index][2]
        # anm_start_index_global = anm_start_index + test_start

        anm_mag1 = restored_data[ts_index][1]
        anm_mag2 = restored_data[ts_index][2]
        anm_start_index1 = restored_data[ts_index][3]
        anm_start_index2 = restored_data[ts_index][4]

        # Get baselines for comparison
        # True baselines
        true_LL_baseline = np.zeros(len(df_k))
        true_LT_baseline = np.zeros(len(df_k))
        anm_mag2_perweek = anm_mag2 / 52
        # LL to LT anomaly
        true_LL_baseline[anm_start_index1:] = anm_mag1
        true_LL_baseline[anm_start_index2:] += np.arange(len(true_LL_baseline)-anm_start_index2) * anm_mag2_perweek
        true_LT_baseline[anm_start_index2:] = anm_mag2_perweek

        # Convert the baselines to strings and save to results_all
        true_LL_baseline_str = str(true_LL_baseline.tolist())
        true_LT_baseline_str = str(true_LT_baseline.tolist())

        anm_detect_points = []
        change_points_predicted = []
        latest_changepoint = None

        online_LL = np.full((begin_idx,), 0).tolist()
        online_LT = np.full((begin_idx,), 0).tolist()

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

        estimate_LL_baseline_str = str(online_LL)
        estimate_LT_baseline_str = str(online_LT)

        # # # Plot all the baselines, online_LL vs LL_baseline_true, online_LT vs LT_baseline_true
        # fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        # ax[0].plot(df_k['ds'], online_LL, label='LL Online Estimate', color='blue')
        # ax[0].plot(df_k['ds'], true_LL_baseline, label='LL True', color='orange')
        # ax[0].set_title('LL Baseline Estimate vs True')
        # ax[0].set_xlabel('Date')
        # ax[0].set_ylabel('LL Value')
        # ax[0].legend()  
        # ax[1].plot(df_k['ds'], online_LT, label='LT Online Estimate', color='blue')
        # ax[1].plot(df_k['ds'], true_LT_baseline, label='LT True', color='orange')
        # ax[1].set_title('LT Baseline Estimate vs True')
        # ax[1].set_xlabel('Date')
        # ax[1].set_ylabel('LT Value')
        # ax[1].legend()
        # plt.tight_layout()
        # plt.show()

        all_detection_points = str(anm_detect_points)

        itv_log = []
        itv_applied_times = []

        results_all.append([anm_mag2, anm_start_index1, anm_start_index2, all_detection_points, itv_log, itv_applied_times, true_LL_baseline_str, true_LT_baseline_str, estimate_LL_baseline_str, estimate_LT_baseline_str])

# Save the results to a CSV file
results_df = pd.DataFrame(results_all, columns=["anomaly_magnitude", "anomaly_start_index1", "anomaly_start_index2", "anomaly_detected_index", "intervention_log", "intervention_applied_times", "true_LL_baseline", "true_LT_baseline", "estimated_LL_baseline", "estimated_LT_baseline"])
results_df.to_csv("saved_results/prob_eva/syn_simple_ts_results_prophet_lltolt.csv", index=False)