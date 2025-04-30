import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
import numpy as np
import pytagi.metric as metric
import ast
from tqdm import tqdm

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

train_split=0.289
validation_split=0.0693*2

# Remove the last 52*5 rows in df_raw
train_split = train_split * len(df_raw) / len(df_raw[:-52*5])
validation_split = validation_split * len(df_raw) / len(df_raw[:-52*5])
df_raw = df_raw[:-52*5]
# df_raw = df_raw.iloc[:int(len(df_raw) * 1)]

# Get the train and validation set
validation_start = int(np.floor(train_split * len(df_raw)))
test_start = validation_start + int(
    np.ceil(validation_split * len(df_raw))
)

# Remove the first 5 years of data
remove_begin_len = 5*52
df_raw = df_raw.iloc[remove_begin_len:]
validation_start -= remove_begin_len
test_start -= remove_begin_len

# Load the CSV
df = pd.read_csv("data/prob_eva_syn_time_series/toy_simple_tsgen_stationary.csv")
# Containers for restored data
restored_data = []
for _, row in df.iterrows():
    # Convert string to list, then to desired type
    timestamps = pd.to_datetime(ast.literal_eval(row["timestamps"])).strftime('%Y-%m-%d')
    values = np.array(ast.literal_eval(row["values"]), dtype=float)
    anomaly_magnitude = float(row["anomaly_magnitude"])
    anomaly_start_index = int(row["anomaly_start_index"])
    
    restored_data.append((timestamps, values, anomaly_magnitude, anomaly_start_index))

begin_idx = int(len(df_raw) * 0.33)
threshold = 0.5
results_all = []

for ts_index in tqdm(range(len(restored_data))):

    # Insert the restored data into the original DataFrame
    test_len = len(df_raw) - test_start
    df_raw = df_raw[:-test_len]

    gen_time_series = restored_data[ts_index][1]
    generate_dates = restored_data[ts_index][0]
    anm_mag = restored_data[ts_index][2]
    anm_start_index = restored_data[ts_index][3]
    anm_start_index_global = anm_start_index + len(df_raw)

    new_df = pd.DataFrame({'ds': generate_dates, 'y': gen_time_series})
    df_raw = pd.concat([df_raw, new_df], ignore_index=True)
    # print(df_raw[test_start-5:test_start+5])

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

    # # Plot data with validation and test starts marked
    # plt.figure()
    # plt.plot(df_raw['ds'], df_raw['y'], label="Data", color='blue')
    # plt.axvline(x=df_raw['ds'][validation_start], color='r', linestyle='--', label="Validation Start")
    # plt.axvline(x=df_raw['ds'][test_start], color='g', linestyle='--', label="Test Start")
    # plt.plot(df_raw['ds'], LL_baseline_true, label="LL Baseline True", color='orange')
    # plt.xlabel('Date')
    # plt.ylabel('Value')
    # plt.title('Data with Validation and Test Starts')
    # plt.show()

    anm_detect_points = []
    change_points_predicted = []
    latest_changepoint = None

    for i in range(len(df_raw)-begin_idx):
        current_idx = begin_idx + i
        df = df_raw.iloc[:current_idx]

        # m = Prophet(changepoint_range=1, n_changepoints=int(len(df)/52*12), changepoint_prior_scale=changepoint_prior_scale, growth='linear')
        m = Prophet(changepoint_range=1)
        m.fit(df)
        changepoint_grid_width = m.changepoints.index[1]- m.changepoints.index[0]

        forecast = m.predict(df)

        # fig1 = m.plot(forecast)
        # a = add_changepoints_to_plot(fig1.gca(), m, forecast, threshold=threshold)
        
        # if i != len(df_raw) - begin_idx - 1:
        #     plt.pause(0.5)
        #     plt.close(fig1)
        # elif i == len(df_raw) - begin_idx - 1:
        #     print("--------finished--------")
        #     fig2 = m.plot_components(forecast)
        #     plt.show()
        #     # # Get the ylimit of the first figure
        #     # print(fig1.gca().get_ylim())
        #     # print(fig1.gca().get_xlim())

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

    # # Plot all the baselines, true and estimated
    # plt.figure()
    # plt.plot(m.history['ds'], LL_baseline_true, label="LL Baseline True", color='blue')
    # plt.plot(m.history['ds'], LL_baseline_estimate, label="LL Baseline Estimate", color='green')
    # plt.axvline(x=m.history['ds'][anm_start_index_global], color='k', linestyle='--')
    # plt.ylabel('LL')

    # plt.figure()
    # plt.plot(m.history['ds'], LT_baseline_true, label="LT Baseline True", color='blue')
    # plt.plot(m.history['ds'], LT_baseline_estimate, label="LT Baseline Estimate", color='green')
    # plt.axvline(x=m.history['ds'][anm_start_index_global], color='k', linestyle='--')
    # plt.ylabel('LT')

    mse_LL = metric.mse(
        LL_baseline_estimate[anm_start_index_global+1:],
        LL_baseline_true[anm_start_index_global+1:],
    )
    mse_LT = metric.mse(
        LT_baseline_estimate[anm_start_index_global+1:],
        LT_baseline_true[anm_start_index_global+1:],
    )
    print("MSE LL: ", mse_LL)
    print("MSE LT: ", mse_LT)

    # fig1 = m.plot(forecast)
    # plt.axvline(x=m.history['ds'][anm_start_index_global], color='k', linestyle='--')
    # if len(anm_detect_points)>0:
    #     for anm_detect_point in anm_detect_points:
    #         plt.axvline(x=m.history['ds'][anm_detect_point], color='r', linestyle='--')
    #     for cp in change_points_predicted:
    #         plt.axvline(x=cp, color='g', linestyle='--')
    # fig2 = m.plot_components(forecast)
    # plt.show()

    # Check if anm_detect_points is empty
    if len(anm_detect_points) == 0:
        detection_time = len(df_raw) - anm_start_index_global
    else:
        detection_time = anm_detect_points[0] - anm_start_index_global

    anm_detected_index = str(anm_detect_points)

    results_all.append([anm_mag, anm_start_index_global, anm_detected_index, mse_LL, mse_LT, detection_time])

# # Save the results to a CSV file
# results_df = pd.DataFrame(results_all, columns=["anomaly_magnitude", "anomaly_start_index", "anomaly_detected_index", "mse_LL", "mse_LT", "detection_time"])
# results_df.to_csv("saved_results/prob_eva/toy_simple_results_prophet_point.csv", index=False)