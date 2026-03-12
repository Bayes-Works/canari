import ast
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def _process_detection_compare_map(
    test_ts_len: int,
    csv_rsic_path: str,
    csv_skf_path: str,
    *,
    evaluate_itv_type: Optional[bool] = False,
    plot_detection_map: Optional[bool] = False,
    detection_col: str = "anomaly_detected_index",
    anm1_col: str = "anomaly_start_index1",
    anm2_col: str = "anomaly_start_index2",
    magnitude_col: str = "anomaly_magnitude",
    itv_log_col: str = "intervention_log",
    true_LL_baseline_col: str = "true_LL_baseline",
    true_LT_baseline_col: str = "true_LT_baseline",
    estimated_LL_baseline_col: str = "estimated_LL_baseline",
    estimated_LT_baseline_col: str = "estimated_LT_baseline",
    default_years: int = 3,
    freq_per_year: int = 52,
    first_anm_type = None,
) -> pd.DataFrame:
    """
    Load a CSV and reproduce your processing:
      - abs() anomaly_magnitude
      - parse detection_col as list (via ast.literal_eval)
      - compute:
          first_anm_detect_index
          detection_index_after_anm1
          detection_time (first detection >= anm2 start after removing <= first_anm_detect_index)

    Returns a NEW DataFrame (doesn't modify on disk).
    """
    df_rsic = pd.read_csv(csv_rsic_path)
    df_skf = pd.read_csv(csv_skf_path)

    # Keep anomaly magnitude as abs for LL anomaly
    if magnitude_col in df_rsic.columns:
        df_rsic[magnitude_col] = np.abs(df_rsic[magnitude_col])
    if magnitude_col in df_skf.columns:
        df_skf[magnitude_col] = np.abs(df_skf[magnitude_col])

    # Parse detected indices list
    if detection_col in df_rsic.columns:
        df_rsic[detection_col] = df_rsic[detection_col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    if detection_col in df_skf.columns:
        df_skf[detection_col] = df_skf[detection_col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    if itv_log_col in df_rsic.columns:
        df_rsic[itv_log_col] = df_rsic[itv_log_col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df_rsic["intervention_applied_times"] = df_rsic["intervention_applied_times"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    df_rsic[true_LL_baseline_col] = df_rsic[true_LL_baseline_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df_skf[true_LT_baseline_col] = df_skf[true_LT_baseline_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df_rsic["estimated_LL_baseline"] = df_rsic["estimated_LL_baseline"].apply(
        lambda x: ast.literal_eval(x.replace("nan", "None")))
    df_rsic[estimated_LL_baseline_col] = df_rsic[estimated_LL_baseline_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df_skf["estimated_LT_baseline"] = df_skf["estimated_LT_baseline"].apply(
        lambda x: ast.literal_eval(x.replace("nan", "None")))
    df_skf[estimated_LT_baseline_col] = df_skf[estimated_LT_baseline_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Plot the detection map for both rsic and skf
    x_plot = np.arange(test_ts_len)
    # Check all the rows in df_rsic and df_skf, record their detection density at each time index, and plot the detection map
    rsix_detection_density = np.zeros(test_ts_len)
    skfx_detection_density = np.zeros(test_ts_len)
    anomaly1_density = np.zeros(test_ts_len)
    anomaly2_density = np.zeros(test_ts_len)
    anomaly_density = np.zeros(test_ts_len)
    for idx, row in df_rsic.iterrows():
        detected_indices = row[detection_col]
        if isinstance(detected_indices, list):
            for detected_index in detected_indices:
                rsix_detection_density[detected_index] += 1
    for idx, row in df_skf.iterrows():
        detected_indices = row[detection_col]
        if isinstance(detected_indices, list):
            for detected_index in detected_indices:
                skfx_detection_density[detected_index] += 1
    for idx, row in df_rsic.iterrows():
        anm1_start = row[anm1_col]
        anm2_start = row[anm2_col]
        anomaly1_density[anm1_start] += 1
        anomaly2_density[anm2_start] += 1
        anomaly_density[anm1_start] += 1
        anomaly_density[anm2_start] += 1

    # Count only after the second anomaly
    # Find the smallest index where anomaly2_density > 0, and only keep the detection density after that index
    first_anomaly2_index = np.where(anomaly2_density > 0)[0][0]
    rsix_detection_density = rsix_detection_density[first_anomaly2_index:]
    skfx_detection_density = skfx_detection_density[first_anomaly2_index:]
    anomaly_density = anomaly_density[first_anomaly2_index:]
    x_plot = x_plot[first_anomaly2_index:]

    # # Convert the three densities to Smooth continuous curve using kde
    # rsix_kde = gaussian_kde(x_plot, weights=rsix_detection_density)
    # skfx_kde = gaussian_kde(x_plot, weights=skfx_detection_density)
    # anomaly1_kde = gaussian_kde(x_plot, weights=anomaly1_density)
    # anomaly2_kde = gaussian_kde(x_plot, weights=anomaly2_density)
    # anomaly_kde = gaussian_kde(x_plot, weights=anomaly_density)

    # # Normalize the histogram densities to [0, 1]
    # rsix_detection_density = rsix_detection_density / np.max(rsix_detection_density)
    # skfx_detection_density = skfx_detection_density / np.max(skfx_detection_density)
    # anomaly1_density = anomaly1_density / np.max(anomaly1_density)
    # anomaly2_density = anomaly2_density / np.max(anomaly2_density)

    # Plot the detection kde curves and anomaly kde curve
    plt.figure(figsize=(10, 6))

    # plt.plot(x_plot, rsix_kde(x_plot), label="RSIC Detection Density", color="blue")
    # plt.plot(x_plot, skfx_kde(x_plot), label="SKF Detection Density", color="orange")
    # plt.plot(x_plot, anomaly_kde(x_plot), label="Anomaly Density", color="red")
    # plt.plot(x_plot, anomaly2_kde(x_plot), label="Anomaly Density", color="red")

    plt.plot(x_plot, rsix_detection_density/max(rsix_detection_density), label=r"\textbf{RSIC Detection Density}", color="blue")
    plt.plot(x_plot, skfx_detection_density/max(skfx_detection_density), label=r"\textbf{SKF Detection Density}", color="orange")
    plt.plot(x_plot, anomaly_density/max(anomaly_density), label=r"\textbf{Anomaly Density}", color="red")

    # # Plot the histogram of detection density and anomaly density
    # plt.bar(x_plot, rsix_detection_density, label=r"\textbf{RSIC Detection Density}", color="blue", alpha=0.5)
    # plt.bar(x_plot, skfx_detection_density, label=r"\textbf{SKF Detection Density}", color="orange", alpha=0.5)
    # plt.bar(x_plot, anomaly1_density, label=r"\textbf{Anomaly1 Density}", color="red", alpha=0.5)
    # plt.bar(x_plot, anomaly2_density, color="red", alpha=0.5)
    plt.xlabel("Time Index")
    plt.ylabel("Density")
    plt.title("Detection Density and Anomaly Density")
    plt.legend()
    plt.grid()
    plt.show()