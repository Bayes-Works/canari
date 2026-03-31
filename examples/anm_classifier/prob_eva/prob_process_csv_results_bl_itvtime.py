import ast
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import matplotlib.pyplot as plt


def _process_detection_df_bl_itvtime(
    test_ts_len: int,
    csv_path: str,
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
    df = pd.read_csv(csv_path)

    # Keep anomaly magnitude as abs for LL anomaly
    if magnitude_col in df.columns:
        df[magnitude_col] = np.abs(df[magnitude_col])

    # Parse detected indices list
    if detection_col in df.columns:
        df[detection_col] = df[detection_col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    if itv_log_col in df.columns:
        df[itv_log_col] = df[itv_log_col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df["intervention_applied_times"] = df["intervention_applied_times"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    df[true_LL_baseline_col] = df[true_LL_baseline_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df[true_LT_baseline_col] = df[true_LT_baseline_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df[estimated_LL_baseline_col] = df[estimated_LL_baseline_col].apply(
        lambda x: ast.literal_eval(x.replace("nan", "None")))
    df[estimated_LT_baseline_col] = df[estimated_LT_baseline_col].apply(
        lambda x: ast.literal_eval(x.replace("nan", "None")))
    # Output columns
    df["detection_index_after_anm1"] = pd.Series([None] * len(df), dtype=object)
    df["first_anm_detect_index"] = pd.Series([None] * len(df), dtype=object)
    df["detection_time"] = pd.Series([None] * len(df), dtype=object)
    df["intervention_time"] = pd.Series([None] * len(df), dtype=object)

    default_detection_time = freq_per_year * default_years + 1

    true_detection = []
    false_detection = []
    first_classification = []

    # Compute the sum of mse after anomaly_start_index1 for LL and LT baselines for each row, and add two new columns "mse_LL" and "mse_LT" to the dataframe
    # Remove rows with None in true_LL_baseline_col or estimated_LL_baseline_col for mse_LL calculation, and similarly for mse_LT calculation
    df["mse_LL"] = df.apply(
        lambda row: np.nanmean(
            (np.array(row[true_LL_baseline_col], dtype=float)[row[anm1_col]:] -
            np.array(row[estimated_LL_baseline_col], dtype=float)[row[anm1_col]:]) ** 2
        ),
        axis=1
    )

    df["mse_LT"] = df.apply(
        lambda row: np.nanmean(
            (np.array(row[true_LT_baseline_col], dtype=float)[row[anm1_col]:] -
            np.array(row[estimated_LT_baseline_col], dtype=float)[row[anm1_col]:]) ** 2
        ),
        axis=1
    )
    # Plot the detection map
    if plot_detection_map:
        plt.figure(figsize=(10, 6))
        for idx, row in df.iterrows():
            anm1_start = row[anm1_col]
            anm2_start = row[anm2_col]
            detected_indices = row[detection_col]

            # Safety: handle missing / NaN / non-list
            if not isinstance(detected_indices, list) or pd.isna(anm1_start) or pd.isna(anm2_start):
                continue

            plt.scatter(detected_indices, [idx] * len(detected_indices), label="Detected", color="grey", marker="o", alpha=0.5, s=5)
            plt.scatter(anm1_start, idx, label="Anomaly 1 Start", color="red", marker="o", s=5)
            plt.scatter(anm2_start, idx, label="Anomaly 2 Start", color="red", marker="o", s=5)
            # Plot intervention log if available
            if itv_log_col in row and isinstance(row[itv_log_col], list):
                for idx_itv, itv_time in enumerate(row["intervention_applied_times"]):
                    if row[itv_log_col][idx_itv] == 0:
                        plt.scatter(itv_time, idx, label="Intervention", color="tab:blue", marker="o", s=5)
                    elif row[itv_log_col][idx_itv] == 1:
                        plt.scatter(itv_time, idx, label="Intervention", color="tab:orange", marker="o", s=5)

        plt.xlabel("Time Index")
        plt.ylabel("Sample Index")
        plt.title("Detection Map")
        # Add horizontal grid lines for each index
        plt.yticks(np.arange(0, len(df), 10))
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        # plt.show()

        # _plot_detection_bars(df, anm1_col, anm2_col, detection_col, itv_log_col, test_ts_len)

    for idx, row in df.iterrows():
        anm1_start = row[anm1_col]
        anm2_start = row[anm2_col]
        detected_indices = row[detection_col]

        # Safety: handle missing / NaN / non-list
        if not isinstance(detected_indices, list) or pd.isna(anm1_start) or pd.isna(anm2_start):
            continue

        # detections between anomaly1 start and anomaly2 start
        between = [d for d in detected_indices if (d >= anm1_start) and (d < anm2_start)]
        if len(between) == 0:
            # first anomaly not detected
            df.at[idx, "first_anm_detect_index"] = None
            df.at[idx, "detection_index_after_anm1"] = None
            df.at[idx, "detection_time"] = None
            df.at[idx, "intervention_time"] = None
            continue

        first_anm_detect_index = min(between)
        df.at[idx, "first_anm_detect_index"] = first_anm_detect_index

        if evaluate_itv_type:
            if first_anm_detect_index in row["intervention_applied_times"]:
                itv_idx = row["intervention_applied_times"].index(first_anm_detect_index)
                first_classification.append(row[itv_log_col][itv_idx])
            else:
                first_classification.append(2)

        # Remove <= first_anm_detect_index
        after = [d for d in detected_indices if d > first_anm_detect_index]
        df.at[idx, "detection_index_after_anm1"] = after

        # Find first detection >= anomaly2 start
        detection_time = default_detection_time
        for d in after:
            if d >= anm2_start:
                detection_time = d - anm2_start
                if evaluate_itv_type:
                    # Only consider the first detection matches the first anomaly type
                    if first_classification[-1] == (0 if first_anm_type == 'll' else 1):
                        # Evaluate the classification among true detections
                        if d in row["intervention_applied_times"]:
                            itv_idx = row["intervention_applied_times"].index(d)
                            true_detection.append(row[itv_log_col][itv_idx])
                        else:
                            true_detection.append(2)

                        # Evaluate the classification among false detections
                        after_false_detections = after.copy()
                        after_false_detections.remove(d)
                        for fd in after_false_detections:
                            if fd in row["intervention_applied_times"]:
                                itv_idx = row["intervention_applied_times"].index(fd)
                                false_detection.append(row[itv_log_col][itv_idx])
                            else:
                                false_detection.append(2)
                break
        df.at[idx, "detection_time"] = detection_time

        # Find first intervention >= anomaly2 start
        intervention_time = 52 * 5 + 1
        for itv_time in row["intervention_applied_times"]:
            if itv_time >= anm2_start:
                intervention_time = itv_time - anm2_start
                break
        df.at[idx, "intervention_time"] = intervention_time

        
    if evaluate_itv_type:
        print('------------- Classification analysis --------------------')
        print('------------- First detection --------------------')
        print("For the detection of the first anomaly, the counts are as follows:")
        unique, counts = np.unique(first_classification, return_counts=True)
        first_classification_counts = dict(zip(unique, counts))
        sum_count = sum(first_classification_counts.values())
        # Change keys from 0 -> 'LL itv', 1 -> 'LT itv', 2 -> 'non-identified'
        first_classification_counts = {
            ( 'LL itv' if k == 0 else 'LT itv' if k == 1 else 'non-identified' ): round(v / sum_count, 2)
            for k, v in first_classification_counts.items()}
        print(first_classification_counts)
        print('------------- Second detection --------------------')
        print("Among all the true detections, the counts are as follows:")
        # 0: LL itv, 1: LT itv, 2: non-identified
        unique, counts = np.unique(true_detection, return_counts=True)
        detection_type_counts = dict(zip(unique, counts))
        sum_count = sum(detection_type_counts.values())
        # Change keys from 0 -> 'LL itv', 1 -> 'LT itv', 2 -> 'non-identified'
        detection_type_counts = {
            ( 'LL itv' if k == 0 else 'LT itv' if k == 1 else 'non-identified' ): round(v / sum_count, 2)
            for k, v in detection_type_counts.items()}
        print(detection_type_counts)
        print("Among all the false detections, the counts are as follows:")
        unique, counts = np.unique(false_detection, return_counts=True)
        false_detection_type_counts = dict(zip(unique, counts))
        sum_count = sum(false_detection_type_counts.values())
        false_detection_type_counts = {
            ( 'LL itv' if k == 0 else 'LT itv' if k == 1 else 'non-identified' ): round(v / sum_count, 2)
            for k, v in false_detection_type_counts.items()}
        print(false_detection_type_counts)
        print('----------------------------------------------------------')

    df = df[df["detection_time"].notnull()]
    df["detection_time"] = df["detection_time"].astype(int)
    df["intervention_time"] = pd.to_numeric(df["intervention_time"])
    # print("Total time to intervention: ", round(df["intervention_time"].sum()/52, 2), " years")
    print("Time to intervention: ", round(df["intervention_time"].mean()/52, 2), " ± ", round(df["intervention_time"].std()/52, 2), " years")

    df["detection_rate"] = df["detection_time"].apply(
                                    lambda x: 0 if x >= 52 * 3 else 1
                                )
    
    # Compute the false alarm rate for each method
    all_time_after_anm1 = (test_ts_len - df["first_anm_detect_index"] - 1).sum()
    false_alarm = np.sum(df["detection_index_after_anm1"].apply(lambda x: len(x))) - df["detection_rate"].sum()
    false_alarm_rate = round(false_alarm * 10 / (all_time_after_anm1/52), 2)

    df["alarms_num"] = df["anomaly_detected_index"].apply(
                                    lambda x: len(x) if len(x) > 0 else 0
                                )
    
    df_group = df.groupby("anomaly_magnitude").agg(
    {
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
        "intervention_time": ["mean", "std"],
    }
    )

    # Sum the column df["mse_LL"] and df["mse_LT"] for all rows
    total_mse_LL = df["mse_LL"].sum()
    total_mse_LT = df["mse_LT"].sum()
    print("Total MSE for LL baseline: ", total_mse_LL)
    print("Total MSE for LT baseline: ", total_mse_LT)

    return false_alarm_rate, df_group

def _plot_detection_bars(df, anm1_col, anm2_col, detection_col, itv_log_col, test_ts_len):
    # Collect all (time, sample_idx) points per category
    red_points = []
    gray_points = []
    blue_points = []
    orange_points = []

    for idx, row in df.iterrows():
        anm1_start = row[anm1_col]
        anm2_start = row[anm2_col]
        detected_indices = row[detection_col]

        if not isinstance(detected_indices, list) or pd.isna(anm1_start) or pd.isna(anm2_start):
            continue

        # Gray: detected
        for t in detected_indices:
            gray_points.append((t, idx))

        # Red: anomaly starts
        red_points.append((anm1_start, idx))
        red_points.append((anm2_start, idx))

        # Blue / Orange: interventions
        if itv_log_col in row and isinstance(row[itv_log_col], list):
            for idx_itv, itv_time in enumerate(row["intervention_applied_times"]):
                if row[itv_log_col][idx_itv] == 0:
                    blue_points.append((itv_time, idx))
                elif row[itv_log_col][idx_itv] == 1:
                    orange_points.append((itv_time, idx))

    # --- Build the figure with 2 horizontal bar subplots ---
    solo_categories = [
        ("Anomaly Start", red_points, "red"),
    ]
    combined_categories = [
        ("Detected",        gray_points,   "gray"),
        ("LL Intervention", blue_points,   "tab:blue"),
        ("LT Intervention", orange_points, "tab:orange"),
    ]

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8, 1.5),
        sharex=True,
        gridspec_kw={"hspace": 0.3, "height_ratios": [1, 1]}
    )

    # Determine x range
    all_times = [p[0] for cat in solo_categories + combined_categories for p in cat[1]]
    x_min, x_max = min(all_times), max(all_times)

    # Resolution for the density bar (pixels along x)
    x_bins = np.linspace(x_min, x_max, 700)
    bin_width = x_bins[1] - x_bins[0]

    # --- Subplot 0: Red only ---
    ax = axes[0]
    times = np.array([p[0] for p in red_points])
    counts, edges = np.histogram(times, bins=x_bins)
    norm_counts = counts / counts.max() if counts.max() > 0 else counts
    for val, left in zip(norm_counts, edges[:-1]):
        if val > 0:
            ax.axvspan(left, left + bin_width, ymin=0, ymax=1,
                       color="red", alpha=float(val) * 0.85)
    ax.set_yticks([])
    # ax.set_ylabel("Anomaly Start", rotation=0, labelpad=100, va="center", fontsize=9)
    ax.set_xlim(0, test_ts_len)
    ax.spines[["top", "right", "left"]].set_visible(False)

    # --- Subplot 1: Gray + Blue + Orange combined ---
    ax = axes[1]
    combined_label_parts = []
    for label, points, color in combined_categories:
        if not points:
            continue
        times = np.array([p[0] for p in points])
        counts, edges = np.histogram(times, bins=x_bins)
        norm_counts = counts / counts.max() if counts.max() > 0 else counts
        for val, left in zip(norm_counts, edges[:-1]):
            if val > 0:
                ax.axvspan(left, left + bin_width, ymin=0, ymax=1,
                           color=color, alpha=float(val) * 0.5)
        combined_label_parts.append(label)
    ax.set_yticks([])
    # ax.set_ylabel("Detection", rotation=0, labelpad=100, va="center", fontsize=9)
    ax.set_xlim(0, test_ts_len)
    ax.spines[["top", "right", "left"]].set_visible(False)

    # axes[-1].set_xlabel("Time Index")
    fig.suptitle("Detection Map")
    
    # Fix title + label clipping
    plt.tight_layout()
    plt.subplots_adjust(left=0.06, top=0.88, bottom=0.15)