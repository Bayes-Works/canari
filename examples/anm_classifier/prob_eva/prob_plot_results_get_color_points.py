import ast
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import matplotlib.pyplot as plt

def _get_color_points(
    test_ts_len: int,
    csv_path: str,
    *,
    detection_col: str = "anomaly_detected_index",
    anm1_col: str = "anomaly_start_index1",
    anm2_col: str = "anomaly_start_index2",
    magnitude_col: str = "anomaly_magnitude",
    itv_log_col: str = "intervention_log",
    true_LL_baseline_col: str = "true_LL_baseline",
    true_LT_baseline_col: str = "true_LT_baseline",
    estimated_LL_baseline_col: str = "estimated_LL_baseline",
    estimated_LT_baseline_col: str = "estimated_LT_baseline",
    SKF = False,
):
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
    df["estimated_LL_baseline"] = df["estimated_LL_baseline"].apply(
        lambda x: ast.literal_eval(x.replace("nan", "None")))
    df[estimated_LL_baseline_col] = df[estimated_LL_baseline_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["estimated_LT_baseline"] = df["estimated_LT_baseline"].apply(
        lambda x: ast.literal_eval(x.replace("nan", "None")))
    df[estimated_LT_baseline_col] = df[estimated_LT_baseline_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    # Output columns
    df["detection_index_after_anm1"] = pd.Series([None] * len(df), dtype=object)
    df["first_anm_detect_index"] = pd.Series([None] * len(df), dtype=object)
    df["detection_time"] = pd.Series([None] * len(df), dtype=object)

    if SKF:
        # Collapse the consecutive detected indices to the first index
        for idx, row in df.iterrows():
            detected_indices = np.array(row[detection_col])
            if len(detected_indices) != 0:
                detected_indices = detected_indices[np.insert(np.diff(detected_indices) != 1, 0, True)]
            detected_indices = detected_indices.tolist()
            # Replace the original list with the aggregated list
            df.at[idx, detection_col] = detected_indices

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
                    # Remove the itv_time from gray points if exists
                    if (itv_time, idx) in gray_points:
                        gray_points.remove((itv_time, idx))
                elif row[itv_log_col][idx_itv] == 1:
                    orange_points.append((itv_time, idx))
                    # Remove the itv_time from gray points if exists
                    if (itv_time, idx) in gray_points:
                        gray_points.remove((itv_time, idx))

    return red_points, gray_points, blue_points, orange_points