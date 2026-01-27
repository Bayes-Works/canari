import ast
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def _process_detection_df(
    test_ts_len: int,
    csv_path: str,
    *,
    evaluate_itv_type: Optional[bool] = False,
    detection_col: str = "anomaly_detected_index",
    anm1_col: str = "anomaly_start_index1",
    anm2_col: str = "anomaly_start_index2",
    magnitude_col: str = "anomaly_magnitude",
    itv_log_col: str = "intervention_log",
    default_years: int = 3,
    freq_per_year: int = 52,
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
    # Output columns
    df["detection_index_after_anm1"] = pd.Series([None] * len(df), dtype=object)
    df["first_anm_detect_index"] = pd.Series([None] * len(df), dtype=object)
    df["detection_time"] = pd.Series([None] * len(df), dtype=object)

    default_detection_time = freq_per_year * default_years + 1

    true_detection = []
    false_detection = []

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
            continue

        first_anm_detect_index = min(between)
        df.at[idx, "first_anm_detect_index"] = first_anm_detect_index

        # Remove <= first_anm_detect_index
        after = [d for d in detected_indices if d > first_anm_detect_index]
        df.at[idx, "detection_index_after_anm1"] = after

        # Find first detection >= anomaly2 start
        detection_time = default_detection_time
        for d in after:
            if d >= anm2_start:
                detection_time = d - anm2_start
                if evaluate_itv_type:
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

    if evaluate_itv_type:
        print('------------- Classification analysis --------------------')
        print("In the true detections, the counts are as follows:")
        # 0: LL itv, 1: LT itv, 2: non-identified
        unique, counts = np.unique(true_detection, return_counts=True)
        detection_type_counts = dict(zip(unique, counts))
        sum_count = sum(detection_type_counts.values())
        # Change keys from 0 -> 'LL itv', 1 -> 'LT itv', 2 -> 'non-identified'
        detection_type_counts = {
            ( 'LL itv' if k == 0 else 'LT itv' if k == 1 else 'non-identified' ): round(v / sum_count, 2)
            for k, v in detection_type_counts.items()}
        print(detection_type_counts)
        print("In the false detections, the counts are as follows:")
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
    }
    ).reset_index()

    return false_alarm_rate, df_group