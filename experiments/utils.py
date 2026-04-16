import pandas as pd
import numpy as np

from canari import DataProcess
from pytagi import Normalizer as normalizer


def _apply_train_split_via_part_to_remove(
    df_raw_full: pd.DataFrame,
    train_split: float,
    validation_start: str,
    warmup_len: int,
):
    validation_ts = pd.Timestamp(validation_start)
    n_train_total = int(df_raw_full.index.searchsorted(validation_ts, side="left"))
    if n_train_total <= warmup_len:
        raise ValueError("Training window is too short after preprocessing.")

    n_train_use = max(int(train_split * n_train_total), warmup_len + 1)
    n_train_use = min(n_train_use, n_train_total)
    part_to_remove = n_train_total - n_train_use

    df_subset = df_raw_full.iloc[part_to_remove:].copy()
    return df_subset, n_train_total, n_train_use, part_to_remove


def _load_base_dataframe(experiment_config: dict) -> pd.DataFrame:
    df = pd.read_csv(experiment_config["data_file"])

    if "datetime_file" in experiment_config:
        time_series = pd.read_csv(experiment_config["datetime_file"])
        datetime_column = experiment_config.get("datetime_column", time_series.columns[0])
        df.index = pd.to_datetime(time_series[datetime_column])
    else:
        date_column = experiment_config.get("date_column")
        if date_column is None:
            for candidate in ("Date", "date", "date_time", "datetime"):
                if candidate in df.columns:
                    date_column = candidate
                    break
        if date_column is None:
            raise ValueError(
                "Could not infer a datetime column. Set `date_column` or `datetime_file` in the config."
            )
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)

    value_column = experiment_config.get("value_column")
    if value_column is not None:
        df = df[[value_column]].copy()

    df.index.name = "date_time"

    df.index = df.index - pd.Timedelta(weeks=1)

    return df


def _trim_trailing_target_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Remove trailing rows where the target series is NaN."""

    if df.empty:
        return df

    target_series = df.iloc[:, 0]
    valid_positions = np.flatnonzero(~pd.isna(target_series.to_numpy()))
    if valid_positions.size == 0:
        raise ValueError("Input time series contains only NaN values.")

    last_valid_pos = int(valid_positions[-1])
    return df.iloc[: last_valid_pos + 1].copy()


def _resolve_time_index(index: pd.DatetimeIndex, time_start_time: str) -> int:
    target_time = pd.Timestamp(time_start_time)
    time_idx = int(index.searchsorted(target_time, side="left"))

    if time_idx <= 0:
        return 0
    if time_idx >= len(index):
        return len(index) - 1

    prev_idx = time_idx - 1
    prev_delta = abs(index[prev_idx] - target_time)
    next_delta = abs(index[time_idx] - target_time)
    if prev_delta <= next_delta:
        return prev_idx
    return time_idx


def _inject_synthetic_anomaly(
    df: pd.DataFrame,
    anomaly_end_offset: float,
    anomaly_idx: int,
) -> tuple[pd.DataFrame, int]:
    """Inject one anomaly through DataProcess.add_synthetic_anomaly at a fixed timestep.

    `DataProcess.add_synthetic_anomaly` expects a per-step slope. The experiment configs
    currently express `anomaly_slope` as the final offset reached at the end of the
    series, so convert that offset into the equivalent per-step slope before injecting.
    """

    num_steps = len(df)
    if num_steps == 0:
        raise ValueError("Cannot inject an anomaly into an empty dataframe.")

    anomaly_idx = int(np.clip(anomaly_idx, 0, num_steps - 1))
    remaining_steps = max(num_steps - anomaly_idx - 1, 0)
    if remaining_steps > 0:
        slope_per_step = anomaly_end_offset / remaining_steps
    else:
        slope_per_step = 0.0

    anomaly_start = anomaly_idx / num_steps
    anomaly_end = min((anomaly_idx + 1) / num_steps, 1.0)
    injected = DataProcess.add_synthetic_anomaly(
        data={"y": df.iloc[:, 0].to_numpy(dtype=float).reshape(-1, 1)},
        num_samples=1,
        slope=[slope_per_step],
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end,
    )[0]

    df_with_anomaly = df.copy()
    df_with_anomaly.iloc[:, 0] = injected["y"].reshape(-1)
    return df_with_anomaly, int(injected["anomaly_timestep"])


def prepare_dataset(
    train_split: float,
    anomaly_slope: float,
    experiment_config: dict,
):

    warmup_len = int(experiment_config["global_warmup_lookback_len"])
    df_original = _load_base_dataframe(experiment_config)
    df_original = _trim_trailing_target_nans(df_original)

    # resolve dates if needed
    validation_idx = _resolve_time_index(
        df_original.index, experiment_config["validation_start"]
    )
    validation_date_time = df_original.index[validation_idx]
    test_idx = _resolve_time_index(df_original.index, experiment_config["test_start"])
    test_date_time = df_original.index[test_idx]

    # Fit scaler on full training set (before part removal)
    data_pro_scale = DataProcess(
        data=df_original,
        time_covariates=experiment_config.get("time_covariates", ["week_of_year"]),
        validation_start=experiment_config["validation_start"],
        test_start=experiment_config["test_start"],
        output_col=[0],
    )

    # Appply split with removing parts of train logic
    df_raw, n_train_total, n_train_use, part_to_remove = (
        _apply_train_split_via_part_to_remove(
            df_original,
            train_split,
            experiment_config["validation_start"],
            warmup_len,
        )
    )

    # Warmup lookback preparation
    warmup_lookback_mu = None
    warmup_lookback_var = None
    if len(df_raw) <= warmup_len:
        raise ValueError(
            f"Not enough data after part removal to create a {warmup_len}-step warmup."
        )
    df_warmup = df_raw.iloc[:warmup_len].copy()
    df_raw = df_raw.iloc[warmup_len:].copy()
    time_anomaly = _resolve_time_index(
        df_raw.index, experiment_config["anomaly_start_time"]
    )
    warmup_lookback_mu = df_warmup.iloc[:, 0].values.flatten()
    warmup_lookback_mu = normalizer.standardize(
        warmup_lookback_mu,
        data_pro_scale.scale_const_mean[data_pro_scale.output_col],
        data_pro_scale.scale_const_std[data_pro_scale.output_col],
    )
    warmup_lookback_var = np.ones_like(warmup_lookback_mu) * 0.1

    # Inject the evaluation anomaly through the shared synthetic-anomaly helper.
    df, time_anomaly = _inject_synthetic_anomaly(
        df=df_raw,
        anomaly_end_offset=anomaly_slope,
        anomaly_idx=time_anomaly,
    )

    df_plot_full = df_original.iloc[warmup_len:].copy()
    plot_time_anomaly = _resolve_time_index(
        df_plot_full.index, experiment_config["anomaly_start_time"]
    )
    df_plot_full, plot_time_anomaly = _inject_synthetic_anomaly(
        df=df_plot_full,
        anomaly_end_offset=anomaly_slope,
        anomaly_idx=plot_time_anomaly,
    )

    # Data processor preparation
    data_processor = DataProcess(
        data=df,
        time_covariates=experiment_config.get("time_covariates", ["week_of_year"]),
        validation_start=validation_date_time,
        test_start=test_date_time,
        output_col=[0],
        scale_const_mean=data_pro_scale.scale_const_mean,
        scale_const_std=data_pro_scale.scale_const_std,
    )
    plot_data_processor = DataProcess(
        data=df_plot_full,
        time_covariates=experiment_config.get("time_covariates", ["week_of_year"]),
        validation_start=validation_date_time,
        test_start=test_date_time,
        output_col=[0],
        scale_const_mean=data_pro_scale.scale_const_mean,
        scale_const_std=data_pro_scale.scale_const_std,
    )
    train_data, validation_data, test_data, all_data = data_processor.get_splits()

    train_val = data_processor.get_splits(split="train_val")

    return {
        "data_processor": data_processor,
        "plot_data_processor": plot_data_processor,
        "train_data": train_data,
        "validation_data": validation_data,
        "test_data": test_data,
        "all_data": all_data,
        "warmup_time": df_warmup.index.to_numpy(),
        "warmup_values": df_warmup.iloc[:, 0].to_numpy(dtype=float),
        "warmup_lookback_mu": warmup_lookback_mu,
        "warmup_lookback_var": warmup_lookback_var,
        "anomaly_idx": time_anomaly,
        "anomaly_time": df.index[time_anomaly],
        "plot_anomaly_time": df_plot_full.index[plot_time_anomaly],
        "train_rows_full": n_train_total,
        "train_rows_used": n_train_use,
        "unused_train_rows": part_to_remove,
        "metadata": {
            "time_anomaly": time_anomaly,
            "plot_time_anomaly": plot_time_anomaly,
        },
        "train_val": train_val,
    }
