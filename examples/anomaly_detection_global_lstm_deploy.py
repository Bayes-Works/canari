"""Filter a full raw time series with a saved, fine-tuned LSTM SKF.

Run this example from the repository root after training the model with
``examples/anomaly_detection_global_lstm.py``:

    python examples/anomaly_detection_global_lstm_deploy.py

All input and output settings are constants below; no configuration file is read.
"""

import pickle
from pathlib import Path

import pandas as pd

from canari import DataProcess, SKF

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/BM_detrend_data/weekly/weekly_values_raw.csv"
DATETIME_PATH = ROOT / "data/BM_detrend_data/weekly/weekly_datetimes_raw.csv"
SAVED_SKF_PATH = ROOT / "saved_params/anomaly_detection_lstm_finetuned.pkl"
OUTPUT_PATH = ROOT / "saved_results/anomaly_detection_lstm_filtered.csv"

SERIES_NAME = "ts1"
BASELINE_INIT_LEN = 2 * 52


def read_time_series():
    values = pd.read_csv(DATA_PATH)[SERIES_NAME]
    datetimes = pd.to_datetime(pd.read_csv(DATETIME_PATH)[SERIES_NAME])

    last_valid_index = values.last_valid_index()
    values = values.iloc[: last_valid_index + 1]
    datetimes = datetimes.iloc[: last_valid_index + 1]
    return pd.DataFrame(
        {"values": values.to_numpy()},
        index=pd.DatetimeIndex(datetimes, name="date_time"),
    )


def prepare_full_series(dataframe, preprocessing):
    data_processor = DataProcess(
        data=dataframe,
        time_covariates=preprocessing["time_covariates"],
        train_split=1.0,
        validation_split=0.0,
        output_col=preprocessing["output_col"],
        standardization=preprocessing["standardization"],
        scale_const_mean=preprocessing["scale_const_mean"],
        scale_const_std=preprocessing["scale_const_std"],
    )
    _, _, _, all_data = data_processor.get_splits()
    return all_data


def main():
    dataframe = read_time_series()
    with SAVED_SKF_PATH.open("rb") as file:
        saved_model = pickle.load(file)

    all_data = prepare_full_series(dataframe, saved_model["preprocessing"])
    skf = SKF.load_dict(saved_model)
    skf.model["norm_norm"].lstm_net.teacher_forcing = False
    skf.auto_initialize_baseline_states(all_data["y"][:BASELINE_INIT_LEN])
    anomaly_probability, _ = skf.filter(data=all_data)
    threshold = float(saved_model["threshold"])

    results = pd.DataFrame(
        {
            "values": dataframe["values"].to_numpy(),
            "anomaly_probability": anomaly_probability,
            "is_anomaly": anomaly_probability > threshold,
        },
        index=dataframe.index,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH)

    print(f"Loaded complete fine-tuned SKF model from {SAVED_SKF_PATH}")
    print(f"Filtered {len(results)} observations with threshold {threshold:.3f}")
    print(f"Detected {int(results['is_anomaly'].sum())} anomalous observations")
    print(f"Saved filtering results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
