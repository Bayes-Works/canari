"""Run global-LSTM anomaly detection for multiple time series in parallel.

Run this launcher from the repository root:

    python -m examples.anomaly_detection_global_lstm_parallel

Each series writes to its own log in ``logs/anomaly_detection_global_lstm``.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/BM_detrend_data/weekly/weekly_values.csv"
DATETIME_PATH = ROOT / "data/BM_detrend_data/weekly/weekly_datetimes.csv"
TEST_DATA_PATH = ROOT / "data/BM_detrend_data/weekly/weekly_values_raw.csv"
TEST_DATETIME_PATH = ROOT / "data/BM_detrend_data/weekly/weekly_datetimes_raw.csv"
LOG_DIR = ROOT / "logs/anomaly_detection_global_lstm"

SERIES_NAMES = ["ts51", "ts52", "ts53", "ts54", "ts55", "ts56"]


def validate_series_names():
    if len(SERIES_NAMES) != len(set(SERIES_NAMES)):
        raise ValueError("Series names must be unique.")

    with DATA_PATH.open(newline="") as file:
        value_series = set(next(csv.reader(file)))
    with DATETIME_PATH.open(newline="") as file:
        datetime_series = set(next(csv.reader(file)))
    with TEST_DATA_PATH.open(newline="") as file:
        test_value_series = set(next(csv.reader(file)))
    with TEST_DATETIME_PATH.open(newline="") as file:
        test_datetime_series = set(next(csv.reader(file)))
    available_series = (
        value_series & datetime_series & test_value_series & test_datetime_series
    )

    unknown_series = sorted(set(SERIES_NAMES) - available_series)
    if unknown_series:
        raise ValueError(f"Unknown series: {', '.join(unknown_series)}")


def main():
    validate_series_names()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for series_name in SERIES_NAMES:
        log_path = LOG_DIR / f"{series_name}.log"
        log_file = log_path.open("w")
        environment = os.environ.copy()
        environment["CANARI_SERIES_NAME"] = series_name
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-m", "examples.anomaly_detection_global_lstm"],
            cwd=ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        jobs.append((series_name, process, log_file, log_path))
        print(
            f"Started {series_name} with PID {process.pid}; log: {log_path}",
            flush=True,
        )

    failed_series = []
    try:
        for series_name, process, _, log_path in jobs:
            return_code = process.wait()
            if return_code == 0:
                print(f"Completed {series_name}; log: {log_path}", flush=True)
            else:
                failed_series.append(series_name)
                print(
                    f"Failed {series_name} with exit code {return_code}; "
                    f"log: {log_path}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("Stopping running series...", flush=True)
        for _, process, _, _ in jobs:
            if process.poll() is None:
                process.terminate()
        for _, process, _, _ in jobs:
            process.wait()
        return 130
    finally:
        for _, _, log_file, _ in jobs:
            log_file.close()

    return 1 if failed_series else 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        exit_code = 2
    raise SystemExit(exit_code)
