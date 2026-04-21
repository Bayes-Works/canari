import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chronos import Chronos2Pipeline
from canari import DataProcess

# Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
linear_space = np.linspace(0, 2, num=len(df_raw))
df_raw = df_raw.add(linear_space, axis=0)
data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Resampling data
df = df_raw.resample("H").mean()

# Define parameters
output_col = [0]

# Build data processor
data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day"],
    train_split=0.8,
    validation_split=0.1,
    output_col=output_col,
)

# Split data
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# --- Build context from train + validation using the split dicts directly ---
train_val = data_processor.get_splits(split="train_val")

context_values    = train_val["y"].flatten()          # numpy array, already standardized
context_timestamps = train_val["time"]                # DatetimeIndex from the split

prediction_length = len(test_data["y"])
test_timestamps   = test_data["time"]
test_values       = test_data["y"].flatten()

# --- Load Chronos-2 ---
device = "cpu"
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map=device,
)

# --- Build context DataFrame ---
context_df = pd.DataFrame({
    "id":        ["series_0"] * len(context_values),
    "timestamp": context_timestamps,
    "target":    context_values,
})

# --- Forecast ---
pred_df = pipeline.predict_df(
    context_df,
    prediction_length=prediction_length,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)

forecast_median = pred_df["0.5"].values
forecast_lower  = pred_df["0.1"].values
forecast_upper  = pred_df["0.9"].values

# --- Plot ---
plt.figure(figsize=(12, 5))
plt.plot(context_timestamps, context_values, label="Context (train+val)", color="steelblue")
plt.plot(test_timestamps, test_values, label="Test (ground truth)", color="black", linestyle="--")
plt.plot(test_timestamps, forecast_median, label="Chronos-2 median", color="tomato")
plt.fill_between(
    test_timestamps,
    forecast_lower,
    forecast_upper,
    alpha=0.3,
    color="tomato",
    label="Chronos-2 10–90% interval",
)
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Chronos-2 Zero-Shot Forecast on Test Set")
plt.legend()
plt.tight_layout()
plt.show()

# --- Metrics ---
mae  = np.mean(np.abs(forecast_median - test_values))
rmse = np.sqrt(np.mean((forecast_median - test_values) ** 2))
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")