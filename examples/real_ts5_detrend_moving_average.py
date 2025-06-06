import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


# # Read data
data_file = "./data/benchmark_data/test_5_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]


# Detrending data
df_detrend = df_raw.copy()
df_detrend = df_detrend.interpolate()
decomposition = seasonal_decompose(df_detrend["values"], model="additive", period=52)
trend = decomposition.trend.to_frame()
seasonal = decomposition.seasonal.to_frame()
residual = decomposition.resid.to_frame()
residual = residual.fillna(0)

df_detrend.values = seasonal.values + residual.values
# Find the maximum and minimum pf the detrended data
max_value = df_detrend["values"].max()
min_value = df_detrend["values"].min()

fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
axs[0].plot(df_raw["values"], label="Raw Data")
axs[0].plot(df_detrend["values"], label="Detrended Data")
axs[0].axhline(0, color='red', linestyle='--')
axs[0].fill_between(df_detrend.index, min_value, max_value, color='red', alpha=0.1)
axs[0].legend()
axs[0].set_title("Moving average detrending")
axs[1].plot(trend)
axs[1].set_ylabel("Trend")
axs[2].plot(seasonal)
axs[2].set_ylabel("Seasonal")
axs[3].plot(residual)
axs[3].set_ylabel("Residual")
plt.show()
