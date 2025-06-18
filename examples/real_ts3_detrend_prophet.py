import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

data_file = "./data/benchmark_data/test_3_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=";", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 3])
df_raw = df_raw.iloc[:, 6].to_frame()
df_raw.index = time_series
df_raw.index.name = "ds"
df_raw.columns = ["y"]
df_raw = df_raw.resample("W").mean()
# time_series = pd.to_datetime(df_raw.iloc[:, 4])
# df_raw = df_raw.iloc[:, 6].to_frame()
# df_raw.index = time_series
# df_raw.index.name = "ds"
# df_raw.columns = ["y"]
# df_raw = df_raw.resample("W").mean()
# df_raw = df_raw.iloc[30:, :]

df_raw.head()

df_raw = df_raw.reset_index()
df_raw = df_raw[["ds", "y"]]

m = Prophet(changepoint_range=1)
m.fit(df_raw)
forecast = m.predict(df_raw)

trend = forecast[["trend"]]
seasonal = forecast[["yearly"]]
residual = df_raw["y"].sub(trend["trend"], axis=0).sub(seasonal["yearly"], axis=0)

df_detrend = df_raw.copy()
df_detrend["y"] = df_detrend["y"] - trend["trend"]
max_value = df_detrend["y"].max()
min_value = df_detrend["y"].min()

# # Save the detrended data
# df_detrend.to_csv("./data/benchmark_data/detrended_data/test_3_data_detrended.csv", index=False)

fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
axs[0].plot(df_raw['ds'], df_raw["y"], label="Raw Data")
axs[0].plot(df_raw['ds'],df_detrend["y"], label="Detrended Data")
# Plot a horizontal line at y=0
axs[0].axhline(0, color='red', linestyle='--')
axs[0].fill_between(df_raw['ds'], min_value, max_value, color='red', alpha=0.1)
axs[0].legend(loc='upper left')
axs[0].set_title("Prophet detrend: real TS3")
axs[1].plot(df_raw['ds'],trend)
axs[1].set_ylabel("Trend")
axs[2].plot(df_raw['ds'],seasonal)
axs[2].set_ylabel("Seasonal")
axs[3].plot(df_raw['ds'],residual)
axs[3].set_ylabel("Residual")
plt.show()
