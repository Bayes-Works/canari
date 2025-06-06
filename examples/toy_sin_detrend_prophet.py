import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot


# # # Read data
data_file_time = "./data/toy_time_series/sine_datetime.csv"
df_raw = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
df_raw[0] = pd.to_datetime(df_raw[0]).dt.strftime('%Y-%m-%d %H:%M:%S')
df_raw.columns = ["ds"]

data_file = "./data/toy_time_series/sine.csv"
values = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
df_raw["y"] = values
df_raw.head()

anm_baseline = np.linspace(0, 2, num=len(df_raw))
time_anomaly = 120
new_trend = np.linspace(0, -1, num=len(df_raw) - time_anomaly)
anm_baseline[time_anomaly:] = anm_baseline[time_anomaly:] + new_trend
df_raw["y"] = df_raw["y"].add(anm_baseline, axis=0)

m = Prophet(changepoint_range=1)
m.fit(df_raw)
forecast = m.predict(df_raw)

trend = forecast[["trend"]]
seasonal = forecast[["daily"]]
residual = df_raw["y"].sub(trend["trend"], axis=0).sub(seasonal["daily"], axis=0)

df_detrend = df_raw.copy()
df_detrend["y"] = df_detrend["y"] - trend["trend"]
max_value = df_detrend["y"].max()
min_value = df_detrend["y"].min()

fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
axs[0].plot(df_raw["y"], label="Raw Data")
axs[0].plot(df_detrend["y"], label="Detrended Data")
# Plot a horizontal line at y=0
axs[0].axhline(0, color='red', linestyle='--')
axs[0].fill_between(df_detrend.index, min_value, max_value, color='red', alpha=0.1)
axs[0].legend()
axs[0].set_title("Prophet detrending")
axs[1].plot(trend)
axs[1].set_ylabel("Trend")
axs[2].plot(seasonal)
axs[2].set_ylabel("Seasonal")
axs[3].plot(residual)
axs[3].set_ylabel("Residual")
plt.show()
