import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

# # Load time series data (monthly airline passengers)
# y = load_airline()
# y.index = y.index.to_timestamp()  # Convert PeriodIndex to Timestamp for plotting

# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "Period"
df_raw.columns = ["values"]

# Add synthetic anomaly to data
trend = np.linspace(0, 2, num=len(df_raw))
time_anomaly = 120
new_trend = np.linspace(0, -1, num=len(df_raw) - time_anomaly)
trend[time_anomaly:] = trend[time_anomaly:] + new_trend
df_raw = df_raw.add(trend, axis=0)

# Convert to <class 'pandas.core.series.Series'>
y = df_raw.squeeze()  # Convert DataFrame to Series
y.index.freq = 'H'

print(y)
print(type(y))

# Fit ETS model with automatic configuration
model = ETSModel(
    y,
    error='add',
    trend='add',
    seasonal='add',
    seasonal_periods=24,
    # bounds = {
    #     "smoothing_level": (1e-2, 1e-2),
    #     "smoothing_trend": (1e-2, 1e-2),
    #     # "smoothing_seasonal": (0.1, 0.9),
    #     # "damping_trend": (0.8, 0.98),
    #     # optional: initial_level, initial_trend, initial_seasonal.0, etc.
    # },
    # damped_trend=True
)
fit = model.fit()

# Extract components
trend = fit.level
# trend = fit.slope if fit.model.trend else np.zeros_like(level)
seasonal = fit.season
fittedvalues = fit.fittedvalues
residual = fit.resid

df_detrend = df_raw.copy()
df_detrend.values = df_raw.values - trend.values.reshape(-1, 1)
max_value = df_detrend["values"].max()
min_value = df_detrend["values"].min()

fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
axs[0].plot(df_raw["values"], label="Raw Data")
axs[0].plot(df_detrend["values"], label="Detrended Data")
axs[0].axhline(0, color='red', linestyle='--')
axs[0].fill_between(df_detrend.index, min_value, max_value, color='red', alpha=0.1)
axs[0].legend()
axs[0].set_title("Exponential smoothing detrending")
axs[1].plot(trend)
axs[1].set_ylabel("Trend")
axs[2].plot(seasonal)
axs[2].set_ylabel("Seasonal")
axs[3].plot(residual)
axs[3].set_ylabel("Residual")
plt.show()