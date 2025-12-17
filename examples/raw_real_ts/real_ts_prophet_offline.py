import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot
import numpy as np

import os
os.environ['OMP_NUM_THREADS'] = '1'

##################################### TS 1 #####################################
# data_file = "./data/benchmark_data/test_1_data.csv"
# df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df_raw.iloc[:, 0])
# # Set the first column name to "ds"
# df_raw.columns = ['ds', 'y']
# df_raw = df_raw.iloc[:int(len(df_raw) * 1)]
# threshold = 0.1

##################################### TS 2 #####################################
data_file = "./data/benchmark_data/test_2_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=";", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 4])
df_raw = df_raw.iloc[:, 6].to_frame()
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["y"]
df_raw = df_raw.resample("W").mean()
df_raw = df_raw.iloc[30:, :]
df_raw = df_raw.reset_index().rename(columns={"date_time": "ds"})
threshold = 0.3
threshold = 0.1

##################################### TS 3 #####################################
# data_file = "./data/benchmark_data/test_11_data.csv"
# df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df_raw.iloc[:, 0])
# df_raw = df_raw.iloc[:, 1:]
# df_raw.index = time_series
# df_raw.index.name = "date_time"
# df_raw.columns = ["y"]
# df_raw = df_raw.reset_index().rename(columns={"date_time": "ds"})
# df_raw = df_raw.iloc[:int(len(df_raw) * 1)]
# threshold = 0.3

##################################### TS 4 #####################################
# data_file = "./data/benchmark_data/test_4_data.csv"
# df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df_raw.iloc[:, 0])
# df_raw = df_raw.iloc[:, 1:]
# df_raw.index = time_series
# df_raw.index.name = "date_time"
# df_raw.columns = ["y", "water_level", "temp_min", "temp_max"]
# df_raw = df_raw.iloc[:, :-3]
# df_raw = df_raw.reset_index().rename(columns={"date_time": "ds"})
# threshold = 0.2

##################################### TS 5 #####################################
# data_file = "./data/benchmark_data/test_5_data.csv"
# df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df_raw.iloc[:, 0])
# df_raw = df_raw.iloc[:, 1:]
# df_raw.index = time_series
# df_raw.index.name = "date_time"
# df_raw.columns = ["y", "water_level", "temp_min", "temp_max"]
# df_raw = df_raw.iloc[:, :-3]
# df_raw = df_raw.reset_index().rename(columns={"date_time": "ds"})
# threshold = 0.1

##################################### TS 6 #####################################
# data_file = "./data/benchmark_data/test_6_data.csv"
# df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df_raw.iloc[:, 0])
# df_raw = df_raw.iloc[:, 1:]
# # Remove the first 52 rows
# df_raw.index = time_series
# df_raw.index.name = "date_time"
# df_raw.columns = ["y", "water_level", "temp_min", "temp_max"]
# df_raw = df_raw.iloc[:, :-3]
# df_raw = df_raw.iloc[52:, :]
# df_raw = df_raw.reset_index().rename(columns={"date_time": "ds"})
# threshold = 0.2

##################################### TS 7 #####################################
# data_file = "./data/benchmark_data/test_7_data.csv"
# df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(df_raw.iloc[:, 0])
# df_raw = df_raw.iloc[:, 1:]
# df_raw.index = time_series
# df_raw.index.name = "date_time"
# df_raw.columns = ["y", "water_level", "temp_min", "temp_max"]
# df_raw = df_raw.iloc[:, :-3]
# df_raw = df_raw.reset_index().rename(columns={"date_time": "ds"})
# threshold = 0.6

m = Prophet(changepoint_range=1)
m.fit(df_raw)
forecast = m.predict(df_raw)

fig1 = m.plot(forecast)
a = add_changepoints_to_plot(fig1.gca(), m, forecast, threshold=threshold)
fig2 = m.plot_components(forecast)
plt.show()