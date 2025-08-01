import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

sine = df_raw.values
exp_sine = np.exp(sine)
linear_space = np.linspace(0, 2, num=len(df_raw))
df_raw = df_raw.add(linear_space, axis=0)
noise = np.random.normal(loc=0.0, scale=0.05, size=len(df_raw))
noise_1 = np.random.normal(loc=0.0, scale=0.05, size=len(df_raw))
data_exp_sine = exp_sine.flatten() + linear_space + noise
data_sine = sine.flatten() + noise_1

df = pd.DataFrame({"data_exp_sine": data_exp_sine, "data_sine": data_sine})
df.to_csv("data/toy_time_series/toy_time_series_dependency.csv", index=False)

plt.plot(data_exp_sine)
plt.show()

plt.plot(data_sine)
plt.show()

data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]
