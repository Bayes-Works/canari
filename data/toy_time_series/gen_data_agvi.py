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
# df_raw = df_raw.add(linear_space, axis=0)
linear_space = np.linspace(0, 2, num=len(df_raw))
noise = np.random.normal(loc=0.0, scale=0.1, size=len(df_raw))
sine = df_raw.values
data_sine = sine.flatten() + noise
data_exp_sine = np.exp(data_sine) + linear_space


df = pd.DataFrame({"data_exp_sine": data_exp_sine})
df.to_csv("data/toy_time_series/exp_sine_agvi.csv", index=False)

plt.plot(data_exp_sine)
plt.show()
