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
linear_space = np.linspace(0, 2, num=len(df_raw))

noise = np.random.normal(loc=0.0, scale=0.06, size=len(df_raw))
sine = df_raw.values.flatten() + noise
exp_sine = np.exp(sine) + linear_space

rand_ints = np.random.randint(0, len(sine), size=int(len(sine) / 6))
sine[rand_ints] = np.nan
df = pd.DataFrame({"data_exp_sine": exp_sine, "data_sine": sine})
df.to_csv("data/toy_time_series/exp_sine_dependency.csv", index=False)

plt.plot(exp_sine)
plt.show()

plt.plot(sine)
plt.show()
