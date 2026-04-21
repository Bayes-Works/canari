import copy
import pandas as pd
from pytagi import Normalizer as normalizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_data,
    plot_prediction,
    plot_skf_states,
)
from canari.component import LocalTrend, LocalAcceleration, Chronos, WhiteNoise
import json

# # Read data
with open("examples/benchmark/BM_metadata.json", "r") as f:
    metadata = json.load(f)

benchmark = "2"
# Load configuration from metadata for a specific benchmark
config = metadata[benchmark]
print("----------------------------")
print(f"Benchmark being analyzed: #{benchmark}")
print("----------------------------")

######### Data processing #########
# Read data
data_file = config["data_path"]
df = pd.read_csv(data_file, skiprows=0, delimiter=",")
date_time = pd.to_datetime(df["date"])
df = df.drop("date", axis=1)
df.index = date_time
df.index.name = "date_time"
# Data pre-processing
output_col = config["output_col"]
data_processor = DataProcess(
    data=df,
    train_split=0.15,
    validation_split=0.85,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Components
sigma_v = 0.12
context_len = 80
local_trend = LocalTrend(var_states=[1e-3,1e-7])
local_acceleration = LocalAcceleration()
chronos = Chronos(
        look_back_len=context_len,
    )
noise = WhiteNoise(std_error=sigma_v)

# Normal model
model = Model(
    local_trend,
    chronos,
    noise,
)

#  Abnormal model
ab_model = Model(
    local_acceleration,
    chronos,
    noise,
)

model.lstm_output_history.mu = train_data["y"][-context_len:].flatten()
model.lstm_output_history.time = train_data["time"][-context_len:]

# Switching Kalman filter
skf = SKF(
    norm_model=model,
    abnorm_model=ab_model,
    std_transition_error=1e-5,
    norm_to_abnorm_prob=1e-5,
    abnorm_to_norm_prob=0.1,
)
skf.auto_initialize_baseline_states(validation_data["y"][0:52])

#  Training

# # Anomaly Detection
filter_marginal_abnorm_prob, states = skf.filter(data=validation_data)

# # Plot
marginal_abnorm_prob_plot = filter_marginal_abnorm_prob

fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=states,
    # states_type="prior",
    model_prob=marginal_abnorm_prob_plot,
    standardization=True,
    color="b",
    plot_observation = True,
    time_start_index = len(train_data["y"]),
)
# ax[0].plot(validation_data["time"], validation_data["y"])
fig.suptitle("SKF hidden states", fontsize=10, y=1)
plt.show()
