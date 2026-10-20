import pickle
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_states,
    plot_data,
    plot_prediction,
)
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

with open("./saved_params/toy_anomaly_detection_tune.pkl", "rb") as f:
    skf_optim_dict = pickle.load(f)
skf_optim = SKF.load_dict(skf_optim_dict)
lstm_optim = skf_optim_dict["norm_model"]["components"]["lstm 1"]

# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Add synthetic anomaly to data
trend = np.linspace(0, 0, num=len(df_raw))
time_anomaly = 120
new_trend = np.linspace(0, 1, num=len(df_raw) - time_anomaly)
trend[time_anomaly:] = trend[time_anomaly:] + new_trend
df_raw = df_raw.add(trend, axis=0)

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["hour_of_day"],
    train_split=0.4,
    validation_split=0.1,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Model
sigma_v = 1e-5
model = Model(
    LocalTrend(var_states=[1e-2,1e-2]),
    LstmNetwork(
        look_back_len=lstm_optim.look_back_len,
        infer_len=lstm_optim.infer_len,
        num_features=lstm_optim.num_features,
        num_layer=lstm_optim.num_layer,
        num_hidden_unit=lstm_optim.num_hidden_unit,
        device="cpu",
        manual_seed=1,
        smoother=lstm_optim.smoother,
    ),
    WhiteNoise(std_error=sigma_v),
)
model.lstm_net.load_state_dict(skf_optim.lstm_net.state_dict())

model.auto_initialize_baseline_states(train_data["y"][0:24])

model.filter(data=all_data, train_lstm=False)

#  Plot
fig, axes = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="posterior",
)
fig.suptitle("Posterior States")
plot_data(
    data_processor=data_processor,
    plot_column=output_col,
    sub_plot=axes[0],
)

fig, axes = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="prior",
)
fig.suptitle("Prior States")
plot_data(
    data_processor=data_processor,
    plot_column=output_col,
    sub_plot=axes[0],
)
plt.show()
