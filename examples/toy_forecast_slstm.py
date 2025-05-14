import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend


# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
linear_space = np.linspace(0, 2, num=len(df_raw))
df_raw = df_raw.add(linear_space, axis=0)

data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Resampling data
df = df_raw.resample("H").mean()

# Define parameters
output_col = [0]
num_epoch = 10

data_processor = DataProcess(
    data=df,
    train_split=0.8,
    validation_split=0.2,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
sigma_v = 0.001
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=12,
        num_features=1,
        num_layer=1,
        num_hidden_unit=40,
        device="cpu",
        manual_seed=1,
        smoother=True,
    ),
    WhiteNoise(std_error=sigma_v),
)
model.auto_initialize_baseline_states(train_data["y"][0:24])

if model.lstm_net.smooth:
    model.lstm_net.num_samples = (
        # model.lstm_net.lstm_look_back_len
        +len(train_data["y"])
        # + len(validation_data["y"])
    )

# Training
for epoch in range(num_epoch):

    # set white noise decay
    model._white_noise_decay(epoch, white_noise_max_std=3, white_noise_decay_factor=0.9)

    # filter on train data
    mu_preds_train, std_preds_train, states = model.filter(train_data, train_lstm=True)

    # smooth on train data
    model.smoother()

    # smooth lstm states
    if model.lstm_net.smooth:
        mu_zo_smooth, var_zo_smooth = model.lstm_net.smoother()

    fig, ax = plot_states(
        data_processor=data_processor,
        states=model.states,
        states_type="prior",
    )
    filename = f"saved_results/smoother#{epoch}.png"
    plt.savefig(filename)
    plt.close()

    model.set_memory(states=model.states, time_step=0)
