import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
import tqdm
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend

from scipy.signal import detrend

# Read data
data_file = "./data/detrended_data/values.csv"
df_values = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/detrended_data/dates.csv"
df_dates = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)

# discard first 13 columns
df_values = df_values.iloc[:, 13:]
df_dates = df_dates.iloc[:, 13:]

data_processors = []

# iterate over the columns of the data
for i in range(df_values.shape[1]):
    df = df_values.iloc[:, i].to_frame(name="values")
    df.index = pd.to_datetime(df_dates.iloc[:, i])
    df.index.name = "date_time"

    df = df.dropna(subset=["values"])

    # build data processor
    data_processor = DataProcess(
        data=df,
        time_covariates=["week_of_year"],
        train_split=0.7,
        validation_split=0.2,
        output_col=[0],
    )
    data_processors.append(data_processor)

# load target data
data_files = [
    "./data/benchmark_data/test_1_data.csv",
    "./data/benchmark_data/test_4_data.csv",
    "./data/benchmark_data/test_5_data.csv",
    "./data/benchmark_data/test_6_data.csv",
    "./data/benchmark_data/test_7_data.csv",
    "./data/benchmark_data/test_8_data.csv",
    "./data/benchmark_data/test_9_data.csv",
    "./data/benchmark_data/test_10_data.csv",
    "./data/benchmark_data/test_11_data.csv",
]
for data_file_target in data_files:
    df_target = pd.read_csv(
        data_file_target,
        usecols=[0, 1],  # Read only the first two columns
        skiprows=1,
        header=None,
        names=["date_time", "values"],
        parse_dates=["date_time"],
        index_col="date_time",
    )
    # Resample weekly averages
    df_target = df_target.resample("W").mean()
    df_target["values"] = df_target["values"].interpolate(method="time")
    df_target["values"] = df_target["values"].bfill().ffill()

    # Apply detrending
    df_target["values"] = detrend(df_target["values"].values)

    # Build data processor for the target series
    data_processor = DataProcess(
        data=df_target,
        time_covariates=["week_of_year"],
        train_split=0.3,
        validation_split=0.1,
        output_col=[0],
    )
    data_processors.append(data_processor)

# iterate over the data processors and split per time series
data_splits = {}
for i, data_processor in enumerate(data_processors):
    train_data, validation_data, test_data, normalized_data = (
        data_processor.get_splits()
    )
    data_splits[i] = {
        "train": train_data,
        "validation": validation_data,
        "test": test_data,
        "normalized": normalized_data,
    }

# Define parameters
output_col = [0]
num_epoch = 200

# Model
sigma_v = 0.3
model = Model(
    LstmNetwork(
        look_back_len=19,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
    ),
    WhiteNoise(std_error=sigma_v),
)

# set white noise decay to True
white_noise_decay = True


# Training
pbar = tqdm.tqdm(range(num_epoch), desc="Epoch", unit="epoch")
for epoch in pbar:

    mses = []

    # set white noise decay
    if white_noise_decay and model.get_states_index("white noise") is not None:
        model._white_noise_decay(
            model._current_epoch, white_noise_max_std=3, white_noise_decay_factor=0.9
        )

    # iterate in a random manner over the data in the data_splits
    random_order = np.random.permutation(len(data_processors))
    for i in random_order:
        data_processor = data_processors[i]
        train_data = data_splits[i]["train"]
        validation_data = data_splits[i]["validation"]
        normalized_data = data_splits[i]["normalized"]

        # filter on train data
        model.filter(train_data, train_lstm=True)

        # forecast on the validation set
        mu_validation_preds, std_validation_preds, _ = model.forecast(validation_data)

        # reset memory
        model.set_memory(states=model.states, time_step=0)

        # Unstandardize the predictions
        mu_validation_preds = normalizer.unstandardize(
            mu_validation_preds,
            data_processor.norm_const_mean[output_col],
            data_processor.norm_const_std[output_col],
        )
        std_validation_preds = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor.norm_const_std[output_col],
        )

        # Calculate the log-likelihood metric
        validation_obs = data_processor.get_data("validation").flatten()
        mses.append(metric.mse(mu_validation_preds, validation_obs))

    mse = np.mean(mses)
    pbar.set_postfix({"val_mse": f"{mse:.4f}"})

    # Early-stopping
    model.early_stopping(evaluate_metric=mse, mode="min", patience=10)
    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
        states_optim = copy.copy(
            model.states
        )  # If we want to plot the states, plot those from optimal epoch
        model_optim_dict = model.get_dict()
    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

model.load_dict(model_optim_dict)

# save global lstm network
params_path = "saved_params/lstm_net_test.pth"
model.lstm_net.save(filename=params_path)
