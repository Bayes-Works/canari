import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from canari.component import LocalTrend, LstmNetwork, Autoregression, WhiteNoise
import pickle
from scipy.interpolate import interp1d

def stretch_timeseries(y, t, change_index, new_period_ratio):
    """
    Stretch/compress a sine time series after a given index to change its period.

    Parameters:
    - y: original time series (1D array)
    - t: time vector (same length as y)
    - change_index: index after which to stretch
    - new_period_ratio: new_period / old_period (>1 = stretch, <1 = compress)

    Returns:
    - new_t: time vector after stretching
    - new_y: new interpolated time series
    """
    t_before = t[:change_index]
    y_before = y[:change_index]

    t_after = t[change_index:]
    t0 = t_after[0]

    # Stretch/compress time after change point
    t_after_stretched = t0 + (t_after - t0) * new_period_ratio

    # Combine time
    new_t = np.concatenate([t_before, t_after_stretched])

    # Interpolation function over original time series
    interp = interp1d(t, y, kind='linear', fill_value="extrapolate")

    # New values from interpolation
    new_y = interp(new_t)

    return new_t, new_y


###########################
###########################
#  Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Add synthetic anomaly to data
anm_mag = 0
anm_index = 700

# LT anomaly
anm_mag = -25/52
anm_baseline = np.arange(0, len(df_raw)-anm_index, dtype='float')
anm_baseline *= anm_mag

# # LL anomaly
# anm_mag = -50
# anm_baseline = np.zeros(len(df_raw)-anm_index, dtype='float')
# anm_baseline += anm_mag

# # Recurrent anomaly
# anm_index = 700
# anm_mag = 50
# anm_baseline = np.zeros(len(df_raw)-anm_index, dtype='float')
# for i in range(len(df_raw) - anm_index):
#     anm_baseline[i] = anm_mag * np.sin(i / 10)

df_raw.values[anm_index:] = (df_raw.values[anm_index:].squeeze() + anm_baseline).reshape(-1, 1)

# anm_mag=1
# _, df_raw.values = stretch_timeseries(
#     df_raw.values.squeeze(),
#     np.arange(len(df_raw)),
#     change_index=anm_index,
#     new_period_ratio=0.6,
# )


# Add trend
trend_true = 0.1
df_raw["values"] += np.arange(len(df_raw)) * trend_true

# Data processor initialization
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=0.6,
    validation_split=0,
    output_col=output_col,
)

# data_processor.scale_const_mean = np.array([35.571228, 26.068333])
# data_processor.scale_const_std = np.array([28.92418, 15.090957])

train_data, val_data, test_data, standardized_data = data_processor.get_splits()

# Standardization constants
scale_const_mean = data_processor.scale_const_mean[output_col].item()
scale_const_std = data_processor.scale_const_std[output_col].item()

with open("saved_params/toy_simple_model_white_noise_smallest.pkl", "rb") as f:
    model_dict = pickle.load(f)

lstm = LstmNetwork(
    look_back_len=52,
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
    # manual_seed=1,
)

###########################
###########################
# Reload pretrained model

# # # # # # #
sigma_v = 1e-3
# Define pretrained model:
pretrained_model = Model(
    LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), 
               var_states=[model_dict['states_optimal'].var_prior[0][0,0].item(), 
                           model_dict['states_optimal'].var_prior[0][1,1].item()]),
    lstm,
    WhiteNoise(std_error=sigma_v),
)

# load lstm's component
pretrained_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

# filter and smoother
mu_obs_preds, var_obs_preds, _ = pretrained_model.filter(standardized_data, train_lstm=False)
# pretrained_model.smoother()

# Get lstm states
lstm_states = pretrained_model.states.get_mean(
    states_name="lstm",
    states_type="prior",
)

# Compute the matrix profile of the lstm states
import stumpy
def past_only_matrix_profile(T, m, normalize=False):
    n = len(T)
    profile = np.full(n, 0.0)
    profile_idx = np.full(n, -1)

    for i in range(m, n):  # Start at m to ensure room for comparison
        Q = T[i-m:i]
        # D = stumpy.mass(Q, T[:i], normalize=normalize)
        D = stumpy.mass(Q, T[:int(0.6*len(T))], normalize=normalize)

        # Find best match in the past only
        min_idx = np.argmin(D)
        profile[i] = D[min_idx]
        profile_idx[i] = min_idx

    return profile, profile_idx

    return profile, profile_idx

# Example usage
m = 52
mp, mpi = past_only_matrix_profile(np.array(lstm_states).astype("float64"), m, normalize=False)
# Normalize mpi
# mp = (mp - np.min(mp)) / (np.max(mp) - np.min(mp))
# mp = stumpy.stump(np.array(lstm_states).astype("float64"), m=52, normalize=False)

state_type = "posterior"
# # Plotting results from pre-trained model
fig, ax = plot_states(
    data_processor=data_processor,
    states=pretrained_model.states,
    states_type=state_type,
    standardization=True,
    states_to_plot=[
        "level",
        "trend",
        "lstm",
        "white noise",
    ],
)
plot_data(
    data_processor=data_processor,
    standardization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax[0],
    plot_test_data=True,
)
time = data_processor.get_time(split="all")
ax[2].plot(
    time[:len(mp)],
    mp,
    label="MP metric",
    color="C1",
)
ax[2].legend()
ax[0].plot(time, mu_obs_preds)
ax[0].fill_between(
    time,
    mu_obs_preds - np.sqrt(var_obs_preds),
    mu_obs_preds + np.sqrt(var_obs_preds),
    alpha=0.2,
    color="C0",
)
# Plot the location when the anomaly starts
if anm_mag != 0:
    ax[0].axvline(
        x=df_raw.index[anm_index],
        color="red",
        linestyle="--",
        label="Anomaly start",
    )
# fig.suptitle("Hidden states estimated by the pre-trained model", fontsize=10, y=1)
plt.show()
