import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend
from examples.view_param import ParameterViewer

# # Read data
# data_file = "./data/toy_time_series/sine.csv"
# df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

# data_file_time = "./data/toy_time_series/sine_datetime.csv"
# time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
# time_series = pd.to_datetime(time_series[0])
# df_raw.index = time_series
# df_raw.index.name = "date_time"
# df_raw.columns = ["values"]

# # Resampling data
# df = df_raw.resample("H").mean()


# synthesize data
def generate_changing_amplitude_sine(
    frequency=1,
    phase=0,
    sampling_rate=100,
    duration=10,
    change_points=None,
    noise_std=0.0,
):
    """
    Generate a sine wave time series with variable amplitude and frequency,
    ensuring continuity at changepoints by adjusting the phase.

    If `change_points` is None, a constant amplitude and frequency are used.
    Otherwise, the amplitude and frequency change at the specified time points,
    and the phase is updated to keep the sine wave continuous at each changepoint.

    Parameters
    ----------
    frequency : float, optional
        Default frequency of the sine wave (default is 1). This is used if a change point
        does not specify a frequency.
    phase : float, optional
        Initial phase in radians (default is 0).
    sampling_rate : int, optional
        Number of samples per second (default is 100).
    duration : int or float, optional
        Duration of the signal.
    change_points : list of tuple, optional
        Each tuple should specify (time, amplitude) or (time, amplitude, frequency).
        The amplitude and frequency change at these time points.

    Returns
    -------
    tuple
        t : ndarray
            Time points.
        y : ndarray
            Sine wave values.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration))
    if change_points is None:
        y = np.sin(2 * np.pi * frequency * t + phase)
    else:
        y = np.zeros_like(t)
        # Initialize with the default frequency and phase for the first segment
        current_phase = phase
        current_freq = frequency

        # Process each segment defined by change_points
        for i in range(len(change_points) - 1):
            cp = change_points[i]
            start_time = cp[0]
            amplitude = cp[1]
            seg_freq = cp[2] if len(cp) > 2 else frequency

            # For segments after the first, adjust phase to ensure continuity
            if i > 0:
                # t_c is the current changepoint time
                t_c = start_time
                # Adjust phase so that:
                # sin(2*pi*seg_freq*t_c + new_phase) = sin(2*pi*current_freq*t_c + current_phase)
                current_phase = (2 * np.pi * current_freq * t_c + current_phase) - (
                    2 * np.pi * seg_freq * t_c
                )
                current_freq = seg_freq

            # Determine end time for this segment
            next_cp = change_points[i + 1]
            end_time = next_cp[0]
            mask = (t >= start_time) & (t < end_time)
            y[mask] = amplitude * np.sin(2 * np.pi * seg_freq * t[mask] + current_phase)

        # Handle the final segment
        last_cp = change_points[-1]
        start_time = last_cp[0]
        amplitude = last_cp[1]
        seg_freq = last_cp[2] if len(last_cp) > 2 else frequency
        if len(change_points) > 1:
            t_c = start_time
            current_phase = (2 * np.pi * current_freq * t_c + current_phase) - (
                2 * np.pi * seg_freq * t_c
            )
        mask = t >= start_time
        y[mask] = amplitude * np.sin(2 * np.pi * seg_freq * t[mask] + current_phase)
    if noise_std > 0.0:
        noise = np.random.normal(loc=0.0, scale=noise_std, size=len(y))
        y = y + noise
    return t, y


# Generate synthetic data
frequency = 1 / 24  # One cycle per 24 hours
phase = 0  # Initial phase
sampling_rate = 1  # 1 sample per hour
duration = 1 / frequency * 20  # Total duration
change_points = [(0, 1), (24 * 10, 1.5), (24 * 12, 1), (24 * 15, 1.5, 1 / 24)]

t, y = generate_changing_amplitude_sine(
    frequency=frequency,
    phase=phase,
    sampling_rate=sampling_rate,
    duration=duration,
    change_points=change_points,
    # noise_std=0.1,
)

# Create a DataFrame
df = pd.DataFrame(
    {"values": y}, index=pd.date_range(start="2023-01-01", periods=len(y), freq="H")
)
df.index.name = "date_time"
df.columns = ["values"]


# Define parameters
output_col = [0]
D = 24  # length of smooth window

# Build data processor
data_processor = DataProcess(
    data=df,
    # time_covariates=["hour_of_day"],
    train_split=0.8,
    validation_split=0.1,
    output_col=output_col,
)

# split data
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
sigma_v = 0.001
model = Model(
    LstmNetwork(
        look_back_len=24,
        num_features=1,
        infer_len=24,  # corresponds to one period
        num_layer=1,
        num_hidden_unit=20,
        device="cpu",
        manual_seed=1,
        # smoother=False,
    ),
    WhiteNoise(std_error=sigma_v),
)

state_dict = model.lstm_net.state_dict()
# check the max value of the variances
mu_w, var_w, mu_b, var_b = state_dict["SLSTM.0"]
max_mu_lstm = max(max(mu_w), max(mu_b))

mu_w, mu_w, mu_b, mu_b = state_dict["SLinear.1"]
max_mu_linear = max(max(mu_w), max(mu_b))


# viewer = ParameterViewer(model.lstm_net)

if model.lstm_net.smooth:
    model.lstm_net.num_samples = D

# extract the training data
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# store all predictions
mu_preds = []
std_preds = []

post_lstm_mu = []
post_lstm_var = []

for olstm_idx in range(0, len(train_data["y"]) - D, 1):

    # Prepare data for online LSTM
    x_train = train_data["x"][olstm_idx : olstm_idx + D]
    y_train = train_data["y"][olstm_idx : olstm_idx + D]

    x_val = train_data["x"][olstm_idx + D : olstm_idx + D + 1]
    y_val = train_data["y"][olstm_idx + D : olstm_idx + D + 1]

    # rebuild into dictionary
    training_window = {
        "x": x_train,
        "y": y_train,
    }

    validation_window = {
        "x": x_val,
        "y": y_val,
    }

    # train the model
    model.lstm_net.train()
    mu_filt, std_filt, states = model.filter(training_window, train_lstm=True)
    forecast_lstm_states = model.lstm_net.get_lstm_states()

    # if olstm_idx % 1 == 0:
    #     viewer.heatmap(
    #         "0",
    #         epoch=olstm_idx,
    #         cmap="plasma",
    #         which="mean",
    #         vmin=0.0,
    #         vmax=max_mu_lstm,
    #         layer_type="LSTM",
    #         return_img=True,
    #         save=False,
    #     )
    #     viewer.heatmap(
    #         "1",
    #         epoch=olstm_idx,
    #         cmap="plasma",
    #         which="mean",
    #         vmin=0.0,
    #         vmax=max_mu_linear,
    #         # layer_type="LSTM",
    #         return_img=True,
    #         save=False,
    #     )

    # store posteriors
    post_mu = states.get_mean("lstm", states_type="posterior")[-D:]
    post_var = states.get_std("lstm", states_type="posterior")[-D:]

    post_lstm_mu[olstm_idx : olstm_idx + D] = post_mu
    post_lstm_var[olstm_idx : olstm_idx + D] = post_var

    # plt.figure(figsize=(10, 4))
    # post_lstm_mu_arr = np.array(post_lstm_mu)
    # post_lstm_var_arr = np.array(post_lstm_var)

    # x_axis = np.arange(len(post_lstm_mu_arr))

    # plt.plot(x_axis, post_lstm_mu_arr, label="Posterior Mean")
    # plt.fill_between(
    #     x_axis,
    #     post_lstm_mu_arr - post_lstm_var_arr,
    #     post_lstm_mu_arr + post_lstm_var_arr,
    #     color="blue",
    #     alpha=0.3,
    # )
    # plt.tight_layout()
    # plt.show()

    # foreacst one step ahead
    model.lstm_net.eval()
    mu_pred, std_pred, _ = model.forecast(validation_window)
    mu_preds.append(mu_pred)
    std_preds.append(std_pred)

    # call smoother for online LSTM
    if model.lstm_net.smooth and olstm_idx < len(train_data["y"]) - D:
        _, _ = model.lstm_net.smoother()

        if olstm_idx > D:
            mu_zo_smooth = post_lstm_mu
            var_zo_smooth = post_lstm_var

            # convert to numpy arrays
            mu_zo_smooth = np.array(mu_zo_smooth, dtype=np.float32)
            var_zo_smooth = np.array(var_zo_smooth, dtype=np.float32)

            # set smoothed LSTM output history
            model.lstm_output_history.set(
                mu_zo_smooth[olstm_idx - model.lstm_net.lstm_look_back_len : olstm_idx],
                var_zo_smooth[
                    olstm_idx - model.lstm_net.lstm_look_back_len : olstm_idx
                ],
            )

        # set smoothed LSTM states
        model.lstm_net.set_lstm_states(model.lstm_net.get_lstm_states_smooth(1))

# viewer.save_gif("lstm0.gif", key="0", fps=3)
# viewer.save_gif("linear1.gif",  key="1",  fps=3)

# plot the results
# plt.figure(figsize=(12, 5))
# plt.plot(train_data["y"], label="True values", color="red")

# # Offset prediction to align with the validation point after each training window
# offset = D
# x_range = np.arange(offset, offset + len(mu_preds))

# mu_preds_arr = np.array(mu_preds).flatten()
# std_preds_arr = np.array(std_preds).flatten()

# plt.plot(x_range, mu_preds_arr, label="Predicted mean", color="blue")
# plt.fill_between(
#     x_range,
#     mu_preds_arr - std_preds_arr,
#     mu_preds_arr + std_preds_arr,
#     color="blue",
#     alpha=0.3,
#     label="±1 std dev",
# )
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# set the correct lstm states
model.lstm_output_history.mu = np.array(
    post_mu[-model.lstm_net.lstm_look_back_len :], dtype=np.float32
)
model.lstm_output_history.std = np.array(
    post_var[-model.lstm_net.lstm_look_back_len :], dtype=np.float32
)
model.lstm_net.set_lstm_states(forecast_lstm_states)

# forecast a multi step ahead
# mu_forecast, std_forecast, _ = model.filter(validation_data, train_lstm=False)
mu_forecast, std_forecast, _ = model.forecast(validation_data)

# Add multi-step ahead forecast to the same plot as one-step ahead
# Plot training data
plt.figure(figsize=(12, 5))
plt.plot(
    np.arange(len(train_data["y"])),
    train_data["y"].flatten(),
    label="Train true values",
    color="red",
)

# Plot one-step ahead predictions already stored
offset = D
x_range = np.arange(offset, offset + len(mu_preds))
mu_preds_arr = np.array(mu_preds).flatten()
std_preds_arr = np.array(std_preds).flatten()
plt.plot(x_range, mu_preds_arr, label="One-step predicted mean", color="blue")
plt.fill_between(
    x_range,
    mu_preds_arr - std_preds_arr,
    mu_preds_arr + std_preds_arr,
    color="blue",
    alpha=0.3,
    label="±1 std dev (1-step)",
)

# Plot validation (future) true values and multi-step forecasts
true_vals = validation_data["y"].flatten()
forecast_range = np.arange(len(train_data["y"]), len(train_data["y"]) + len(true_vals))
plt.plot(forecast_range, true_vals, label="Future true values", color="green")
plt.plot(forecast_range, mu_forecast.flatten(), label="Forecasted mean", color="orange")
plt.fill_between(
    forecast_range,
    mu_forecast.flatten() - std_forecast.flatten(),
    mu_forecast.flatten() + std_forecast.flatten(),
    color="orange",
    alpha=0.3,
    label="±1 std dev (forecast)",
)
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Online LSTM: Train, One-step, and Multi-step Forecasts")
plt.grid(True)
plt.tight_layout()
plt.show()
