import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari.data_process import DataProcess
from canari.baseline_component import LocalTrend
from canari.lstm_component import LstmNetwork
from canari.white_noise_component import WhiteNoise
from canari.model import Model
from canari.data_visualization import (
    plot_data,
    plot_prediction,
    plot_states,
)


# def generate_changing_amplitude_sine(
#     frequency=1,
#     phase=0,
#     sampling_rate=100,
#     duration=10,
#     change_points=None,
#     noise_std=0.0,
# ):
#     """
#     Generate a sine wave time series with variable amplitude and frequency,
#     ensuring continuity at changepoints by adjusting the phase.

#     If `change_points` is None, a constant amplitude and frequency are used.
#     Otherwise, the amplitude and frequency change at the specified time points,
#     and the phase is updated to keep the sine wave continuous at each changepoint.

#     Parameters
#     ----------
#     frequency : float, optional
#         Default frequency of the sine wave (default is 1). This is used if a change point
#         does not specify a frequency.
#     phase : float, optional
#         Initial phase in radians (default is 0).
#     sampling_rate : int, optional
#         Number of samples per second (default is 100).
#     duration : int or float, optional
#         Duration of the signal.
#     change_points : list of tuple, optional
#         Each tuple should specify (time, amplitude) or (time, amplitude, frequency).
#         The amplitude and frequency change at these time points.

#     Returns
#     -------
#     tuple
#         t : ndarray
#             Time points.
#         y : ndarray
#             Sine wave values.
#     """
#     t = np.linspace(0, duration, int(sampling_rate * duration))
#     if change_points is None:
#         y = np.sin(2 * np.pi * frequency * t + phase)
#     else:
#         y = np.zeros_like(t)
#         # Initialize with the default frequency and phase for the first segment
#         current_phase = phase
#         current_freq = frequency

#         # Process each segment defined by change_points
#         for i in range(len(change_points) - 1):
#             cp = change_points[i]
#             start_time = cp[0]
#             amplitude = cp[1]
#             seg_freq = cp[2] if len(cp) > 2 else frequency

#             # For segments after the first, adjust phase to ensure continuity
#             if i > 0:
#                 # t_c is the current changepoint time
#                 t_c = start_time
#                 # Adjust phase so that:
#                 # sin(2*pi*seg_freq*t_c + new_phase) = sin(2*pi*current_freq*t_c + current_phase)
#                 current_phase = (2 * np.pi * current_freq * t_c + current_phase) - (
#                     2 * np.pi * seg_freq * t_c
#                 )
#                 current_freq = seg_freq

#             # Determine end time for this segment
#             next_cp = change_points[i + 1]
#             end_time = next_cp[0]
#             mask = (t >= start_time) & (t < end_time)
#             y[mask] = amplitude * np.sin(2 * np.pi * seg_freq * t[mask] + current_phase)

#         # Handle the final segment
#         last_cp = change_points[-1]
#         start_time = last_cp[0]
#         amplitude = last_cp[1]
#         seg_freq = last_cp[2] if len(last_cp) > 2 else frequency
#         if len(change_points) > 1:
#             t_c = start_time
#             current_phase = (2 * np.pi * current_freq * t_c + current_phase) - (
#                 2 * np.pi * seg_freq * t_c
#             )
#         mask = t >= start_time
#         y[mask] = amplitude * np.sin(2 * np.pi * seg_freq * t[mask] + current_phase)
#     if noise_std > 0.0:
#         noise = np.random.normal(loc=0.0, scale=noise_std, size=len(y))
#         y = y + noise
#     return t, y


# # Generate synthetic data
# frequency = 1 / 24  # One cycle per 24 hours
# phase = 0  # Initial phase
# sampling_rate = 1  # 1 sample per hour
# duration = 1 / frequency * 15  # Total duration
# change_points = [(0, 1), (24 * 8, 1.5), (24 * 12, 1), (24 * 16, 1.5, 1 / 48)]

# np.random.seed(1)

# t, y = generate_changing_amplitude_sine(
#     frequency=frequency,
#     phase=phase,
#     sampling_rate=sampling_rate,
#     duration=duration,
#     # change_points=change_points,
#     noise_std=0.05,
# )

# # build data into a dataframe
# df = pd.DataFrame(y, columns=["values"])
# df.index = pd.date_range(start="2023-01-01", periods=len(y), freq="H")
# df.index.name = "date_time"

####

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
num_epoch = 200

# Build data processor
data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day"],
    train_split=0.6,
    validation_split=0.2,
    output_col=output_col,
)

# split data
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
sigma_v = 0.003
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=24,
        num_features=2,
        num_layer=1,
        num_hidden_unit=40,
        device="cpu",
        manual_seed=1,
    ),
    WhiteNoise(std_error=sigma_v),
)

# model.auto_initialize_baseline_states(train_data["y"][0:24])

# set white noise decay to True
white_noise_decay = True

# Training
for epoch in range(num_epoch):

    # set white noise decay
    if white_noise_decay and model.get_states_index("white noise") is not None:
        model._white_noise_decay(
            model._current_epoch, white_noise_max_std=2, white_noise_decay_factor=0.9
        )

    # filter on train data
    model.filter(train_data, train_lstm=True)

    # smooth on train data
    model.smoother(train_data)

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
    mse = metric.mse(mu_validation_preds, validation_obs)

    # Early-stopping
    model.early_stopping(evaluate_metric=mse, mode="min", patience=10)
    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
        states_optim = copy.copy(
            model.states
        )  # If we want to plot the states, plot those from optimal epoch
        model_optim_dict = model.get_dict()
        lstm_optimal_states = model.lstm_net.get_lstm_states()
    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

# set memory and parameters to optimal epoch
model.load_dict(model_optim_dict)
model.set_memory(
    states=states_optim,
    time_step=data_processor.test_start,
    lstm_states=lstm_optimal_states,
)

# forecat on the test set
mu_test_preds, std_test_preds, test_states = model.forecast(
    data=test_data,
)

# Unstandardize the predictions
mu_test_preds = normalizer.unstandardize(
    mu_test_preds,
    data_processor.norm_const_mean[output_col],
    data_processor.norm_const_std[output_col],
)
std_test_preds = normalizer.unstandardize_std(
    std_test_preds,
    data_processor.norm_const_std[output_col],
)

# calculate the test metrics
test_obs = data_processor.get_data("test").flatten()
mse = metric.mse(mu_test_preds, test_obs)
log_lik = metric.log_likelihood(mu_test_preds, test_obs, std_test_preds)

print(f"Test MSE            :{mse: 0.4f}")
print(f"Test Log-Lik        :{log_lik: 0.2f}")

# plot the test data
fig, ax = plt.subplots(figsize=(10, 6))
plot_data(
    data_processor=data_processor,
    normalization=False,
    plot_column=output_col,
    validation_label="y",
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds_optim,
    std_validation_pred=std_validation_preds_optim,
    validation_label=[r"$\mu$", r"$\pm\sigma$"],
)
plot_prediction(
    data_processor=data_processor,
    mean_test_pred=mu_test_preds,
    std_test_pred=std_test_preds,
    test_label=[r"$\mu^{\prime}$", r"$\pm\sigma^{\prime}$"],
)
plt.legend(loc=(0.1, 1.01), ncol=6, fontsize=12)
plt.tight_layout()
plt.show()

# plot states
fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="smooth",
    states_to_plot=[
        "local level",
        "local trend",
        "lstm",
    ],
)
plt.show()
