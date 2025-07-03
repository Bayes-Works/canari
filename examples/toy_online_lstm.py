import copy
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend
from examples.view_param import ParameterViewer
from examples.param_intervener import ParameterIntervener
from typing import Dict, Tuple

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


# ------------------------------------------------------------
#  Utilities for KL‑divergence diagnostics
# ------------------------------------------------------------
def _kl_divergence_gaussian(
    prior_mu: list,
    prior_var: list,
    post_mu: list,
    post_var: list,
) -> list:
    """
    Element‑wise KL divergence D_KL[ q‖p ] between two univariate
    Gaussians where q ≜ 𝒩(post_mu, post_var) (posterior) and
    p ≜ 𝒩(prior_mu, prior_var) (prior).

    All arguments must be lists that broadcast to the same
    shape. Returns a list containing the KL contribution of each parameter.
    """
    prior_mu = np.array(prior_mu)
    prior_var = np.array(prior_var)
    post_mu = np.array(post_mu)
    post_var = np.array(post_var)

    kl_div = 0.5 * (
        np.log(post_var / prior_var)
        + (prior_var + (prior_mu - post_mu) ** 2) / post_var
        - 1.0
    )
    return kl_div.tolist()


def compute_layer_kl(
    prior_entry: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    post_entry: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute KL divergence for a layer stored in the state_dict as the
    tuple (mu_w, var_w, mu_b, var_b).

    Returns a dict with keys 'weights' and 'bias' containing the
    element‑wise KL values.
    """
    mu_w0, var_w0, mu_b0, var_b0 = prior_entry
    mu_w1, var_w1, mu_b1, var_b1 = post_entry

    kl_w = _kl_divergence_gaussian(mu_w0, var_w0, mu_w1, var_w1)
    kl_b = _kl_divergence_gaussian(mu_b0, var_b0, mu_b1, var_b1)
    return {"weights": kl_w, "bias": kl_b}


# ------------------------------------------------------------
#  Utilities for Wasserstein distance diagnostics
# ------------------------------------------------------------
def _wasserstein_distance_gaussian(
    prior_mu: list,
    prior_var: list,
    post_mu: list,
    post_var: list,
) -> list:
    """
    Element‑wise 2‑Wasserstein distance W₂ between two univariate
    Gaussians 𝒩(prior_mu, prior_var) and 𝒩(post_mu, post_var).

    For 1‑D Gaussians the squared W₂ distance simplifies to
        (μ₁ − μ₂)² + (σ₁ − σ₂)²
    where σ = √var.

    Returns a list containing W₂ for each parameter.
    """
    prior_mu = np.array(prior_mu)
    prior_std = np.sqrt(np.array(prior_var))
    post_mu = np.array(post_mu)
    post_std = np.sqrt(np.array(post_var))

    w2_sq = (prior_mu - post_mu) ** 2 + (prior_std - post_std) ** 2
    return np.sqrt(w2_sq).tolist()


def compute_layer_wasserstein(
    prior_entry: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    post_entry: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute 2‑Wasserstein distance for a layer stored in the state_dict as
    (mu_w, var_w, mu_b, var_b).

    Returns a dict with keys 'weights' and 'bias' containing the
    element‑wise W₂ values.
    """
    mu_w0, var_w0, mu_b0, var_b0 = prior_entry
    mu_w1, var_w1, mu_b1, var_b1 = post_entry

    w_w = _wasserstein_distance_gaussian(mu_w0, var_w0, mu_w1, var_w1)
    w_b = _wasserstein_distance_gaussian(mu_b0, var_b0, mu_b1, var_b1)
    return {"weights": w_w, "bias": w_b}


# Generate synthetic data
frequency = 1 / 24  # One cycle per 24 hours
phase = 0  # Initial phase
sampling_rate = 1  # 1 sample per hour
duration = 1 / frequency * 30  # Total duration
# change_points = [(0, 1), (24 * 10, 1.5), (24 * 12, 1), (24 * 15, 1.5, 1 / 48)]
change_points = [(0, 1), (24 * 10, 2), (24 * 12, 1), (24 * 15, 2, 1 / 48)]

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
D = 48  # length of smooth window

# Build data processor
data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day"],
    train_split=0.8,
    validation_split=0.1,
    output_col=output_col,
)

# split data
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
sigma_v = 0.01
model = Model(
    LstmNetwork(
        look_back_len=24,
        num_features=2,
        infer_len=24,  # corresponds to one period
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
        # smoother=False,
    ),
    WhiteNoise(std_error=sigma_v),
)

# add model intervention
pi = ParameterIntervener(model.lstm_net)

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
# history of mean KL divergence per layer (weights / bias)
kl_history = collections.defaultdict(lambda: {"weights": [], "bias": []})
wasserstein_history = collections.defaultdict(lambda: {"weights": [], "bias": []})

for olstm_idx in range(0, len(train_data["y"]) - D, 1):

    # ----- capture prior parameter distributions -----
    prior_state = copy.deepcopy(model.lstm_net.state_dict())

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
    # ----- compute KL divergence between prior and posterior -----
    posterior_state = model.lstm_net.state_dict()
    kl_results = {
        layer: compute_layer_kl(prior_state[layer], posterior_state[layer])
        for layer in prior_state
    }

    # Print mean KL per layer for quick inspection
    mean_kl = {
        layer: {part: np.mean(values[part]) for part in values}
        for layer, values in kl_results.items()
    }
    # print(f"Window {olstm_idx}: mean KL per layer → {mean_kl}")
    # store KL history
    for lyr in mean_kl:
        for part in mean_kl[lyr]:
            kl_history[lyr][part].append(mean_kl[lyr][part])

    # ----- compute Wasserstein distance between prior and posterior -----
    w_results = {
        layer: compute_layer_wasserstein(prior_state[layer], posterior_state[layer])
        for layer in prior_state
    }
    mean_w = {
        layer: {part: np.mean(values[part]) for part in values}
        for layer, values in w_results.items()
    }
    # print(f"Window {olstm_idx}: mean W₂ per layer → {mean_w}")
    # store Wasserstein history
    for lyr in mean_w:
        for part in mean_w[lyr]:
            wasserstein_history[lyr][part].append(mean_w[lyr][part])

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

    # if olstm_idx == 224:
    #     mask = pi.inflate_variance(threshold=1e-4, factor=1000.0)
    # baseline = pi.snapshot_state()
    # pi.add_noise_to_means(mask, std_scale=1.0)
    # pi.prune_means_to_zero(mask)

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


# ------------------------------------------------------------
#  Plot predictions and residuals
# ------------------------------------------------------------
fig, (ax_pred, ax_res) = plt.subplots(
    2, 1, figsize=(14, 6), sharex=True, height_ratios=[2, 1]
)

# ------------------- predictions (upper axis) -------------------
# Plot training true values
ax_pred.plot(
    np.arange(len(train_data["y"])),
    train_data["y"].flatten(),
    label="Train true values",
    color="red",
)

# Plot one‑step ahead predictions
offset = D
x_range = np.arange(offset, offset + len(mu_preds))
mu_preds_arr = np.array(mu_preds).flatten()
std_preds_arr = np.array(std_preds).flatten()
ax_pred.plot(x_range, mu_preds_arr, label="One‑step predicted mean", color="blue")
ax_pred.fill_between(
    x_range,
    mu_preds_arr - std_preds_arr,
    mu_preds_arr + std_preds_arr,
    color="blue",
    alpha=0.3,
    label="±1 std dev (1‑step)",
)

# Plot multi‑step ahead forecasts
true_vals = validation_data["y"].flatten()
forecast_range = np.arange(len(train_data["y"]), len(train_data["y"]) + len(true_vals))
ax_pred.plot(forecast_range, true_vals, label="Future true values", color="green")
ax_pred.plot(
    forecast_range,
    mu_forecast.flatten(),
    label="Forecasted mean",
    color="orange",
)
ax_pred.fill_between(
    forecast_range,
    mu_forecast.flatten() - std_forecast.flatten(),
    mu_forecast.flatten() + std_forecast.flatten(),
    color="orange",
    alpha=0.3,
    label="±1 std dev (forecast)",
)

ax_pred.set_ylabel("Value")
ax_pred.set_title("Online LSTM: Predictions")
ax_pred.grid(True)
ax_pred.legend()

# ------------------- residuals (lower axis) -------------------
# Residuals for one‑step predictions
one_step_true = train_data["y"][offset:].flatten()[: len(mu_preds_arr)]
residual_one_step = one_step_true - mu_preds_arr
ax_res.plot(
    x_range,
    residual_one_step,
    label="Residual (1‑step)",
    linestyle="-",
    marker="o",
    markersize=2,
)

# Residuals for multi‑step forecasts
residual_forecast = true_vals - mu_forecast.flatten()
ax_res.plot(
    forecast_range,
    residual_forecast,
    label="Residual (forecast)",
    linestyle="-",
    marker="x",
    markersize=3,
)

ax_res.axhline(0.0, color="black", linewidth=0.8)
ax_res.set_xlabel("Time")
ax_res.set_ylabel("Residual (true − predicted)")
ax_res.set_title("Residuals")
ax_res.grid(True)
ax_res.legend()

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
#  Plot KL divergence history over training windows
# ------------------------------------------------------------------
plt.figure(figsize=(12, 4))
for lyr, parts in kl_history.items():
    if lyr == "SLinear.1":
        # skip linear layer
        continue
    kl_total = np.array(parts["weights"]) + np.array(parts["bias"])
    plt.plot(
        np.arange(len(kl_total)),
        kl_total,
        label=f"{lyr} (w+b)",
    )
plt.xlabel("Training window index")
plt.ylabel("Mean KL divergence")
plt.ylim(0, 0.0075)  # Set y-axis limits for better visibility
plt.title("Layer‑wise KL divergence across online training")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
#  Plot Wasserstein distance history over training windows
# ------------------------------------------------------------------
plt.figure(figsize=(12, 4))
for lyr, parts in wasserstein_history.items():
    if lyr == "SLinear.1":
        # skip linear layer
        continue
    w_total = np.array(parts["weights"]) + np.array(parts["bias"])
    plt.plot(
        np.arange(len(w_total)),
        w_total,
        label=f"{lyr} (w+b)",
        linestyle="-",
        marker="x",
    )
plt.xlabel("Training window index")
plt.ylabel("Mean 2‑Wasserstein distance")
plt.title("Layer‑wise 2‑Wasserstein distance across online training")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
