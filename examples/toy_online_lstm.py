import copy
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "experiments" / "out"
MPLCONFIG_DIR = OUT_DIR / "mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import pandas as pd
import numpy as np
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend
from typing import Dict, Tuple

SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (6.5, 3.5)

mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
        "lines.linewidth": 1,
        "figure.figsize": SINGLE_COL,
        "font.size": 9,
        "savefig.dpi": 300,
    }
)

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


def make_window(
    data: Dict[str, np.ndarray], start: int, end_exclusive: int
) -> Dict[str, np.ndarray]:
    return {
        "x": data["x"][start:end_exclusive],
        "y": data["y"][start:end_exclusive],
    }


def snapshot_runtime_state(model: Model) -> Dict[str, object]:
    model_attrs = [
        "mu_states",
        "var_states",
        "mu_states_prior",
        "var_states_prior",
        "mu_states_posterior",
        "var_states_posterior",
        "mu_obs_predict",
        "var_obs_predict",
        "states",
    ]
    snapshot = {name: copy.deepcopy(getattr(model, name, None)) for name in model_attrs}
    snapshot["lstm_history_mu"] = copy.deepcopy(
        getattr(model.lstm_output_history, "mu", None)
    )
    snapshot["lstm_history_var"] = copy.deepcopy(
        getattr(model.lstm_output_history, "var", None)
    )
    snapshot["lstm_states"] = copy.deepcopy(model.lstm_net.get_lstm_states())
    return snapshot


def restore_runtime_state(model: Model, snapshot: Dict[str, object]) -> None:
    for name, value in snapshot.items():
        if name in {"lstm_history_mu", "lstm_history_var", "lstm_states"}:
            continue
        setattr(model, name, copy.deepcopy(value))

    if snapshot["lstm_history_mu"] is not None:
        model.lstm_output_history.mu = copy.deepcopy(snapshot["lstm_history_mu"])
    if snapshot["lstm_history_var"] is not None:
        model.lstm_output_history.var = copy.deepcopy(snapshot["lstm_history_var"])
    model.lstm_net.set_lstm_states(copy.deepcopy(snapshot["lstm_states"]))


def forecast_one_step_without_committing_state(
    model: Model, data: Dict[str, np.ndarray], index: int
) -> Tuple[float, float]:
    runtime_state = snapshot_runtime_state(model)
    model.lstm_net.eval()
    forecast_window = {"x": data["x"][index : index + 1]}
    mu_pred, std_pred, _ = model.forecast(forecast_window)
    restore_runtime_state(model, runtime_state)
    return float(np.ravel(mu_pred)[0]), float(np.ravel(std_pred)[0])


def smooth_lstm_window_and_set_end_state(
    model: Model, states, smooth_window_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    if model.lstm_net.smooth:
        mu_smooth, var_smooth = model.lstm_net.smoother()
        mu_smooth = np.asarray(mu_smooth, dtype=np.float32).flatten()
        var_smooth = np.asarray(var_smooth, dtype=np.float32).flatten()

        if len(mu_smooth) < smooth_window_len:
            raise ValueError(
                "LSTM smoother returned fewer samples than the smoothing window."
        )

        smooth_state_index = smooth_window_len - 1
        # get_lstm_states indexes the smoothed num_samples buffer after smoother().
        model.lstm_output_history.set(
            np.array([mu_smooth[smooth_state_index]], dtype=np.float32),
            np.array([var_smooth[smooth_state_index]], dtype=np.float32),
        )
        model.lstm_net.set_lstm_states(
            model.lstm_net.get_lstm_states(smooth_state_index)
        )
        return mu_smooth, var_smooth

    post_mu = states.get_mean("lstm", states_type="posterior")
    post_std = states.get_std("lstm", states_type="posterior")
    model.lstm_output_history.set(
        np.array([post_mu[-1]], dtype=np.float32),
        np.array([post_std[-1] ** 2], dtype=np.float32),
    )
    return post_mu, post_std**2


def store_parameter_diagnostics(
    prior_state: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    posterior_state: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    kl_history: Dict[str, Dict[str, list]],
    wasserstein_history: Dict[str, Dict[str, list]],
) -> None:
    kl_results = {
        layer: compute_layer_kl(prior_state[layer], posterior_state[layer])
        for layer in prior_state
    }
    mean_kl = {
        layer: {part: np.mean(values[part]) for part in values}
        for layer, values in kl_results.items()
    }
    for lyr in mean_kl:
        for part in mean_kl[lyr]:
            kl_history[lyr][part].append(mean_kl[lyr][part])

    w_results = {
        layer: compute_layer_wasserstein(prior_state[layer], posterior_state[layer])
        for layer in prior_state
    }
    mean_w = {
        layer: {part: np.mean(values[part]) for part in values}
        for layer, values in w_results.items()
    }
    for lyr in mean_w:
        for part in mean_w[lyr]:
            wasserstein_history[lyr][part].append(mean_w[lyr][part])


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.pgf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")


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
        look_back_len=1,
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

if model.lstm_net.smooth:
    model.lstm_net.num_samples = D

# extract the training data
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# store online predictions and state histories
num_train = len(train_data["y"])
if num_train <= D:
    raise ValueError("Training data must be longer than the smoothing window D.")

mu_preds = []
std_preds = []
pred_indices = []
diagnostic_indices = []

post_lstm_mu = np.full(num_train, np.nan, dtype=np.float32)
post_lstm_std = np.full(num_train, np.nan, dtype=np.float32)
smooth_lstm_mu = np.full(num_train, np.nan, dtype=np.float32)
smooth_lstm_var = np.full(num_train, np.nan, dtype=np.float32)
# history of mean KL divergence per layer (weights / bias)
kl_history = collections.defaultdict(lambda: {"weights": [], "bias": []})
wasserstein_history = collections.defaultdict(lambda: {"weights": [], "bias": []})


def run_online_window(window_start: int, window_end: int):
    prior_state = copy.deepcopy(model.lstm_net.state_dict())
    training_window = make_window(train_data, window_start, window_end)

    model.lstm_net.train()
    mu_filt, std_filt, states = model.filter(training_window, train_lstm=True)

    post_mu = states.get_mean("lstm", states_type="posterior")
    post_std = states.get_std("lstm", states_type="posterior")
    window_len = window_end - window_start
    post_len = min(window_len, len(post_mu))
    post_slice = slice(window_end - post_len, window_end)
    post_lstm_mu[post_slice] = post_mu[-post_len:]
    post_lstm_std[post_slice] = post_std[-post_len:]

    mu_smooth, var_smooth = smooth_lstm_window_and_set_end_state(model, states, D)
    smooth_len = min(window_len, len(mu_smooth))
    smooth_slice = slice(window_end - smooth_len, window_end)
    smooth_lstm_mu[smooth_slice] = mu_smooth[-smooth_len:]
    smooth_lstm_var[smooth_slice] = var_smooth[-smooth_len:]

    posterior_state = model.lstm_net.state_dict()
    store_parameter_diagnostics(
        prior_state,
        posterior_state,
        kl_history,
        wasserstein_history,
    )
    diagnostic_indices.append(window_end - 1)
    return mu_filt, std_filt, states


# Bootstrap with the first D observations, then predict the next one.
run_online_window(0, D)
mu_pred, std_pred = forecast_one_step_without_committing_state(model, train_data, D)
mu_preds.append(mu_pred)
std_preds.append(std_pred)
pred_indices.append(D)

for next_idx in range(D, num_train):
    # y[next_idx] is now available; replay the D-window ending at next_idx.
    window_start = next_idx - D + 1
    window_end = next_idx + 1
    run_online_window(window_start, window_end)

    forecast_idx = next_idx + 1
    if forecast_idx < num_train:
        mu_pred, std_pred = forecast_one_step_without_committing_state(
            model, train_data, forecast_idx
        )
        mu_preds.append(mu_pred)
        std_preds.append(std_pred)
        pred_indices.append(forecast_idx)


# forecast a multi step ahead
# mu_forecast, std_forecast, _ = model.filter(validation_data, train_lstm=False)
model.lstm_net.eval()
mu_forecast, std_forecast, _ = model.forecast(validation_data)


# ------------------------------------------------------------
#  Plot predictions and residuals
# ------------------------------------------------------------
fig, (ax_pred, ax_res) = plt.subplots(
    2,
    1,
    figsize=DOUBLE_COL,
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]},
)

ax_pred.plot(
    np.arange(num_train),
    train_data["y"].flatten(),
    label="training observations",
    color="tab:red",
)

pred_indices_arr = np.array(pred_indices)
mu_preds_arr = np.array(mu_preds).flatten()
std_preds_arr = np.array(std_preds).flatten()
ax_pred.plot(
    pred_indices_arr,
    mu_preds_arr,
    label="online one-step mean",
    color="tab:blue",
)
ax_pred.fill_between(
    pred_indices_arr,
    mu_preds_arr - std_preds_arr,
    mu_preds_arr + std_preds_arr,
    color="tab:blue",
    alpha=0.3,
    label=r"$\pm 1\sigma$ online",
)

true_vals = validation_data["y"].flatten()
forecast_range = np.arange(num_train, num_train + len(true_vals))
ax_pred.plot(
    forecast_range,
    true_vals,
    label="validation observations",
    color="tab:red",
    linestyle="--",
)
ax_pred.plot(
    forecast_range,
    mu_forecast.flatten(),
    label="validation forecast mean",
    color="tab:blue",
    linestyle="--",
)
ax_pred.fill_between(
    forecast_range,
    mu_forecast.flatten() - std_forecast.flatten(),
    mu_forecast.flatten() + std_forecast.flatten(),
    color="tab:blue",
    alpha=0.3,
    label=r"$\pm 1\sigma$ forecast",
)

ax_pred.set_ylabel("Value")
ax_pred.grid(True, alpha=0.25, linewidth=0.5)
ax_pred.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
)

one_step_true = train_data["y"][pred_indices_arr].flatten()
residual_one_step = one_step_true - mu_preds_arr
ax_res.plot(
    pred_indices_arr,
    residual_one_step,
    label="online one-step",
    color="tab:blue",
    linestyle="-",
)

residual_forecast = true_vals - mu_forecast.flatten()
ax_res.plot(
    forecast_range,
    residual_forecast,
    label="validation forecast",
    color="tab:blue",
    linestyle="--",
)

ax_res.axhline(0.0, color="black", linewidth=0.8)
ax_res.set_xlabel("Time step")
ax_res.set_ylabel("Residual")
ax_res.grid(True, alpha=0.25, linewidth=0.5)
ax_res.legend(
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
)

fig.tight_layout()
save_figure(fig, "toy_online_lstm_predictions")

# ------------------------------------------------------------------
#  Plot KL divergence history over training windows
# ------------------------------------------------------------------
fig_kl, ax_kl = plt.subplots(figsize=SINGLE_COL)
for lyr, parts in kl_history.items():
    if lyr == "SLinear.1":
        # skip linear layer
        continue
    kl_total = np.array(parts["weights"]) + np.array(parts["bias"])
    ax_kl.plot(
        np.array(diagnostic_indices[: len(kl_total)]),
        kl_total,
        label=f"{lyr} (w+b)",
    )
ax_kl.set_xlabel("Window end index")
ax_kl.set_ylabel("Mean KL divergence")
ax_kl.grid(True, alpha=0.25, linewidth=0.5)
ax_kl.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
)
fig_kl.tight_layout()
save_figure(fig_kl, "toy_online_lstm_kl")

# ------------------------------------------------------------------
#  Plot Wasserstein distance history over training windows
# ------------------------------------------------------------------
fig_w, ax_w = plt.subplots(figsize=SINGLE_COL)
for lyr, parts in wasserstein_history.items():
    if lyr == "SLinear.1":
        # skip linear layer
        continue
    w_total = np.array(parts["weights"]) + np.array(parts["bias"])
    ax_w.plot(
        np.array(diagnostic_indices[: len(w_total)]),
        w_total,
        label=f"{lyr} (w+b)",
    )
ax_w.set_xlabel("Window end index")
ax_w.set_ylabel("Mean 2-Wasserstein distance")
ax_w.grid(True, alpha=0.25, linewidth=0.5)
ax_w.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
)
fig_w.tight_layout()
save_figure(fig_w, "toy_online_lstm_wasserstein")
