"""
Online LSTM on a trending time series.

Same online scheme as `toy_online_lstm.py`:

    filter D steps -> filter one additional t+1 step -> smooth the D window
    -> restart from the second step of that window -> repeat

but the synthetic signal now carries a constant linear trend, and a `LocalTrend` component
is added to the model to take care of it. At every step the prediction of the observation
couples the baseline states with the LSTM's t+1 prediction:

    y(t+1) = level(t+1) + lstm(t+1) + white noise

`Model.forward` inserts the LSTM's t+1 prediction into the prior state vector at the
lstm index, and the lstm transition row is zero, so the lstm state never carries over
from one step to the next: the coupled quantity is always the newest LSTM prediction.

Shared helpers live in `utils.py`. See `online_lstm_explained.md` for the details.
"""

import collections
import copy

import numpy as np
import pandas as pd
import pytagi.metric as metric

# utils sets MPLCONFIGDIR and owns the matplotlib configuration, so import it before
# anything pulls matplotlib in.
from experiments.utils import (
    LstmLookBackBuffer,
    add_piecewise_trend,
    generate_periodic_signal,
    initialize_state_records,
    make_window,
    plot_decomposition,
    plot_decomposition_rows,
    plot_parameter_diagnostics,
    plot_predictions_and_residuals,
    plot_state_records,
    print_regime_metrics,
    record_window_states,
    regime_metrics,
    rewind_to_step,
    smooth_window,
    store_parameter_diagnostics,
)

from canari import DataProcess, Model
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

# ------------------------------------------------------------
#  Generate synthetic data: periodic signal on top of a trend
# ------------------------------------------------------------
# Hourly signal over 30 days with observation noise. Nothing changes over the series: one
# amplitude, one period, and a single constant slope from the first step to the last. The
# only thing the `LocalTrend` has to do is find that slope and hold it, so the trend and
# level states can be read against a known constant.
NUM_TIME_STEPS = 24 * 30
NOISE_STD = 0.2
REGIMES = [
    # (start, amplitude, period in time steps)
    (0, 1.0, 24),
]
SLOPES = [
    # (start, slope per time step) - one entry, so the trend is constant
    (0, 0.006),
]
CHANGEPOINTS = []

y = generate_periodic_signal(
    num_time_steps=NUM_TIME_STEPS,
    regimes=REGIMES,
    noise_std=NOISE_STD,
    seed=0,
)
y = add_piecewise_trend(y, SLOPES)

# Create a DataFrame
df = pd.DataFrame(
    {"values": y},
    index=pd.date_range(start="2023-01-01", periods=NUM_TIME_STEPS, freq="H"),
)
df.index.name = "date_time"
df.columns = ["values"]


# Define parameters
output_col = [0]
D = 48  # length of smooth window
# Step of the previous window each new window restarts from, i.e. the stride of the
# online loop. 1 means every time step gets its own one-step-ahead prediction.
RESTART_STEP = 1

# Build data processor
data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day"],
    train_split=0.8,
    validation_split=0.1,
    output_col=output_col,
)

# split data
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Model: a local trend for the baseline, an LSTM for the periodic pattern.
#
# Only the sum of the baseline and the LSTM output is observed, so the two can drift
# against each other: with a loose prior on the slope, the level follows the periodic
# signal and the LSTM output compensates. Predictions stay good but the decomposition is
# meaningless and the multi-step forecast extrapolates a bogus trend. Keeping the slope
# prior tight, together with a small process noise so it can still adapt slowly, leaves
# the periodic pattern to the LSTM.
# The assumed observation noise is the true noise expressed in the standardized units the
# model works in, so the one-step-ahead MSE can be read against sigma_v ** 2.
sigma_v = float(NOISE_STD / data_processor.scale_const_std[output_col[0]])
model = Model(
    LocalTrend(var_states=[1e-1, 1e-3], std_error=1e-5),
    LstmNetwork(
        # Any look-back length works: `LstmLookBackBuffer` carries the smoothed LSTM
        # outputs across windows. `num_features` stays 2 whatever it is, since the input
        # layer is `num_features + look_back_len - 1`, i.e. look_back_len past LSTM
        # outputs plus the one covariate.
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

# Initial level and trend from a decomposition of the first periods. `decompose_data`
# estimates the seasonality by averaging the samples that share the same phase, so it
# needs several periods: on a single period it returns a zero slope, and the trend then
# has to be recovered online from an initialization that says "no trend".
model.auto_initialize_baseline_states(train_data["y"][0 : 24 * 8])

if not model.lstm_net.smooth:
    raise ValueError("The online loop needs the LSTM smoother to be enabled.")
if not 1 <= RESTART_STEP <= D:
    raise ValueError("RESTART_STEP must be between 1 and D.")

# The LSTM smoother buffer holds the D steps of the window plus the extra t+1 filter
# step that closes the backward recursion.
model.lstm_net.num_samples = D + 1

# store online predictions and state histories
num_train = len(train_data["y"])
if num_train <= D:
    raise ValueError("Training data must be longer than the smoothing window D.")

mu_preds = []
std_preds = []
pred_indices = []
diagnostic_indices = []

tracked_states = [name for name in model.states_name if name != "white noise"]
state_records = initialize_state_records(tracked_states, num_train)

# smoothed LSTM outputs the next window looks back into
look_back_buffer = LstmLookBackBuffer(
    look_back_len=model.lstm_net.lstm_look_back_len, num_time_steps=num_train
)

# history of mean KL divergence per layer (weights / bias)
kl_history = collections.defaultdict(lambda: {"weights": [], "bias": []})
wasserstein_history = collections.defaultdict(lambda: {"weights": [], "bias": []})


# Online loop:
#   filter D steps, filter one additional t+1 step, smooth the D window,
#   restart RESTART_STEP steps into that window, and repeat until the training
#   data runs out.
num_windows = (num_train - D - 1) // RESTART_STEP + 1

for window_index in range(num_windows):
    # The window starts where the previous one was rewound to.
    window_start = window_index * RESTART_STEP
    # D filter steps, plus the extra t+1 filter step at the end of the window.
    window_end = window_start + D + 1
    training_window = make_window(train_data, window_start, window_end)

    prior_state = copy.deepcopy(model.lstm_net.state_dict())

    model.lstm_net.train()
    mu_filt, std_filt, states = model.filter(training_window, train_lstm=True)

    # The extra filter step is a genuine one-step-ahead prediction: it is made before
    # y[window_end - 1] is used in its own update step. Its prior couples the baseline
    # states with the newest LSTM prediction.
    pred_index = window_end - 1
    mu_preds.append(mu_filt[-1])
    std_preds.append(std_filt[-1])
    pred_indices.append(pred_index)

    mu_smooth_lstm, var_smooth_lstm = smooth_window(model)

    # Keep the full-lag smoothed output of this window's first step: later windows look
    # back into it once the look-back reaches past their own start.
    look_back_buffer.store(window_start, mu_smooth_lstm, var_smooth_lstm)

    record_window_states(
        state_records,
        states,
        window_start=window_start,
        smooth_len=D,
        is_last_window=window_index == num_windows - 1,
    )

    store_parameter_diagnostics(
        prior_state,
        model.lstm_net.state_dict(),
        kl_history,
        wasserstein_history,
    )
    diagnostic_indices.append(pred_index)

    if window_index < num_windows - 1:
        rewind_to_step(model, look_back_buffer, window_start, RESTART_STEP)
    else:
        # Last window: keep the filtered end-of-window memory for the forecast below.
        # `lstm_net.smoother()` clears the current cell/hidden states, and the last
        # entry of the smoothed buffer is the filtered one.
        model.lstm_net.set_lstm_states(model.lstm_net.get_lstm_states(D))


# With RESTART_STEP > 1 the last window can end before the last training observation.
# Filter the remaining steps, without smoothing them, so that the forecast starts from the
# end of the training data instead of from wherever the last window happened to stop. No
# effect when RESTART_STEP = 1, where the last window already ends at num_train.
tail_start = (num_windows - 1) * RESTART_STEP + D + 1
if tail_start < num_train:
    print(f"Uncovered tail steps    : {num_train - tail_start} (filtered, not smoothed)")
    model.lstm_net.train()
    model.filter(make_window(train_data, tail_start, num_train), train_lstm=True)

# forecast a multi step ahead
model.lstm_net.eval()
model.initialize_states_history()
mu_forecast, std_forecast, _ = model.forecast(validation_data)

# metrics on the standardized data
pred_indices = np.array(pred_indices)
mu_preds = np.array(mu_preds).flatten()
std_preds = np.array(std_preds).flatten()
train_obs = train_data["y"].flatten()
validation_obs = validation_data["y"].flatten()

online_mse = metric.mse(mu_preds, train_obs[pred_indices])
validation_mse = metric.mse(mu_forecast, validation_obs)
noise_floor = sigma_v**2
print(f"Number of windows       : {num_windows}")
print(f"Noise floor (sigma_v^2) :{noise_floor: 0.4f}")
print(f"Online one-step MSE     :{online_mse: 0.4f}"
      f"  ({online_mse / noise_floor: 0.2f} x floor)")
print(f"Validation forecast MSE :{validation_mse: 0.4f}"
      f"  ({validation_mse / noise_floor: 0.2f} x floor)")
print()
print_regime_metrics(
    regime_metrics(
        pred_indices=pred_indices,
        mu_preds=mu_preds,
        observations=train_obs,
        changepoints=CHANGEPOINTS,
        num_time_steps=num_train,
    ),
    noise_floor=noise_floor,
)


# ------------------------------------------------------------
#  Figures
# ------------------------------------------------------------
plot_predictions_and_residuals(
    train_obs=train_obs,
    pred_indices=pred_indices,
    mu_preds=mu_preds,
    std_preds=std_preds,
    validation_obs=validation_obs,
    mu_forecast=mu_forecast.flatten(),
    std_forecast=std_forecast.flatten(),
    stem="toy_online_lstm_trend_predictions",
    changepoints=CHANGEPOINTS,
)

plot_state_records(
    records=state_records,
    states_name=tracked_states,
    smooth_lag=D,
    stem="toy_online_lstm_trend_states",
    changepoints=CHANGEPOINTS,
)

# The observation prediction is level + lstm, so the sum should lie on the data.
plot_decomposition(
    records=state_records,
    component_names=["level", "lstm"],
    observations=train_obs,
    stem="toy_online_lstm_trend_decomposition",
    changepoints=CHANGEPOINTS,
)

# Stacked decomposition, following the one-state-per-row convention used by
# ``canari.plot_states`` in the examples. The trend is shown as an SSM transition state,
# but only level + lstm is reconstructed on the observation row.
plot_decomposition_rows(
    records=state_records,
    states_name=["level", "trend", "lstm"],
    observation_component_names=["level", "lstm"],
    observations=train_obs,
    stem="toy_online_lstm_trend_decomposition_rows",
    changepoints=CHANGEPOINTS,
)

plot_parameter_diagnostics(
    history=kl_history,
    diagnostic_indices=diagnostic_indices,
    ylabel="Mean KL divergence",
    stem="toy_online_lstm_trend_kl",
    changepoints=CHANGEPOINTS,
)

plot_parameter_diagnostics(
    history=wasserstein_history,
    diagnostic_indices=diagnostic_indices,
    ylabel="Mean 2-Wasserstein distance",
    stem="toy_online_lstm_trend_wasserstein",
    changepoints=CHANGEPOINTS,
)
