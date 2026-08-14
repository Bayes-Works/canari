"""
Online LSTM on a stationary periodic time series.

There is no epoch loop: the model walks forward through the training set one observation
at a time, and every step of that walk is a window of D filtered steps that is smoothed:

    filter D steps -> filter one additional t+1 step -> smooth the D window
    -> restart from the second step of that window -> repeat

The extra t+1 step serves two purposes: smoothing time step t needs the prior at t+1, so
it closes the backward recursion over the whole D window, and its prediction is a
genuine one-step-ahead forecast, since `Model.filter` predicts before it uses the
observation.

That walk covers the training span, which here is everything but the last 10 % of the
series: there is no epoch loop and nothing to early-stop, so there is no validation split
to hold back. The test span that follows is forecast in one open loop, and every figure
covers both, so what the hidden states do once the observations stop can be read against
what they did while they were arriving.

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
    Splits,
    generate_periodic_signal,
    initialize_state_records,
    make_window,
    plot_parameter_diagnostics,
    plot_predictions_and_residuals,
    plot_state_records,
    print_regime_metrics,
    record_forecast_states,
    record_window_states,
    regime_metrics,
    rewind_to_step,
    set_output_subdir,
    smooth_window,
    store_parameter_diagnostics,
)

from canari import DataProcess, Model
from canari.component import LstmNetwork, WhiteNoise

# Everything this experiment writes goes under `experiments/out/toy_online_lstm/`.
set_output_subdir("toy_online_lstm")

# ------------------------------------------------------------
#  Generate synthetic data
# ------------------------------------------------------------
# Hourly signal over 30 days, with observation noise so that the one-step-ahead error has
# a floor to be compared against. The signal is stationary: one amplitude, one period, no
# regime change anywhere, so the error is the cost of the scheme itself rather than the
# cost of catching a change.
NUM_TIME_STEPS = 24 * 30
NOISE_STD = 0.2
REGIMES = [
    # (start, amplitude, period in time steps)
    (0, 1.0, 24),
]
CHANGEPOINTS = []

y = generate_periodic_signal(
    num_time_steps=NUM_TIME_STEPS,
    regimes=REGIMES,
    noise_std=NOISE_STD,
    seed=0,
)

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

# Build data processor. There is no validation split: this experiment has no epoch loop
# and nothing to early-stop, so a held-out span in the middle would only shorten the online
# walk for no purpose. Everything up to the test set is walked online, and the test span is
# the one held-out stretch.
data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day"],
    train_split=0.9,
    validation_split=0.0,
    output_col=output_col,
)

# split data
train_data, _, test_data, all_data = data_processor.get_splits()
# Every figure is drawn over the whole series, so the split boundaries travel with it.
splits = Splits.from_processor(data_processor)

# Model. The assumed observation noise is the true noise expressed in the standardized
# units the model works in, so the one-step-ahead MSE can be read against sigma_v ** 2.
sigma_v = float(NOISE_STD / data_processor.scale_const_std[output_col[0]])
model = Model(
    LstmNetwork(
        # Any look-back length works: `LstmLookBackBuffer` carries the smoothed LSTM
        # outputs across windows. `num_features` stays 2 whatever it is, since the input
        # layer is `num_features + look_back_len - 1`, i.e. look_back_len past LSTM
        # outputs plus the one covariate.
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

if not model.lstm_net.smooth:
    raise ValueError("The online loop needs the LSTM smoother to be enabled.")
if not 1 <= RESTART_STEP <= D:
    raise ValueError("RESTART_STEP must be between 1 and D.")

# The LSTM smoother buffer holds the D steps of the window plus the extra t+1 filter
# step that closes the backward recursion.
model.lstm_net.num_samples = D + 1

# store online predictions and state histories
num_train = len(train_data["y"])
num_all = len(all_data["y"])
if num_train <= D:
    raise ValueError("Training data must be longer than the smoothing window D.")

mu_preds = []
std_preds = []
pred_indices = []
diagnostic_indices = []

tracked_states = [name for name in model.states_name if name != "white noise"]
# The records span the whole series: the online loop fills the training part, the forecast
# below fills the test part.
state_records = initialize_state_records(tracked_states, num_all)

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
    # y[window_end - 1] is used in its own update step.
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

# Forecast the test span in one open loop, from the end of the training data to the end of
# the series. A forecast has no observation in it at all, so what it leaves in the records
# is a prior at every step and nothing else.
model.lstm_net.eval()
model.lstm_net.num_samples = num_all - num_train
model.initialize_states_history()
mu_forecast, std_forecast, forecast_states = model.forecast(
    make_window(all_data, num_train, num_all)
)
record_forecast_states(state_records, forecast_states, start=num_train)
forecast_indices = np.arange(num_train, num_all)

# metrics on the standardized data
pred_indices = np.array(pred_indices)
mu_preds = np.array(mu_preds).flatten()
std_preds = np.array(std_preds).flatten()
mu_forecast = np.asarray(mu_forecast).flatten()
std_forecast = np.asarray(std_forecast).flatten()
all_obs = all_data["y"].flatten()
train_obs = train_data["y"].flatten()
test_obs = test_data["y"].flatten()

online_mse = metric.mse(mu_preds, train_obs[pred_indices])
test_mse = metric.mse(mu_forecast, test_obs)
noise_floor = sigma_v**2
print(f"Number of windows       : {num_windows}")
print(f"Noise floor (sigma_v^2) :{noise_floor: 0.4f}")
print(f"Online one-step MSE     :{online_mse: 0.4f}"
      f"  ({online_mse / noise_floor: 0.2f} x floor)"
      f"   [steps {D}-{num_train - 1}]")
print(f"Test forecast MSE       :{test_mse: 0.4f}"
      f"  ({test_mse / noise_floor: 0.2f} x floor)"
      f"   [steps {num_train}-{num_all - 1}]")
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
    observations=all_obs,
    pred_indices=pred_indices,
    mu_preds=mu_preds,
    std_preds=std_preds,
    forecast_indices=forecast_indices,
    mu_forecast=mu_forecast,
    std_forecast=std_forecast,
    stem="toy_online_lstm_predictions",
    splits=splits,
    changepoints=CHANGEPOINTS,
)

plot_state_records(
    records=state_records,
    states_name=tracked_states,
    smooth_lag=D,
    stem="toy_online_lstm_states",
    splits=splits,
    changepoints=CHANGEPOINTS,
)

plot_parameter_diagnostics(
    history=kl_history,
    diagnostic_indices=diagnostic_indices,
    ylabel="Mean KL divergence",
    stem="toy_online_lstm_kl",
    splits=splits,
    changepoints=CHANGEPOINTS,
)

plot_parameter_diagnostics(
    history=wasserstein_history,
    diagnostic_indices=diagnostic_indices,
    ylabel="Mean 2-Wasserstein distance",
    stem="toy_online_lstm_wasserstein",
    splits=splits,
    changepoints=CHANGEPOINTS,
)
