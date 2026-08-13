"""
Online LSTM from global weights vs from a random initialization, on weekly data.

Both models are LSTM + white noise, both run the same online scheme as
`toy_online_lstm.py`:

    filter D steps -> filter one additional t+1 step -> smooth the D window
    -> restart from the second step of that window -> repeat

They differ only in where their weights start:

* **global** - pretrained offline over a pool of related weekly series that share the
  52-week seasonal shape of the target, then adapted online on the target;
* **local** - started from the random initialization the global model itself started from,
  so the pretraining is the only difference between the two runs.

The data is weekly with a `week_of_year` covariate, eight years of it, and the target is
stationary: a fixed amplitude and a fixed 52-week period throughout. What the comparison
measures is therefore how much of a head start the pretrained weights give, not how the
two starts cope with a change.

Shared helpers live in `utils.py`. See `online_lstm_explained.md` for the scheme itself.
"""

import copy
from typing import Dict, List

import numpy as np
import pandas as pd
import pytagi.metric as metric

# utils sets MPLCONFIGDIR and owns the matplotlib configuration, so import it before
# anything pulls matplotlib in.
from experiments.utils import (
    LstmLookBackBuffer,
    drain_lstm_smoother_buffer,
    generate_periodic_signal,
    make_window,
    plot_error_comparison,
    plot_model_comparison,
    print_regime_metrics,
    regime_metrics,
    reset_lstm_memory,
    rewind_to_step,
    smooth_window,
)

from canari import DataProcess, Model
from canari.component import LstmNetwork, WhiteNoise

# ------------------------------------------------------------
#  Data: weekly, 6 years, one sample per week
# ------------------------------------------------------------
NUM_TIME_STEPS = 52 * 8
PERIOD = 52  # one year
NOISE_STD = 0.15
START_DATE = "2018-01-07"  # a Sunday, so the weekly index is clean

# The target: a stationary 52-week season. Nothing changes over the eight years, so the
# only thing separating the two runs is where their weights start.
TARGET_REGIMES = [
    # (start, amplitude, period in time steps)
    (0, 1.0, PERIOD),
]
CHANGEPOINTS = []

# The pool the global model is pretrained on: the same annual season, but each series has
# its own noise level and its own amount of a second harmonic, so the global model learns a
# family of annual shapes rather than one exact sine. Standardization removes plain
# amplitude differences, so the diversity has to be in the shape and the noise.
POOL = [
    # (harmonic weight, noise std, seed)
    (0.00, 0.10, 11),
    (0.15, 0.15, 12),
    (0.30, 0.20, 13),
    (0.10, 0.25, 14),
    (0.25, 0.12, 15),
]

# ------------------------------------------------------------
#  Online scheme and model parameters
# ------------------------------------------------------------
output_col = [0]
D = 52  # length of the smoothing window: one year
RESTART_STEP = 1  # stride of the online loop
LOOK_BACK_LEN = 1
NUM_HIDDEN_UNIT = 50
PRETRAIN_EPOCHS = 20
MANUAL_SEED = 1  # weight initialization, shared by both runs
TARGET_SEED = 0  # observation noise of the target series
ROLLING_WINDOW = 13  # a quarter of a year, for the error comparison figure

# The numbers this prints are one run of one seed. The first-year column is the one to
# read: it is where a pretrained start can show, and the rest of the series gives the
# local model time to catch up. Re-run over several TARGET_SEED values before reading a
# winner out of the overall online error or the validation forecast.


def build_data_processor(values: np.ndarray) -> DataProcess:
    """A `DataProcess` over one weekly series, with the week-of-year covariate."""

    df = pd.DataFrame(
        {"values": values},
        index=pd.date_range(start=START_DATE, periods=len(values), freq="W"),
    )
    df.index.name = "date_time"
    return DataProcess(
        data=df,
        time_covariates=["week_of_year"],
        train_split=0.8,
        validation_split=0.1,
        output_col=output_col,
    )


def build_model(sigma_v: float) -> Model:
    """
    An LSTM-only model. The same seed every time, so every run starts from the same
    weights unless a state dict is loaded over them.
    """

    return Model(
        LstmNetwork(
            look_back_len=LOOK_BACK_LEN,
            num_features=2,  # one past LSTM output + the week-of-year covariate
            infer_len=PERIOD,
            num_layer=1,
            num_hidden_unit=NUM_HIDDEN_UNIT,
            device="cpu",
            manual_seed=MANUAL_SEED,
        ),
        WhiteNoise(std_error=sigma_v),
    )


def run_online(model: Model, train_data: Dict, validation_data: Dict) -> Dict:
    """
    The online scheme of `toy_online_lstm.py`, run on one model.

    Kept as a function here because this experiment runs it twice, on two models that must
    be treated identically.
    """

    num_train = len(train_data["y"])
    num_windows = (num_train - D - 1) // RESTART_STEP + 1
    look_back_buffer = LstmLookBackBuffer(
        look_back_len=model.lstm_net.lstm_look_back_len, num_time_steps=num_train
    )

    mu_preds, std_preds, pred_indices = [], [], []
    for window_index in range(num_windows):
        window_start = window_index * RESTART_STEP
        window_end = window_start + D + 1

        model.lstm_net.train()
        mu_filt, std_filt, _ = model.filter(
            make_window(train_data, window_start, window_end), train_lstm=True
        )

        # the extra t+1 step is a genuine one-step-ahead prediction
        mu_preds.append(mu_filt[-1])
        std_preds.append(std_filt[-1])
        pred_indices.append(window_end - 1)

        mu_smooth_lstm, var_smooth_lstm = smooth_window(model)
        look_back_buffer.store(window_start, mu_smooth_lstm, var_smooth_lstm)

        if window_index < num_windows - 1:
            rewind_to_step(model, look_back_buffer, window_start, RESTART_STEP)
        else:
            # keep the filtered end-of-window memory: the smoother zeroed the states
            model.lstm_net.set_lstm_states(model.lstm_net.get_lstm_states(D))

    tail_start = (num_windows - 1) * RESTART_STEP + D + 1
    if tail_start < num_train:
        model.filter(make_window(train_data, tail_start, num_train), train_lstm=True)

    model.lstm_net.eval()
    model.initialize_states_history()
    mu_forecast, std_forecast, _ = model.forecast(validation_data)

    return {
        "pred_indices": np.array(pred_indices),
        "mu_preds": np.array(mu_preds).flatten(),
        "std_preds": np.array(std_preds).flatten(),
        "mu_forecast": mu_forecast.flatten(),
        "std_forecast": std_forecast.flatten(),
        "num_windows": num_windows,
    }


# ------------------------------------------------------------
#  Build the target series
# ------------------------------------------------------------
target_values = generate_periodic_signal(
    num_time_steps=NUM_TIME_STEPS,
    regimes=TARGET_REGIMES,
    noise_std=NOISE_STD,
    seed=TARGET_SEED,
)
target_processor = build_data_processor(target_values)
train_data, validation_data, test_data, all_data = target_processor.get_splits()

num_train = len(train_data["y"])
if num_train <= D:
    raise ValueError("Training data must be longer than the smoothing window D.")
if not 1 <= RESTART_STEP <= D:
    raise ValueError("RESTART_STEP must be between 1 and D.")

# The assumed observation noise is the true noise in the standardized units the model works
# in, so the MSEs can be read against sigma_v ** 2.
sigma_v = float(NOISE_STD / target_processor.scale_const_std[output_col[0]])
noise_floor = sigma_v**2

print(f"Weekly steps            : {NUM_TIME_STEPS}  (train {num_train}, "
      f"validation {len(validation_data['y'])})")
print(f"Noise floor (sigma_v^2) :{noise_floor: 0.4f}")

# ------------------------------------------------------------
#  Pretrain the global weights on the pool
# ------------------------------------------------------------
pool_series: List[Dict] = []
for harmonic_weight, pool_noise, seed in POOL:
    values = generate_periodic_signal(
        num_time_steps=NUM_TIME_STEPS,
        regimes=[(0, 1.0, PERIOD)],
        noise_std=pool_noise,
        seed=seed,
    )
    if harmonic_weight > 0.0:
        values = values + harmonic_weight * generate_periodic_signal(
            num_time_steps=NUM_TIME_STEPS,
            regimes=[(0, 1.0, PERIOD // 2)],
            noise_std=0.0,
        )
    # every series is standardized on its own, exactly as the target is
    _, _, _, series = build_data_processor(values).get_splits()
    pool_series.append(series)

pretrain_model = build_model(sigma_v)
initial_state_dict = copy.deepcopy(pretrain_model.lstm_net.state_dict())

# `num_samples` is the length of the SLSTM sample buffer and has to cover the whole pass
# before filtering starts. `Model` leaves it at a dummy 1, and filtering hundreds of steps
# with that value hangs or aborts inside the C++ backend rather than raising.
pretrain_model.lstm_net.num_samples = NUM_TIME_STEPS

print(f"\nPretraining on {len(pool_series)} pool series for {PRETRAIN_EPOCHS} epochs")
pretrain_model.lstm_net.train()
for epoch in range(PRETRAIN_EPOCHS):
    epoch_mse = []
    for series in pool_series:
        # nothing may leak from one series into the next, only the parameters carry over
        reset_lstm_memory(pretrain_model)
        mu_preds_pool, _, _ = pretrain_model.filter(series, train_lstm=True)
        # the sample buffer has to be released before the next full-length pass
        drain_lstm_smoother_buffer(pretrain_model)
        epoch_mse.append(metric.mse(mu_preds_pool, series["y"].flatten()))
    if epoch % 5 == 0 or epoch == PRETRAIN_EPOCHS - 1:
        mean_mse = float(np.mean(epoch_mse))
        print(f"  epoch {epoch:>3}  pool one-step MSE {mean_mse: 0.4f}"
              f"  ({mean_mse / noise_floor: 0.2f} x target floor)")

global_state_dict = copy.deepcopy(pretrain_model.lstm_net.state_dict())

# ------------------------------------------------------------
#  Run the online scheme from each starting point
# ------------------------------------------------------------
runs = []
for name, state_dict in (
    ("global", global_state_dict),
    ("local", initial_state_dict),
):
    model = build_model(sigma_v)
    model.lstm_net.load_state_dict(state_dict)
    model.lstm_net.num_samples = D + 1
    if not model.lstm_net.smooth:
        raise ValueError("The online loop needs the LSTM smoother to be enabled.")

    result = run_online(model, train_data, validation_data)
    result["name"] = name
    runs.append(result)

# ------------------------------------------------------------
#  Metrics
# ------------------------------------------------------------
train_obs = train_data["y"].flatten()
validation_obs = validation_data["y"].flatten()
first_year = 52

print(f"\nOnline windows per model: {runs[0]['num_windows']}")
print(f"  {'start':<8}{'online':>10}{'x floor':>9}{'first year':>12}"
      f"{'validation':>12}{'x floor':>9}")
for run in runs:
    online_mse = metric.mse(run["mu_preds"], train_obs[run["pred_indices"]])
    validation_mse = metric.mse(run["mu_forecast"], validation_obs)
    early = run["pred_indices"] < D + first_year
    early_mse = metric.mse(
        run["mu_preds"][early], train_obs[run["pred_indices"][early]]
    )
    run["online_mse"] = online_mse
    run["validation_mse"] = validation_mse
    print(
        f"  {run['name']:<8}{online_mse:>10.4f}{online_mse / noise_floor:>9.2f}"
        f"{early_mse / noise_floor:>12.2f}"
        f"{validation_mse:>12.4f}{validation_mse / noise_floor:>9.2f}"
    )

for run in runs:
    print(f"\n{run['name']} model")
    print_regime_metrics(
        regime_metrics(
            pred_indices=run["pred_indices"],
            mu_preds=run["mu_preds"],
            observations=train_obs,
            changepoints=CHANGEPOINTS,
            num_time_steps=num_train,
            transient_len=13,  # a quarter of a year
        ),
        noise_floor=noise_floor,
        transient_len=13,
    )

# ------------------------------------------------------------
#  Figures
# ------------------------------------------------------------
plot_model_comparison(
    runs=runs,
    train_obs=train_obs,
    validation_obs=validation_obs,
    stem="toy_online_lstm_global_vs_local_predictions",
    changepoints=CHANGEPOINTS,
)

plot_error_comparison(
    runs=runs,
    observations=train_obs,
    stem="toy_online_lstm_global_vs_local_error",
    window=ROLLING_WINDOW,
    changepoints=CHANGEPOINTS,
    noise_floor=noise_floor,
)
