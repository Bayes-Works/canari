"""
Online LSTM from global weights vs from a random initialization.

Both models are LSTM + white noise, both run the same online scheme as
`toy_online_lstm.py`:

    filter D steps -> filter one additional t+1 step -> smooth the D window
    -> restart from the second step of that window -> repeat

They differ only in where their weights start:

* **global** - the weights `toy_frozen_vs_online.py` pretrained, read from
  `saved_params/global_model.bin`, then adapted online on this series;
* **local** - the random initialization, so where the weights start is the only difference
  between the two runs.

Nothing is pretrained here: run `python -m experiments.toy_frozen_vs_online` first, which is
where the global weights come from, and this script only loads them. It is also the only
experiment that loads them - every other one starts from a random initialization.

The target is a fresh draw of the family those weights were fitted on: the same stationary
hourly sine with a 24-step season, on a different noise seed. That is what makes the
comparison a global-vs-local one rather than a model rediscovering its own training data.
The architecture has to match the saved file exactly - `look_back_len` and the hidden width
are baked into it, and `lstm_net.load` refuses a mismatch - so those are taken from
`toy_frozen_vs_online.py`.

Both runs walk the training span online - everything but the last 10 % of the series - and
then forecast the test span in one open loop. The figures cover both, so the head start can
be read where it shows, the first periods, and again at the far end, where the two have had
every chance to converge.

Shared helpers live in `utils.py`. See `online_lstm_explained.md` for the scheme itself.
"""

from typing import Dict

import numpy as np
import pandas as pd
import pytagi.metric as metric

# utils sets MPLCONFIGDIR and owns the matplotlib configuration, so import it before
# anything pulls matplotlib in.
from experiments.utils import (
    GLOBAL_WEIGHTS_PATH,
    PROJECT_ROOT,
    LstmLookBackBuffer,
    Splits,
    generate_periodic_signal,
    make_window,
    plot_error_comparison,
    plot_model_comparison,
    print_regime_metrics,
    regime_metrics,
    rewind_to_step,
    set_output_subdir,
    smooth_window,
)

from canari import DataProcess, Model
from canari.component import LstmNetwork, WhiteNoise

# Everything this experiment writes goes under
# `experiments/out/toy_online_lstm_global_vs_local/`.
set_output_subdir("toy_online_lstm_global_vs_local")

# ------------------------------------------------------------
#  Data: hourly, 30 days, the family the global weights were fitted on
# ------------------------------------------------------------
NUM_TIME_STEPS = 24 * 30
PERIOD = 24  # one day
NOISE_STD = 0.2

# The target: a stationary 24-step season, like the series the global weights come from, but
# a different draw of the observation noise. Nothing changes over the 30 days, so the only
# thing separating the two runs is where their weights start.
TARGET_REGIMES = [
    # (start, amplitude, period in time steps)
    (0, 1.0, PERIOD),
]
CHANGEPOINTS = []

# ------------------------------------------------------------
#  Online scheme and model parameters
# ------------------------------------------------------------
output_col = [0]
D = 48  # length of the smoothing window: two periods
RESTART_STEP = 1  # stride of the online loop
# Baked into `saved_params/global_model.bin`: these two decide the layer shapes, so they
# have to be the ones `toy_frozen_vs_online.py` pretrained with.
LOOK_BACK_LEN = 12
NUM_HIDDEN_UNIT = 40
MANUAL_SEED = 1  # the random initialization the local run starts from
TARGET_SEED = 7  # observation noise of the target, a different draw from the pretraining
TRANSIENT_LEN = 24  # one period, for the per-regime split

# The numbers this prints are one run of one seed. The first-periods column is the one to
# read: it is where a pretrained start can show, and the rest of the series gives the local
# model time to catch up. Re-run over several TARGET_SEED values before reading a winner out
# of the overall online error or the test forecast.


def build_data_processor(values: np.ndarray) -> DataProcess:
    """A `DataProcess` over the series, with the hour-of-day covariate."""

    df = pd.DataFrame(
        {"values": values},
        index=pd.date_range(start="2023-01-01", periods=len(values), freq="h"),
    )
    df.index.name = "date_time"
    # No validation split: nothing is trained in an epoch loop here, so there is nothing to
    # early-stop and nothing a held-out span in the middle would buy. The online walk takes
    # everything up to the test set.
    return DataProcess(
        data=df,
        time_covariates=["hour_of_day"],
        train_split=0.9,
        validation_split=0.0,
        output_col=output_col,
    )


def build_model(sigma_v: float) -> Model:
    """
    An LSTM-only model, randomly initialized from `MANUAL_SEED`. The same seed every time,
    so both runs start from the same weights unless the global ones are loaded over them.
    """

    return Model(
        LstmNetwork(
            look_back_len=LOOK_BACK_LEN,
            num_features=2,  # past LSTM outputs + the one covariate
            infer_len=PERIOD,
            num_layer=1,
            num_hidden_unit=NUM_HIDDEN_UNIT,
            device="cpu",
            manual_seed=MANUAL_SEED,
        ),
        WhiteNoise(std_error=sigma_v),
    )


def run_online(model: Model, train_data: Dict, all_data: Dict) -> Dict:
    """
    The online scheme of `toy_online_lstm.py` over the training span, then one open-loop
    forecast over the test span that follows it.

    Kept as a function here because this experiment runs it twice, on two models that must
    be treated identically.
    """

    num_train = len(train_data["y"])
    num_all = len(all_data["y"])
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
    model.lstm_net.num_samples = num_all - num_train
    model.initialize_states_history()
    mu_forecast, std_forecast, _ = model.forecast(
        make_window(all_data, num_train, num_all)
    )

    return {
        "pred_indices": np.array(pred_indices),
        "mu_preds": np.array(mu_preds).flatten(),
        "std_preds": np.array(std_preds).flatten(),
        "forecast_indices": np.arange(num_train, num_all),
        "mu_forecast": np.asarray(mu_forecast).flatten(),
        "std_forecast": np.asarray(std_forecast).flatten(),
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
train_data, _, test_data, all_data = target_processor.get_splits()
# Every figure is drawn over the whole series, so the split boundaries travel with it.
splits = Splits.from_processor(target_processor)

num_train = len(train_data["y"])
if num_train <= D:
    raise ValueError("Training data must be longer than the smoothing window D.")
if not 1 <= RESTART_STEP <= D:
    raise ValueError("RESTART_STEP must be between 1 and D.")

# The assumed observation noise is the true noise in the standardized units the model works
# in, so the MSEs can be read against sigma_v ** 2.
sigma_v = float(NOISE_STD / target_processor.scale_const_std[output_col[0]])
noise_floor = sigma_v**2

print(f"Hourly steps            : {NUM_TIME_STEPS}  (online walk {num_train}, "
      f"test {len(test_data['y'])}, noise seed {TARGET_SEED})")
print(f"Noise floor (sigma_v^2) :{noise_floor: 0.4f}")

if not GLOBAL_WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"{GLOBAL_WEIGHTS_PATH.relative_to(PROJECT_ROOT)} is missing. Nothing is "
        "pretrained here: run `python -m experiments.toy_frozen_vs_online` first, which "
        "writes the global weights this experiment loads."
    )
print(f"Global weights          : {GLOBAL_WEIGHTS_PATH.relative_to(PROJECT_ROOT)}")

# ------------------------------------------------------------
#  Run the online scheme from each starting point
# ------------------------------------------------------------
# `load` overwrites the random initialization with the pretrained weights and raises if the
# architecture does not match the file, which is the check that keeps LOOK_BACK_LEN and
# NUM_HIDDEN_UNIT honest above.
runs = []
for name in ("global", "local"):
    model = build_model(sigma_v)
    if name == "global":
        model.lstm_net.load(str(GLOBAL_WEIGHTS_PATH))
    model.lstm_net.num_samples = D + 1
    if not model.lstm_net.smooth:
        raise ValueError("The online loop needs the LSTM smoother to be enabled.")

    result = run_online(model, train_data, all_data)
    result["name"] = name
    runs.append(result)

# ------------------------------------------------------------
#  Metrics
# ------------------------------------------------------------
all_obs = all_data["y"].flatten()
train_obs = train_data["y"].flatten()
test_obs = test_data["y"].flatten()
early_len = 4 * PERIOD  # the stretch where a pretrained start can still show

print(f"\nOnline windows per model: {runs[0]['num_windows']}")
print(f"  {'start':<8}{'online':>10}{'x floor':>9}{'first days':>12}"
      f"{'test':>10}{'x floor':>9}")
for run in runs:
    online_mse = metric.mse(run["mu_preds"], train_obs[run["pred_indices"]])
    test_mse = metric.mse(run["mu_forecast"], test_obs)
    early = run["pred_indices"] < D + early_len
    early_mse = metric.mse(
        run["mu_preds"][early], train_obs[run["pred_indices"][early]]
    )
    run["online_mse"] = online_mse
    run["test_mse"] = test_mse
    print(
        f"  {run['name']:<8}{online_mse:>10.4f}{online_mse / noise_floor:>9.2f}"
        f"{early_mse / noise_floor:>12.2f}"
        f"{test_mse:>10.4f}{test_mse / noise_floor:>9.2f}"
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
            transient_len=TRANSIENT_LEN,
        ),
        noise_floor=noise_floor,
        transient_len=TRANSIENT_LEN,
    )

# ------------------------------------------------------------
#  Figures
# ------------------------------------------------------------
plot_model_comparison(
    runs=runs,
    observations=all_obs,
    stem="toy_online_lstm_global_vs_local_predictions",
    splits=splits,
    changepoints=CHANGEPOINTS,
)

plot_error_comparison(
    runs=runs,
    observations=all_obs,
    stem="toy_online_lstm_global_vs_local_error",
    splits=splits,
    changepoints=CHANGEPOINTS,
    noise_floor=noise_floor,
)
