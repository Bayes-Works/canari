"""
Frozen weights vs online weights on a *stationary* test set.

One stationary series, split 60 / 40: the first 60 % is used for offline training (50 %
train, 10 % validation) and the last 40 % is the test set neither scenario ever trains on
beforehand. Both scenarios start from the *same* pretrained weights, produced by one
epoch loop with early stopping, and both are scored on exactly the same test steps.

    frozen   after pretraining the weights are fixed. The test set is filtered with
             `train_lstm=False`: the hidden states still update at every step, the
             network does not.

    online   after the same pretraining, the test set is walked with the window scheme of
             `toy_online_lstm.py` - filter D steps, filter one extra t+1 step, smooth the
             window backwards, rewind one step - with `train_lstm=True`, so the weights
             keep being updated from the test observations as they arrive.

    control  the same window scheme with `train_lstm=False`. It shares every mechanism
             with the online run and learns nothing, so it separates what the scheme
             costs from what the updating buys.

The online windows start `D` steps *before* the test set, so the extra t+1 step of the
first window lands exactly on the first test observation and every test step gets a
one-step-ahead prediction. All three are given the identical warm-up over
`[0, test_start - D)` with the weights frozen, so the only difference between them is
what happens from `test_start - D` onwards.

Nothing about this series changes, so the pretrained weights never go stale and there is
nothing for the online updates to adapt to. `toy_frozen_vs_online_shift.py` is the same
comparison on a series that does change inside the test set.

Run from the repository root:

    python -m experiments.toy_frozen_vs_online          # data seed 0
    python -m experiments.toy_frozen_vs_online 3        # another draw of the noise

Shared helpers live in `utils.py`. See `online_lstm.md` for the scheme itself.
"""

import sys

import numpy as np
import pandas as pd
import pytagi.metric as metric

# utils sets MPLCONFIGDIR and owns the matplotlib configuration, so import it before
# anything pulls matplotlib in.
from experiments.utils import (
    DOUBLE_COL,
    generate_periodic_signal,
    make_window,
    plot_error_comparison,
    plot_prior_vs_posterior,
    pretrain_lstm,
    print_calibration,
    run_online_windows,
    save_figure,
    warm_up_filter,
)

import matplotlib.pyplot as plt  # noqa: E402

from canari import DataProcess, Model  # noqa: E402
from canari.component import LstmNetwork, WhiteNoise  # noqa: E402

# ------------------------------------------------------------
#  Data: stationary, hourly, 30 days
# ------------------------------------------------------------
NUM_TIME_STEPS = 24 * 30
NOISE_STD = 0.2
REGIMES = [(0, 1.0, 24)]  # (start, amplitude, period in time steps)
# Observation-noise seed. `python -m experiments.toy_frozen_vs_online 3` reruns the whole
# comparison on a different draw; the weight initialization is unaffected, so the only
# thing that changes is the data.
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# The observation noise the model is *told* about, as a fraction of the true noise. At
# 1.0 the model is given the truth. Below 1.0 it is told the data is cleaner than it is,
# which raises the Kalman gain `var_prior / (var_prior + sigma_v^2)` and so lets the
# update pull the state further onto each observation.
SIGMA_V_FACTOR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

# 50 % train + 10 % validation = the 60 % used for offline training, 40 % test.
TRAIN_SPLIT = 0.5
VALIDATION_SPLIT = 0.1

# ------------------------------------------------------------
#  Scheme and model parameters
# ------------------------------------------------------------
output_col = [0]
D = 48  # length of the smoothing window of the online scenarios
LOOK_BACK_LEN = 12
NUM_HIDDEN_UNIT = 40
INFER_LEN = 24  # one period
NUM_EPOCH = 50
MANUAL_SEED = 1
ROLLING_WINDOW = 24  # one period, for the rolling-error figure

y = generate_periodic_signal(
    num_time_steps=NUM_TIME_STEPS,
    regimes=REGIMES,
    noise_std=NOISE_STD,
    seed=SEED,
)
df = pd.DataFrame(
    {"values": y},
    index=pd.date_range(start="2023-01-01", periods=NUM_TIME_STEPS, freq="h"),
)
df.index.name = "date_time"

data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day"],
    train_split=TRAIN_SPLIT,
    validation_split=VALIDATION_SPLIT,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

TEST_START = data_processor.test_start
TEST_END = data_processor.test_end
NUM_TEST = TEST_END - TEST_START
WARMUP_END = TEST_START - D  # where the online scheme's first window starts

if WARMUP_END <= LOOK_BACK_LEN:
    raise ValueError("Not enough data before the test set for the warm-up.")

test_obs = test_data["y"].flatten()
# The same series without observation noise, standardized identically. Only
# synthetic data has this; it is what lets `print_calibration` check how the
# predictive variance is split between the state and the noise term.
clean_test = (
    generate_periodic_signal(
        num_time_steps=NUM_TIME_STEPS, regimes=REGIMES, noise_std=0.0
    )[TEST_START:TEST_END]
    - float(data_processor.scale_const_mean[output_col[0]])
) / float(data_processor.scale_const_std[output_col[0]])
# The true noise in the standardized units the model works in. Every MSE below is
# reported against this, whatever the model was told, so the numbers stay comparable
# across values of `SIGMA_V_FACTOR`.
true_sigma_v = float(NOISE_STD / data_processor.scale_const_std[output_col[0]])
noise_floor = true_sigma_v**2
sigma_v = true_sigma_v * SIGMA_V_FACTOR

print(f"Steps                   : {NUM_TIME_STEPS}  (noise seed {SEED})")
print(f"  train                 : [0, {data_processor.train_end}) "
      f"({len(train_data['y'])} steps)")
print(f"  validation            : [{data_processor.validation_start}, {TEST_START}) "
      f"({len(validation_data['y'])} steps)")
print(f"  test                  : [{TEST_START}, {TEST_END}) ({NUM_TEST} steps)")
print(f"  online warm-up ends at: {WARMUP_END}  (first window covers "
      f"[{WARMUP_END}, {TEST_START}], first prediction at {TEST_START})")
print(f"True noise variance     :{noise_floor: 0.4f}  (every MSE is reported against it)")
print(f"Assumed by the model    :{sigma_v**2: 0.4f}  "
      f"(sigma_v factor {SIGMA_V_FACTOR})")


def build_model() -> Model:
    """The same architecture and the same initial weights every time."""

    return Model(
        LstmNetwork(
            look_back_len=LOOK_BACK_LEN,
            num_features=2,  # look-back of past LSTM outputs + the one covariate
            infer_len=INFER_LEN,
            num_layer=1,
            num_hidden_unit=NUM_HIDDEN_UNIT,
            device="cpu",
            manual_seed=MANUAL_SEED,
        ),
        WhiteNoise(std_error=sigma_v),
    )


# ------------------------------------------------------------
#  Pretraining, shared by every scenario
# ------------------------------------------------------------
print(f"\nPretraining for up to {NUM_EPOCH} epochs on the training split")
pretrained_state_dict, optimal_epoch, best_validation_log_lik = pretrain_lstm(
    build_model(),
    train_data=train_data,
    validation_data=validation_data,
    num_epoch=NUM_EPOCH,
    noise_floor=noise_floor,
)
print(f"  optimal epoch         : {optimal_epoch}")
print(f"  validation log-lik    :{best_validation_log_lik: 0.3f}"
      "   (epochs are selected on the log-likelihood, not the MSE)")


def prepared_model() -> Model:
    """A pretrained model whose memory has been carried up to `WARMUP_END`."""

    model = build_model()
    model.lstm_net.load_state_dict(pretrained_state_dict)
    model.warm_up_seed = warm_up_filter(model, all_data, WARMUP_END)
    return model


# ------------------------------------------------------------
#  Scenario 1: frozen weights, one straight filter pass
# ------------------------------------------------------------
# The predictions are made before each observation is used, so every step of the pass is
# a one-step-ahead prediction; only the ones from `TEST_START` on are scored.
frozen_model = prepared_model()
frozen_model.lstm_net.num_samples = TEST_END - WARMUP_END
frozen_model.lstm_net.eval()
mu_frozen, std_frozen, frozen_states = frozen_model.filter(
    make_window(all_data, WARMUP_END, TEST_END), train_lstm=False
)
frozen_run = {
    "name": "frozen weights",
    "mu": np.asarray(mu_frozen).flatten()[-NUM_TEST:],
    "std": np.asarray(std_frozen).flatten()[-NUM_TEST:],
}
for key in ("prior", "posterior"):
    frozen_run[f"{key}_mu"] = frozen_states.get_mean("lstm", key)[-NUM_TEST:]
    frozen_run[f"{key}_std"] = frozen_states.get_std("lstm", key)[-NUM_TEST:]

# ------------------------------------------------------------
#  Scenarios 2 and 3: the window scheme, with and without updating
# ------------------------------------------------------------
online_model = prepared_model()
online_run = {"name": "online weights", **run_online_windows(
    online_model,
    data=all_data,
    start=WARMUP_END,
    num_windows=NUM_TEST,
    smooth_len=D,
    look_back_seed=online_model.warm_up_seed,
    train_lstm=True,
)}

control_model = prepared_model()
control_run = {"name": "window, no update", **run_online_windows(
    control_model,
    data=all_data,
    start=WARMUP_END,
    num_windows=NUM_TEST,
    smooth_len=D,
    look_back_seed=control_model.warm_up_seed,
    train_lstm=False,
)}

# ------------------------------------------------------------
#  Compare, on the test set only
# ------------------------------------------------------------
runs = [frozen_run, online_run, control_run]
for run in runs:
    run["pred_indices"] = np.arange(NUM_TEST)
    # The scored quantity is the observation prediction, not the state estimate.
    run["mu_preds"] = run["mu"]
    run["std_preds"] = run["std"]

half = NUM_TEST // 2
print(f"\nTest set only, {NUM_TEST} one-step-ahead predictions per scenario")
print(f"  {'scenario':<18}{'MSE':>9}{'x floor':>9}{'log-lik':>10}"
      f"{'MSE 1st half':>14}{'MSE 2nd half':>14}")
for run in runs:
    mu = run["mu_preds"]
    mse = metric.mse(mu, test_obs)
    log_lik = metric.log_likelihood(mu, test_obs, run["std_preds"])
    print(f"  {run['name']:<18}{mse:>9.4f}{mse / noise_floor:>9.2f}{log_lik:>10.2f}"
          f"{metric.mse(mu[:half], test_obs[:half]):>14.4f}"
          f"{metric.mse(mu[half:], test_obs[half:]):>14.4f}")

print()
print_calibration(
    runs=runs, observations=test_obs, noise_var=sigma_v**2, truth=clean_test
)

# ------------------------------------------------------------
#  Figures
# ------------------------------------------------------------
fig, axes = plt.subplots(
    len(runs), 1, figsize=(DOUBLE_COL[0], 1.9 * len(runs) + 0.6),
    sharex=True, sharey=True,
)
steps = np.arange(NUM_TEST)
for ax, run, color in zip(axes, runs, ("tab:blue", "tab:orange", "tab:green")):
    ax.plot(steps, test_obs, color="0.45", linewidth=0.7, alpha=0.85,
            label="test observation")
    ax.fill_between(steps, run["mu_preds"] - run["std_preds"],
                    run["mu_preds"] + run["std_preds"], color=color, alpha=0.25,
                    linewidth=0, label=r"$\pm 1\sigma$")
    ax.plot(steps, run["mu_preds"], color=color, linewidth=1.0,
            label="one-step-ahead prediction")
    ax.set_title(run["name"], fontsize=9, loc="left")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.legend(loc="lower left", ncol=3, frameon=False, fontsize=7.5)
axes[-1].set_xlabel("Test time step")
fig.tight_layout()
save_figure(fig, "toy_frozen_vs_online_predictions")

plot_error_comparison(
    runs=runs,
    observations=test_obs,
    stem="toy_frozen_vs_online_error",
    window=ROLLING_WINDOW,
    noise_floor=noise_floor,
)

# The prior and the posterior of the `lstm` state, i.e. the same estimate either side of
# the update at each step, plus what that update actually moved.
plot_prior_vs_posterior(
    runs=runs,
    observations=test_obs,
    stem="toy_frozen_vs_online_posterior",
    noise_var=sigma_v**2,
)
