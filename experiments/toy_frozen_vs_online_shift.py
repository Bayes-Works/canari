"""
Frozen weights vs online weights when the series changes *inside the test set*.

The companion of `toy_frozen_vs_online.py`, with one difference that is the whole point:
there, nothing about the series ever changed, the pretrained weights stayed valid, and
freezing them won. Here the training data is stationary, so pretraining sees one regime
and one regime only, and then the test set moves under it:

    test step   0     the regime pretraining learned; both scenarios should tie
    test step  72     the amplitude doubles
    test step 168     the period doubles, 24 -> 48

Nothing about those changes is visible before the test set, so no amount of offline
training can prepare for them. That is the situation online learning exists for, and the
comparison is otherwise identical to the stationary one:

    frozen   weights fixed after pretraining; the test set is filtered with
             `train_lstm=False`. The hidden states still update, the network does not.

    online   the window scheme of `toy_online_lstm.py` over the test set with
             `train_lstm=True`, so the weights keep learning from test observations.

The split is 60 / 40 (50 % train, 10 % validation, 40 % test), both scenarios start
from the same pretrained weights and the same frozen warm-up over `[0, test_start - D)`,
and the online windows start `D` steps before the test set so the first prediction lands
on the first test observation.

Every figure is drawn over the whole series on absolute time steps, so the regime the
pretraining saw and the two regimes it never did sit on one axis. Before the test set the
two scenarios have nothing to disagree about - the weights are frozen for both - so that
stretch is drawn once, in grey, as the context they share.

Run from the repository root:

    python -m experiments.toy_frozen_vs_online_shift        # data seed 0
    python -m experiments.toy_frozen_vs_online_shift 3      # another draw of the noise

Shared helpers live in `utils.py`. See `online_lstm.md` for the scheme itself.
"""

import sys

import numpy as np
import pandas as pd
import pytagi.metric as metric

# utils sets MPLCONFIGDIR and owns the matplotlib configuration, so import it before
# anything pulls matplotlib in.
from experiments.utils import (
    CONTEXT_COLOR,
    DOUBLE_COL,
    Splits,
    generate_periodic_signal,
    make_window,
    mark_changepoints,
    mark_splits,
    plot_error_comparison,
    plot_prior_vs_posterior,
    pretrain_lstm,
    print_calibration,
    print_regime_metrics,
    regime_metrics,
    run_online_windows,
    save_figure,
    set_output_subdir,
    warm_up_filter,
)

import matplotlib.pyplot as plt  # noqa: E402

from canari import DataProcess, Model  # noqa: E402
from canari.component import LstmNetwork, WhiteNoise  # noqa: E402

# Everything this experiment writes goes under
# `experiments/out/toy_frozen_vs_online_shift/`.
set_output_subdir("toy_frozen_vs_online_shift")

# ------------------------------------------------------------
#  Data: stationary until the test set, then two changes
# ------------------------------------------------------------
NUM_TIME_STEPS = 24 * 30
NOISE_STD = 0.2
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# The observation noise the model is *told* about, as a fraction of the true noise. At
# 1.0 the model is given the truth. Below 1.0 it is told the data is cleaner than it is,
# which raises the Kalman gain `var_prior / (var_prior + sigma_v^2)` and so lets the
# update pull the state further onto each observation. That is a deliberate
# mis-specification: it buys reactivity in the state and pays for it in calibration,
# because the predictive variance is then too small for the error actually being made.
SIGMA_V_FACTOR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

TRAIN_SPLIT = 0.5
VALIDATION_SPLIT = 0.1

# The test set starts at 432 with these splits. Both changes land inside it, and the
# first 72 test steps are still the regime pretraining saw, so the comparison has a
# stretch where the two scenarios should agree before they are asked to diverge.
REGIMES = [
    # (start, amplitude, period in time steps)
    (0, 1.0, 24),
    (504, 2.0, 24),
    (600, 2.0, 48),
]
# The same changes as offsets into the test set, for the per-segment metrics and figures.
TEST_CHANGEPOINTS = [(72, "amplitude"), (168, "period")]

# ------------------------------------------------------------
#  Scheme and model parameters
# ------------------------------------------------------------
output_col = [0]
D = 48  # length of the smoothing window of the online scenarios
LOOK_BACK_LEN = 24
NUM_HIDDEN_UNIT = 80
INFER_LEN = 24  # one period of the regime that is trained on
NUM_EPOCH = 100
MANUAL_SEED = 1
TRANSIENT_LEN = 24  # predictions counted as the transient right after a change

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
# Every figure is drawn over the whole series, so the split boundaries travel with it.
splits = Splits.from_processor(data_processor)

TEST_START = data_processor.test_start
TEST_END = data_processor.test_end
NUM_TEST = TEST_END - TEST_START
WARMUP_END = TEST_START - D

if WARMUP_END <= LOOK_BACK_LEN:
    raise ValueError("Not enough data before the test set for the warm-up.")
for change_start, _, _ in REGIMES[1:]:
    if change_start <= TEST_START:
        raise ValueError(
            f"regime change at {change_start} is not inside the test set "
            f"[{TEST_START}, {TEST_END}); pretraining would have seen it"
        )

# The same changes on the absolute axis the figures use.
CHANGEPOINTS = [(TEST_START + offset, label) for offset, label in TEST_CHANGEPOINTS]

all_obs = all_data["y"].flatten()
test_obs = test_data["y"].flatten()
# Absolute time steps of the compared stretch, and the same steps counted from the start of
# the test set. The figures use the first, the per-segment tables the second.
test_steps = np.arange(TEST_START, TEST_END)
test_offsets = np.arange(NUM_TEST)
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
for (change_start, amplitude, period), (offset, label) in zip(
    REGIMES[1:], TEST_CHANGEPOINTS
):
    print(f"    change at {change_start} (test step {offset:>3}): {label:<10} "
          f"-> amplitude {amplitude}, period {period}")
print(f"  online warm-up ends at: {WARMUP_END}  (first prediction at {TEST_START})")
print(f"True noise variance     :{noise_floor: 0.4f}  (every MSE is reported against it)")
print(f"Assumed by the model    :{sigma_v**2: 0.4f}  "
      f"(sigma_v factor {SIGMA_V_FACTOR})")


def build_model() -> Model:
    """The same architecture and the same initial weights every time."""

    return Model(
        LstmNetwork(
            look_back_len=LOOK_BACK_LEN,
            num_features=2,
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
    model.warm_up = warm_up_filter(model, all_data, WARMUP_END)
    return model


# ------------------------------------------------------------
#  Scenario 1: frozen weights
# ------------------------------------------------------------
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
for key, states_type in (("prior", "prior"), ("posterior", "posterior")):
    frozen_run[f"{key}_mu"] = frozen_states.get_mean("lstm", states_type)[-NUM_TEST:]
    frozen_run[f"{key}_std"] = frozen_states.get_std("lstm", states_type)[-NUM_TEST:]

# Everything before the test set is context the two scenarios have no reason to disagree
# about: the warm-up over `[0, WARMUP_END)` is identical by construction, and over the D
# steps between it and the test set the weights are still frozen. Taking that stretch from
# the frozen pass gives the figures one grey trace to put in front of the comparison, so a
# row covers the series from its first step instead of starting mid-way through.
pre_test = {
    "name": "pre-test (weights frozen)",
    "indices": np.arange(TEST_START),
}
for key, filtered in (
    ("mu", np.asarray(mu_frozen).flatten()),
    ("std", np.asarray(std_frozen).flatten()),
    ("prior_mu", frozen_states.get_mean("lstm", "prior")),
    ("prior_std", frozen_states.get_std("lstm", "prior")),
    ("posterior_mu", frozen_states.get_mean("lstm", "posterior")),
    ("posterior_std", frozen_states.get_std("lstm", "posterior")),
):
    pre_test[key] = np.concatenate([frozen_model.warm_up[key], filtered[:D]])

# ------------------------------------------------------------
#  Scenario 2: the window scheme, updating the weights as the test set arrives
# ------------------------------------------------------------
online_model = prepared_model()
online_run = {"name": "online weights", **run_online_windows(
    online_model,
    data=all_data,
    start=WARMUP_END,
    num_windows=NUM_TEST,
    smooth_len=D,
    look_back_seed=online_model.warm_up["look_back_seed"],
    train_lstm=True,
)}

# ------------------------------------------------------------
#  Compare, on the test set only
# ------------------------------------------------------------
runs = [frozen_run, online_run]
for run in runs:
    # Absolute time steps, so the figures put the compared stretch where it belongs in the
    # series. The per-segment tables below stay on test offsets, which is what
    # `TEST_CHANGEPOINTS` and `test_obs` are indexed by.
    run["pred_indices"] = test_steps
    # `plot_error_comparison` and the metrics below score the observation prediction.
    run["mu_preds"] = run["mu"]
    run["std_preds"] = run["std"]

print(f"\nTest set only, {NUM_TEST} one-step-ahead predictions per scenario")
print(f"  {'scenario':<18}{'MSE':>9}{'x floor':>9}{'log-lik':>10}")
for run in runs:
    mse = metric.mse(run["mu_preds"], test_obs)
    log_lik = metric.log_likelihood(run["mu_preds"], test_obs, run["std_preds"])
    print(f"  {run['name']:<18}{mse:>9.4f}{mse / noise_floor:>9.2f}{log_lik:>10.2f}")

# Per segment of the test set: the stretch before any change, then one per change. The
# transient / settled split says whether a scenario is adapting or simply wrong.
for run in runs:
    print(f"\n{run['name']}")
    print_regime_metrics(
        regime_metrics(
            pred_indices=test_offsets,
            mu_preds=run["mu_preds"],
            observations=test_obs,
            changepoints=TEST_CHANGEPOINTS,
            num_time_steps=NUM_TEST,
            transient_len=TRANSIENT_LEN,
        ),
        noise_floor=noise_floor,
        transient_len=TRANSIENT_LEN,
    )

print()
print_calibration(
    runs=runs,
    observations=test_obs,
    noise_var=sigma_v**2,
    changepoints=TEST_CHANGEPOINTS,
    truth=clean_test,
)

# ------------------------------------------------------------
#  Figures
# ------------------------------------------------------------
fig, axes = plt.subplots(
    len(runs), 1, figsize=(DOUBLE_COL[0], 1.9 * len(runs) + 0.6),
    sharex=True, sharey=True,
)
time_index = np.arange(len(all_obs))
for index, (ax, run) in enumerate(zip(axes, runs)):
    ax.plot(time_index, all_obs, color="tab:red", linewidth=0.7, alpha=0.85,
            label="observation")
    ax.plot(pre_test["indices"], pre_test["mu"], color=CONTEXT_COLOR, linewidth=0.8,
            linestyle=(0, (4, 2)), label=pre_test["name"])
    ax.fill_between(test_steps, run["mu_preds"] - run["std_preds"],
                    run["mu_preds"] + run["std_preds"], color="tab:blue", alpha=0.3,
                    linewidth=0, label=r"$\pm 1\sigma$")
    ax.plot(test_steps, run["mu_preds"], color="tab:blue", linewidth=1.0,
            label="one-step-ahead prediction")
    ax.set_ylabel(run["name"])
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_xlim(0, len(all_obs) - 1)
    mark_splits(ax, splits, with_labels=index == 0)
    mark_changepoints(ax, CHANGEPOINTS, with_labels=index == 0)
# One legend for the figure: the rows differ only in which scenario they draw, and at full
# series width a per-row legend sits on top of the data.
axes[0].legend(
    loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=7.5,
)
axes[-1].set_xlabel("Time step")
fig.tight_layout()
save_figure(fig, "toy_frozen_vs_online_shift_predictions")

plot_error_comparison(
    runs=runs,
    observations=all_obs,
    stem="toy_frozen_vs_online_shift_error",
    splits=splits,
    changepoints=CHANGEPOINTS,
    noise_floor=noise_floor,
)

# The prior and the posterior of the `lstm` state, i.e. the same estimate either side of
# the update at each step. Once the series moves away from what a model knows, its prior
# drifts off while its posterior is still dragged back onto the data every step.
plot_prior_vs_posterior(
    runs=runs,
    observations=all_obs,
    run_indices=test_steps,
    stem="toy_frozen_vs_online_shift_posterior",
    context=pre_test,
    noise_var=sigma_v**2,
    splits=splits,
    changepoints=CHANGEPOINTS,
)
plot_prior_vs_posterior(
    runs=runs,
    observations=all_obs,
    run_indices=test_steps,
    stem="toy_frozen_vs_online_shift_posterior_zoom",
    context=pre_test,
    noise_var=sigma_v**2,
    splits=splits,
    changepoints=CHANGEPOINTS,
    zoom=(TEST_START + 48, TEST_START + 144),  # across the amplitude change
)
