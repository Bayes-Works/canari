"""
Shared helpers for the online LSTM experiments.

Used by `toy_online_lstm.py` and `toy_online_lstm_trend.py`, which run the same online
scheme:

    filter D steps -> filter one additional t+1 step -> smooth the D window
    -> restart from the second step of that window -> repeat

This module groups what the two experiments have in common: the synthetic signal, the
online-window helpers (`make_window`, `smooth_window`, `LstmLookBackBuffer`,
`rewind_to_step`), the hidden-state records, the LSTM parameter-change diagnostics, and
the figures. The online loop itself stays in each experiment, since that is what an
experiment varies.

Any `look_back_len` is supported: `LstmLookBackBuffer` carries the smoothed LSTM outputs
across windows, which is what a look-back longer than the window stride needs.

Import this module before matplotlib: it sets `MPLCONFIGDIR` at import time, and it is
also the only place matplotlib is imported, so the experiments do not need to.
"""

import copy
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "experiments" / "out"
MPLCONFIG_DIR = OUT_DIR / "mplconfig"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytagi.metric as metric  # noqa: E402
from canari import Model  # noqa: E402

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


# ------------------------------------------------------------
#  Synthetic data
# ------------------------------------------------------------
def generate_periodic_signal(
    num_time_steps: int,
    regimes: Sequence[Tuple[int, float, float]],
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    A sine whose amplitude and period change at given time steps.

    The phase is adjusted at every regime change so the oscillation stays continuous
    across it: only the amplitude and the period jump, not the phase.

    Args:
        num_time_steps (int): Length of the series.
        regimes (Sequence[Tuple[int, float, float]]): `(start, amplitude, period)` per
            regime, in increasing `start`, the first one starting at 0. `period` is in
            time steps.
        noise_std (float): Standard deviation of the additive Gaussian observation noise.
        seed (Optional[int]): Seed for that noise, so a run is reproducible.

    Returns:
        np.ndarray: the series, of length `num_time_steps`.

    Examples:
        >>> y = generate_periodic_signal(720, [(0, 1.0, 24), (360, 2.0, 48)], 0.2, seed=0)
    """

    if regimes[0][0] != 0:
        raise ValueError("the first regime must start at time step 0")

    t = np.arange(num_time_steps, dtype=float)
    y = np.zeros(num_time_steps, dtype=float)
    ends = [regime[0] for regime in regimes[1:]] + [num_time_steps]

    phase = 0.0
    frequency = 1.0 / regimes[0][2]
    for (start, amplitude, period), end in zip(regimes, ends):
        new_frequency = 1.0 / period
        if start > 0:
            # keep sin(2 pi f t + phase) continuous at t = start
            phase = phase + 2 * np.pi * start * (frequency - new_frequency)
        frequency = new_frequency
        segment = slice(start, end)
        y[segment] = amplitude * np.sin(2 * np.pi * frequency * t[segment] + phase)

    if noise_std > 0.0:
        y = y + np.random.default_rng(seed).normal(0.0, noise_std, num_time_steps)
    return y


def add_piecewise_trend(
    y: np.ndarray, slopes: Sequence[Tuple[int, float]]
) -> np.ndarray:
    """
    Add a continuous piecewise-linear trend: the slope changes at the given time steps,
    the level does not jump.

    Args:
        y (np.ndarray): Series to add the trend to.
        slopes (Sequence[Tuple[int, float]]): `(start, slope per time step)`, in
            increasing `start`, the first one starting at 0.

    Returns:
        np.ndarray: a new series.
    """

    if slopes[0][0] != 0:
        raise ValueError("the first slope must start at time step 0")

    per_step = np.zeros(len(y), dtype=float)
    for start, slope in slopes:
        per_step[start:] = slope
    trend = np.cumsum(per_step) - per_step[0]
    return np.asarray(y, dtype=float) + trend


def add_level_shifts(y: np.ndarray, shifts: Sequence[Tuple[int, float]]) -> np.ndarray:
    """
    Add step changes: everything from `start` onwards moves by `size`.

    Args:
        y (np.ndarray): Series to shift.
        shifts (Sequence[Tuple[int, float]]): `(start, size)` per step change.

    Returns:
        np.ndarray: a new series.
    """

    shifted = np.asarray(y, dtype=float).copy()
    for start, size in shifts:
        shifted[start:] += size
    return shifted


# ------------------------------------------------------------
#  Online window helpers
# ------------------------------------------------------------
def make_window(
    data: Dict[str, np.ndarray], start: int, end_exclusive: int
) -> Dict[str, np.ndarray]:
    """Slice `[start, end_exclusive)` out of a data split."""

    return {
        "x": data["x"][start:end_exclusive],
        "y": data["y"][start:end_exclusive],
    }


def smooth_window(model: Model) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth the window that has just been filtered.

    Two smoothers are applied, exactly like :meth:`canari.model.Model.smoother` does,
    but called here step by step:

    - the RTS smoother over the SSM hidden states stored in `model.states`,
    - the LSTM smoother over the `num_samples` buffer of the SLSTM layers.

    The LSTM smoother returns one entry per filtered step, i.e. `D + 1` entries. The
    last one is the filtered estimate (nothing comes after it), so the smoothed window
    is made of the first `D` entries.

    `Model.smoother()` is not used because it calls `lstm_net.get_lstm_states_smooth`,
    which does not exist in the installed pyTAGI.

    Returns:
        Tuple[np.ndarray, np.ndarray]: smoothed LSTM output means and variances.
    """

    num_time_steps = len(model.states.mu_smooth)
    for time_step in reversed(range(0, num_time_steps - 1)):
        model.rts_smoother(time_step)

    mu_smooth, var_smooth = model.lstm_net.smoother()
    return (
        np.asarray(mu_smooth, dtype=np.float32).flatten(),
        np.asarray(var_smooth, dtype=np.float32).flatten(),
    )


def drain_lstm_smoother_buffer(model: Model) -> None:
    """
    Run the LSTM smoother and discard the result, to release the SLSTM sample buffer
    before another pass of `num_samples` filter steps.

    Only a `smoother()` call releases that buffer. Without one, a repeated full-length
    `filter` pass over the same network aborts inside the C++ backend rather than raising:
    with `num_samples = 312`, passes 1 and 2 succeed and the third one kills the process
    (verified). The online loop never meets this because it smooths every window; a
    pretraining loop over a pool of series does.

    It also zeroes the current cell and hidden states, so pair it with
    `reset_lstm_memory` when the next pass is an unrelated series. Parameters are
    untouched: the smoother is a pass over hidden states, not an update.
    """

    model.lstm_net.smoother()


def reset_lstm_memory(model: Model) -> None:
    """
    Put the LSTM memory back to the state a fresh run starts from: a look-back of
    `(mu 0, var 1)` and zeroed cell and hidden states.

    Needed whenever consecutive `filter` calls are on unrelated series, e.g. when
    pretraining one network over a pool of them, so that nothing leaks from one into the
    next. It leaves the network's parameters untouched, which is the point.

    This resets the *memory*, not the sample buffer; see `drain_lstm_smoother_buffer`.
    """

    model.lstm_output_history.initialize(model.lstm_net.lstm_look_back_len)
    lstm_states = model.lstm_net.get_lstm_states()
    zeroed = {
        layer: tuple(np.zeros_like(np.array(value)).tolist() for value in values)
        for layer, values in lstm_states.items()
    }
    model.lstm_net.set_lstm_states(zeroed)


class LstmLookBackBuffer:
    """
    Smoothed LSTM outputs indexed by absolute time step.

    Restarting a filter at time `t` needs the `look_back_len` LSTM outputs that end just
    before `t`, because that is what `common.prepare_lstm_input` feeds the network. With
    `look_back_len = 1` the single value needed is the first entry of the window that was
    just smoothed, but for a longer look-back the earlier values belong to *previous*
    windows, which `Model.filter` and `lstm_net.smoother()` have already overwritten. They
    are therefore collected here as the windows advance.

    Each time step is stored once, from the window in which it was the first step, i.e.
    with the full `D`-step smoothing lag.

    The buffer is padded in front by `look_back_len - 1` slots holding the `(mu 0, var 1)`
    that `LstmOutputHistory.initialize` starts the model from, so the first windows can
    look back before the start of the series exactly as the very first filter step did.

    Examples:
        >>> buffer = LstmLookBackBuffer(look_back_len=3, num_time_steps=576)
        >>> buffer.store(0, mu_smooth_lstm, var_smooth_lstm)   # time step 0
        >>> mu, var = buffer.look_back(1)                      # restarting at time 1
        >>> mu                                                 # pad, pad, smoothed t=0
        array([0., 0., 0.34], dtype=float32)
    """

    def __init__(self, look_back_len: int, num_time_steps: int):
        self.look_back_len = look_back_len
        self.pad = look_back_len - 1
        self.mu = np.zeros(num_time_steps + self.pad, dtype=np.float32)
        self.var = np.ones(num_time_steps + self.pad, dtype=np.float32)

    def store(
        self,
        time_step: int,
        mu_smooth_lstm: np.ndarray,
        var_smooth_lstm: np.ndarray,
        window_step: int = 0,
    ) -> None:
        """
        Keep the smoothed LSTM output of `time_step`, taken from `window_step` of the
        window that was just smoothed. The default reads the window's first step, the one
        smoothed with the full lag.
        """

        self.mu[time_step + self.pad] = mu_smooth_lstm[window_step]
        self.var[time_step + self.pad] = var_smooth_lstm[window_step]

    def look_back(self, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        The `look_back_len` entries ending just before `time_step`.

        Bounds are checked here because `LstmOutputHistory.set` does not validate the
        length of what it is given: a short slice would silently shrink the LSTM input and
        only surface later as a pyTAGI input-size mismatch.
        """

        stop = time_step + self.pad
        start = stop - self.look_back_len
        if start < 0 or stop > len(self.mu):
            raise ValueError(
                f"look-back [{start}:{stop}] for time step {time_step} falls outside the "
                f"buffer of length {len(self.mu)}"
            )
        return self.mu[start:stop], self.var[start:stop]


def rewind_to_step(
    model: Model,
    look_back_buffer: LstmLookBackBuffer,
    window_start: int,
    time_step: int,
) -> None:
    """
    Move the model memory back to `time_step` of the window that was just smoothed, so
    that the next window can be filtered from there. `window_start` is the absolute time
    step of that window's first step, so the filter restarts at `window_start + time_step`.

    Three pieces of memory move:

    - the SSM hidden states, level and trend included, restored by `set_memory` from the
      smoothed estimates at `time_step - 1`;
    - the LSTM look-back, taken from `look_back_buffer`. `set_memory` also writes it, from
      the smoothed SSM 'lstm' state, but its slice `[time_step - look_back_len:time_step]`
      is empty whenever `time_step < look_back_len`, so that write is both redundant and
      wrong here and is always overwritten;
    - the LSTM cell/hidden states, which `set_memory` documents as the caller's job
      whenever `time_step != 0`, and which `lstm_net.smoother()` has just zeroed. After
      `smoother()`, `get_lstm_states(j)` is the smoothed state *after* buffer sample `j`,
      so restarting at `time_step` uses `time_step - 1`.
    """

    model.set_memory(states=model.states, time_step=time_step)
    mu_look_back, var_look_back = look_back_buffer.look_back(window_start + time_step)
    model.lstm_output_history.set(mu=mu_look_back, var=var_look_back)
    model.lstm_net.set_lstm_states(model.lstm_net.get_lstm_states(time_step - 1))


# ------------------------------------------------------------
#  Offline pretraining, and running the online scheme over a later stretch
# ------------------------------------------------------------
def pretrain_lstm(
    model: Model,
    train_data: Dict[str, np.ndarray],
    validation_data: Dict[str, np.ndarray],
    num_epoch: int,
    noise_floor: Optional[float] = None,
    log_every: int = 10,
) -> Tuple[dict, int, float]:
    """
    Train an LSTM offline, one epoch loop with early stopping on the validation split.

    `Model.lstm_train` is not used: it ends in `Model.smoother()`, which calls
    `lstm_net.get_lstm_states_smooth`, a method the installed pyTAGI does not have. This
    is the same recipe written with the calls that do exist. Offline training needs no
    smoothing anyway - the SLSTM sample buffer only has to be released between epochs,
    which is what `drain_lstm_smoother_buffer` does.

    The model is left holding the parameters of its last epoch, not the best ones; use
    the returned state dict for that.

    Epochs are selected on the validation **log-likelihood**, not on the validation MSE.
    The two disagree whenever the predictive variance is wrong: MSE only sees the mean,
    so it will happily keep an epoch whose intervals have collapsed, while the
    log-likelihood scores the mean and the variance together.

    Args:
        model (Model): the model to train, modified in place.
        train_data (Dict[str, np.ndarray]): the split filtered each epoch.
        validation_data (Dict[str, np.ndarray]): the split forecast each epoch, which is
            what the epoch is scored on.
        num_epoch (int): maximum number of epochs.
        noise_floor (Optional[float]): if given, the MSE is also printed as a multiple
            of it.
        log_every (int): print every n epochs; 0 silences the loop.

    Returns:
        Tuple[dict, int, float]: the state dict of the best epoch, that epoch's index,
        and its validation log-likelihood.
    """

    num_steps = len(train_data["y"]) + len(validation_data["y"])
    model.lstm_net.num_samples = model.lstm_net.lstm_infer_len + num_steps
    validation_obs = validation_data["y"].flatten()
    best_state_dict = None

    for epoch in range(num_epoch):
        model.white_noise_decay(
            epoch, white_noise_max_std=5, white_noise_decay_factor=0.9
        )

        # Every epoch is a fresh pass over the same series, so nothing may carry over.
        reset_lstm_memory(model)
        model.lstm_net.train()
        model.filter(train_data, train_lstm=True)

        # `forecast` appends to the states history, so clear it first.
        model.lstm_net.eval()
        model.initialize_states_history()
        mu_validation, std_validation, _ = model.forecast(validation_data)
        mu_validation = np.asarray(mu_validation).flatten()
        std_validation = np.asarray(std_validation).flatten()
        validation_mse = float(np.mean((mu_validation - validation_obs) ** 2))
        validation_log_lik = float(
            metric.log_likelihood(mu_validation, validation_obs, std_validation)
        )

        # `mode="max"`: the log-likelihood is better when larger.
        model.early_stopping(
            evaluate_metric=validation_log_lik,
            current_epoch=epoch,
            max_epoch=num_epoch,
            mode="max",
        )
        if epoch == model.optimal_epoch:
            best_state_dict = copy.deepcopy(model.lstm_net.state_dict())
        if log_every and epoch % log_every == 0:
            floor = (
                f" ({validation_mse / noise_floor: 0.2f} x floor)" if noise_floor else ""
            )
            print(f"  epoch {epoch:>3}  validation log-lik {validation_log_lik: 0.3f}"
                  f"   MSE {validation_mse: 0.4f}{floor}")

        # Release the sample buffer before the next full-length pass.
        drain_lstm_smoother_buffer(model)
        if model.stop_training:
            break

    return best_state_dict, model.optimal_epoch, model.early_stop_metric


def warm_up_filter(
    model: Model, data: Dict[str, np.ndarray], end: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter `data[0:end]` with the parameters frozen, leaving the model memory - hidden
    states, LSTM look-back and LSTM cell/hidden states - sitting exactly at step `end`.

    Used to carry a pretrained model up to the point where a comparison starts, so that
    every scenario begins from the same memory and only differs in what it does next.

    Returns:
        Tuple[np.ndarray, np.ndarray]: mean and variance of the posterior `lstm` state
        over the warm-up, which is what a later online run needs to seed the look-back
        entries that reach back before its first window.
    """

    model.lstm_net.num_samples = end
    model.lstm_net.eval()
    _, _, states = model.filter(make_window(data, 0, end), train_lstm=False)

    # Release the buffer before the next pass. `smoother()` zeroes the cell and hidden
    # states, and the last entry it leaves behind is the filtered one, so putting that
    # back restores the memory the warm-up ended with.
    model.lstm_net.smoother()
    model.lstm_net.set_lstm_states(model.lstm_net.get_lstm_states(end - 1))
    return (
        states.get_mean("lstm", "posterior"),
        states.get_std("lstm", "posterior") ** 2,
    )


def run_online_windows(
    model: Model,
    data: Dict[str, np.ndarray],
    start: int,
    num_windows: int,
    smooth_len: int,
    look_back_seed: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    train_lstm: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk `num_windows` steps of the online scheme over `data`, starting at step `start`.

    Window `k` covers `data[start + k : start + k + smooth_len + 1]`, so its extra t+1
    step is `start + smooth_len + k`: the first window begins `smooth_len` steps before
    the first step being predicted, and every one of those steps gets a one-step-ahead
    prediction. The model memory must already be at `start`, e.g. via `warm_up_filter`.

    Args:
        model (Model): a model whose memory sits at `start`, modified in place.
        data (Dict[str, np.ndarray]): the series the windows are cut from.
        start (int): index of the first window's first step.
        num_windows (int): one window per predicted step.
        smooth_len (int): `D`, the number of steps smoothed per window.
        look_back_seed (Optional[Tuple[np.ndarray, np.ndarray]]): mean and variance of
            the LSTM output over the steps *before* `start`, from which the look-back
            entries reaching back before the first window are taken. Without it those
            slots hold the `(mu 0, var 1)` a fresh run starts from, which is right at the
            beginning of a series and wrong in the middle of one.
        train_lstm (bool): whether the window filter updates the network parameters.

    Returns:
        Dict[str, np.ndarray]: one value per window, under the keys

        - `mu` / `std`: the one-step-ahead prediction of the *observation*, which is what
          a run is scored on. Its variance includes the observation noise.
        - `prior_mu` / `prior_std`: the prior of the `lstm` state at the same step, i.e.
          the same prediction of the underlying signal, before `y` is used.
        - `posterior_mu` / `posterior_std`: the posterior of that state, after `y` is
          used. This is the estimate the model carries forward, and after a change in the
          series it can be far from the prior that preceded it.
    """

    model.lstm_net.num_samples = smooth_len + 1
    look_back_buffer = LstmLookBackBuffer(
        look_back_len=model.lstm_net.lstm_look_back_len,
        num_time_steps=num_windows + smooth_len,
    )
    pad = look_back_buffer.pad
    if pad and look_back_seed is not None:
        mu_seed, var_seed = look_back_seed
        look_back_buffer.mu[:pad] = mu_seed[-pad:]
        look_back_buffer.var[:pad] = var_seed[-pad:]

    records = {key: [] for key in
               ("mu", "std", "prior_mu", "prior_std", "posterior_mu", "posterior_std")}
    for window_index in range(num_windows):
        window_start = window_index  # relative to `start`
        window_end = window_start + smooth_len + 1

        if train_lstm:
            model.lstm_net.train()
        else:
            model.lstm_net.eval()
        mu_filt, std_filt, states = model.filter(
            make_window(data, start + window_start, start + window_end),
            train_lstm=train_lstm,
        )

        # The extra t+1 step: predicted before its own observation is used, then updated
        # with it. Both sides of that step are kept.
        records["mu"].append(mu_filt[-1])
        records["std"].append(std_filt[-1])
        records["prior_mu"].append(states.get_mean("lstm", "prior")[-1])
        records["prior_std"].append(states.get_std("lstm", "prior")[-1])
        records["posterior_mu"].append(states.get_mean("lstm", "posterior")[-1])
        records["posterior_std"].append(states.get_std("lstm", "posterior")[-1])

        mu_smooth_lstm, var_smooth_lstm = smooth_window(model)
        look_back_buffer.store(window_start, mu_smooth_lstm, var_smooth_lstm)

        if window_index < num_windows - 1:
            rewind_to_step(model, look_back_buffer, window_start, 1)

    return {key: np.asarray(value).flatten() for key, value in records.items()}


# ------------------------------------------------------------
#  Hidden-state records
# ------------------------------------------------------------
RECORD_KEYS = (
    "predict_mu",
    "predict_std",
    "filter_mu",
    "filter_std",
    "smooth_mu",
    "smooth_std",
)


def initialize_state_records(
    states_name: List[str], num_time_steps: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    One record per tracked hidden state, holding three estimates over time:

    - 'predict': the t+1 prior, i.e. the value that is coupled with the other
      components to predict the observation at t+1,
    - 'filter': the posterior at t+1, once y(t+1) has been used,
    - 'smooth': the estimate of a time step smoothed with the full D-step lag.
    """

    return {
        name: {
            key: np.full(num_time_steps, np.nan, dtype=np.float32)
            for key in RECORD_KEYS
        }
        for name in states_name
    }


def record_window_states(
    records: Dict[str, Dict[str, np.ndarray]],
    states,
    window_start: int,
    smooth_len: int,
    is_last_window: bool,
) -> None:
    """
    Store the t+1 estimates of the extra filter step and the smoothed estimate of the
    first step of the window, which is the one smoothed with the full lag. The last
    window also provides the tail of the time series.
    """

    predict_index = window_start + smooth_len
    for name, record in records.items():
        record["predict_mu"][predict_index] = states.get_mean(name, "prior")[-1]
        record["predict_std"][predict_index] = states.get_std(name, "prior")[-1]
        record["filter_mu"][predict_index] = states.get_mean(name, "posterior")[-1]
        record["filter_std"][predict_index] = states.get_std(name, "posterior")[-1]

        smooth_mu = states.get_mean(name, "smooth")
        smooth_std = states.get_std(name, "smooth")
        if is_last_window:
            window_slice = slice(window_start, window_start + smooth_len)
            record["smooth_mu"][window_slice] = smooth_mu[:smooth_len]
            record["smooth_std"][window_slice] = smooth_std[:smooth_len]
        else:
            record["smooth_mu"][window_start] = smooth_mu[0]
            record["smooth_std"][window_start] = smooth_std[0]


# ------------------------------------------------------------
#  Parameter-change diagnostics
# ------------------------------------------------------------
def _kl_divergence_gaussian(
    prior_mu: list,
    prior_var: list,
    post_mu: list,
    post_var: list,
) -> list:
    """
    Element‑wise KL divergence D_KL[ q‖p ] between two univariate
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
        (μ₁ − μ₂)² + (σ₁ − σ₂)²
    where σ = √var.

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


def store_parameter_diagnostics(
    prior_state: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    posterior_state: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    kl_history: Dict[str, Dict[str, list]],
    wasserstein_history: Dict[str, Dict[str, list]],
) -> None:
    """
    Append the per-layer mean KL divergence and mean 2-Wasserstein distance between the
    LSTM parameters before and after a window to the two histories.
    """

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


# ------------------------------------------------------------
#  Figures
# ------------------------------------------------------------
def regime_metrics(
    pred_indices: np.ndarray,
    mu_preds: np.ndarray,
    observations: np.ndarray,
    changepoints: Sequence[Tuple[int, str]],
    num_time_steps: int,
    transient_len: int = 24,
) -> List[Dict[str, object]]:
    """
    Online one-step-ahead MSE per regime, split into the `transient_len` predictions right
    after the regime starts and the settled remainder.

    The split is what tells whether the scheme is adapting: a transient much worse than the
    settled part means the error is the cost of catching the change, not a bad fit.

    Args:
        pred_indices (np.ndarray): Time step of each online prediction.
        mu_preds (np.ndarray): The online predictions.
        observations (np.ndarray): The full observation series.
        changepoints (Sequence[Tuple[int, str]]): `(time step, label)` pairs.
        num_time_steps (int): End of the last regime.
        transient_len (int): How many predictions count as the transient.

    Returns:
        List[Dict[str, object]]: one row per regime.
    """

    starts = [0] + [time_step for time_step, _ in changepoints]
    labels = ["start"] + [label for _, label in changepoints]
    ends = starts[1:] + [num_time_steps]

    rows = []
    for start, end, label in zip(starts, ends, labels):
        in_regime = (pred_indices >= start) & (pred_indices < end)
        indices = pred_indices[in_regime]
        if len(indices) == 0:
            continue
        errors = observations[indices] - mu_preds[in_regime]
        transient = errors[:transient_len]
        settled = errors[transient_len:]
        rows.append(
            {
                "label": label,
                "start": start,
                "end": min(end, int(pred_indices[-1]) + 1),
                "count": len(indices),
                "transient": float(np.mean(transient**2)) if len(transient) else np.nan,
                "settled": float(np.mean(settled**2)) if len(settled) else np.nan,
            }
        )
    return rows


def print_regime_metrics(
    rows: Sequence[Dict[str, object]], noise_floor: float, transient_len: int = 24
) -> None:
    """Print `regime_metrics` as multiples of the noise floor."""

    print(
        f"Online MSE per regime, as a multiple of the noise floor "
        f"(transient = first {transient_len} predictions):"
    )
    print(f"  {'change':<12}{'steps':>14}{'transient':>12}{'settled':>10}")
    for row in rows:
        span = f"{row['start']}-{row['end'] - 1}"
        transient = row["transient"] / noise_floor
        settled = row["settled"] / noise_floor
        print(
            f"  {row['label']:<12}{span:>14}"
            f"{transient:>12.2f}{settled:>10.2f}"
            if np.isfinite(settled)
            else f"  {row['label']:<12}{span:>14}{transient:>12.2f}{'--':>10}"
        )


def print_calibration(
    runs: Sequence[Dict[str, object]],
    observations: np.ndarray,
    noise_var: float,
    changepoints: Optional[Sequence[Tuple[int, str]]] = None,
    transient_len: int = 0,
    truth: Optional[np.ndarray] = None,
) -> None:
    """
    Per segment and per model, what the model said its predictive variance was against
    the error it actually made.

    The observation's predictive variance is `var_prior + noise_var`, so `std(z)` only
    ever tests that **sum**. The Kalman gain depends on how the sum is **split**, and the
    split is not identifiable from the observations alone: raise `noise_var` and
    `var_prior` inflates to compensate, leaving `std(z)` where it was. Two columns see
    the split:

    - `sig` needs `truth`, so it only exists for synthetic data: the model's real squared
      error about the noise-free signal divided by the `var_prior` it claims. Above 1 the
      network is overconfident about the signal, below 1 it is conservative.
    - `r1` is the lag-1 autocorrelation of the innovations and needs no ground truth. A
      filter running at the right gain leaves them white. A gain that is too small makes
      the state lag the signal, which shows up as positive `r1`.

    Args:
        runs (Sequence[Dict[str, object]]): one dict per model, with keys `name`, `mu`,
            `std` (the observation prediction) and `prior_mu` / `prior_std` (the state's
            own).
        observations (np.ndarray): what the models were scored against.
        noise_var (float): the observation-noise variance the models were given.
        changepoints (Optional[Sequence[Tuple[int, str]]]): segment boundaries.
        transient_len (int): if non-zero, the first `transient_len` steps of each segment
            are reported separately from the rest.
        truth (Optional[np.ndarray]): the noise-free signal, if it is known.
    """

    starts = [0] + [time_step for time_step, _ in changepoints or ()]
    labels = ["start"] + [label for _, label in changepoints or ()]
    ends = starts[1:] + [len(observations)]

    print(f"Predictive calibration (assumed noise variance {noise_var:.4f}); "
          "z = (y - prediction) / predictive std, calibrated means std(z) = 1")
    print(f"  {'model':<18}{'segment':<14}{'var prior':>10}{'gain':>7}{'pred var':>10}"
          f"{'realized':>10}{'ratio':>8}{'std(z)':>8}{'sig':>7}{'r1':>7}")
    for run in runs:
        for start, end, label in zip(starts, ends, labels):
            spans = [(label, slice(start, end))]
            if transient_len and end - start > transient_len:
                spans = [
                    (f"{label} (early)", slice(start, start + transient_len)),
                    (f"{label} (rest)", slice(start + transient_len, end)),
                ]
            for span_label, span in spans:
                prior_var = np.asarray(run["prior_std"])[span] ** 2
                pred_var = np.asarray(run["std"])[span] ** 2
                error = observations[span] - np.asarray(run["mu"])[span]
                realized = float(np.mean(error**2))
                z_std = float(np.std(error / np.sqrt(pred_var)))
                gain = float(np.mean(prior_var / (prior_var + noise_var)))

                # Does the split hold up, and are the innovations white?
                if truth is not None:
                    signal_error = np.asarray(run["prior_mu"])[span] - truth[span]
                    sig = f"{np.mean(signal_error**2) / prior_var.mean():>7.2f}"
                else:
                    sig = f"{'-':>7}"
                centred = error - error.mean()
                r1 = float(
                    np.sum(centred[1:] * centred[:-1]) / np.sum(centred**2)
                ) if len(centred) > 1 else np.nan

                print(f"  {run['name']:<18}{span_label:<14}{prior_var.mean():>10.4f}"
                      f"{gain:>7.2f}{pred_var.mean():>10.4f}{realized:>10.4f}"
                      f"{realized / pred_var.mean():>8.2f}{z_std:>8.2f}{sig}"
                      f"{r1:>7.2f}")
        print()


CHANGE_COLOR = "0.35"


def mark_changepoints(
    ax: plt.Axes,
    changepoints: Optional[Sequence[Tuple[int, str]]],
    with_labels: bool = False,
) -> None:
    """
    Draw a dashed vertical line at every known change in the signal, so that every figure
    can be read against the same reference.

    Args:
        ax (plt.Axes): Axes to draw on.
        changepoints (Optional[Sequence[Tuple[int, str]]]): `(time step, label)` pairs.
        with_labels (bool): Whether to write the labels. Use it on one axes per figure,
            usually the top one, and leave the others as bare lines.
    """

    for time_step, label in changepoints or ():
        ax.axvline(
            time_step,
            color=CHANGE_COLOR,
            linewidth=0.6,
            linestyle=(0, (3, 2)),
            alpha=0.9,
            zorder=0.5,
        )
        if with_labels:
            ax.annotate(
                label,
                xy=(time_step, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(1.8, -2.0),
                textcoords="offset points",
                rotation=90,
                ha="left",
                va="top",
                fontsize=5.5,
                color=CHANGE_COLOR,
                annotation_clip=False,
                bbox=dict(
                    boxstyle="square,pad=0.08", facecolor="white", edgecolor="none"
                ),
            )


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure to `experiments/out/` as both .pgf and .pdf."""

    fig.savefig(OUT_DIR / f"{stem}.pgf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")


def plot_predictions_and_residuals(
    train_obs: np.ndarray,
    pred_indices: np.ndarray,
    mu_preds: np.ndarray,
    std_preds: np.ndarray,
    validation_obs: np.ndarray,
    mu_forecast: np.ndarray,
    std_forecast: np.ndarray,
    stem: str,
    changepoints: Optional[Sequence[Tuple[int, str]]] = None,
) -> plt.Figure:
    """
    Observations, online one-step-ahead predictions and the validation forecast, with
    the residuals of both underneath.
    """

    fig, (ax_pred, ax_res) = plt.subplots(
        2,
        1,
        figsize=DOUBLE_COL,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    num_train = len(train_obs)
    ax_pred.plot(
        np.arange(num_train),
        train_obs,
        label="training observations",
        color="tab:red",
    )
    ax_pred.plot(
        pred_indices,
        mu_preds,
        label="online one-step mean",
        color="tab:blue",
    )
    ax_pred.fill_between(
        pred_indices,
        mu_preds - std_preds,
        mu_preds + std_preds,
        color="tab:blue",
        alpha=0.3,
        label=r"$\pm 1\sigma$ online",
    )

    forecast_range = np.arange(num_train, num_train + len(validation_obs))
    ax_pred.plot(
        forecast_range,
        validation_obs,
        label="validation observations",
        color="tab:red",
        linestyle="--",
    )
    ax_pred.plot(
        forecast_range,
        mu_forecast,
        label="validation forecast mean",
        color="tab:blue",
        linestyle="--",
    )
    ax_pred.fill_between(
        forecast_range,
        mu_forecast - std_forecast,
        mu_forecast + std_forecast,
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

    ax_res.plot(
        pred_indices,
        train_obs[pred_indices] - mu_preds,
        label="online one-step",
        color="tab:blue",
        linestyle="-",
    )
    ax_res.plot(
        forecast_range,
        validation_obs - mu_forecast,
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

    mark_changepoints(ax_pred, changepoints, with_labels=True)
    mark_changepoints(ax_res, changepoints)

    fig.tight_layout()
    save_figure(fig, stem)
    return fig


def plot_model_comparison(
    runs: Sequence[Dict[str, object]],
    train_obs: np.ndarray,
    validation_obs: np.ndarray,
    stem: str,
    changepoints: Optional[Sequence[Tuple[int, str]]] = None,
) -> plt.Figure:
    """
    One row per model: the observations, that model's online one-step-ahead predictions
    with a ±1σ band, and its validation forecast. The rows share both axes, so the two
    models can be compared by eye at any time step.

    Args:
        runs (Sequence[Dict[str, object]]): one dict per model, with keys `name`,
            `pred_indices`, `mu_preds`, `std_preds`, `mu_forecast`, `std_forecast`.
        train_obs (np.ndarray): Training observations.
        validation_obs (np.ndarray): Validation observations.
        stem (str): File stem to save under.
        changepoints (Optional[Sequence[Tuple[int, str]]]): Marks for every row.
    """

    num_train = len(train_obs)
    forecast_range = np.arange(num_train, num_train + len(validation_obs))
    fig, axes = plt.subplots(
        len(runs),
        1,
        figsize=(DOUBLE_COL[0], 1.9 * len(runs) + 0.9),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for ax, run in zip(axes, runs):
        pred_indices = np.asarray(run["pred_indices"])
        mu_preds = np.asarray(run["mu_preds"]).flatten()
        std_preds = np.asarray(run["std_preds"]).flatten()

        ax.plot(
            np.arange(num_train),
            train_obs,
            label="observations",
            color="tab:red",
            linewidth=0.8,
        )
        ax.plot(
            forecast_range,
            validation_obs,
            color="tab:red",
            linewidth=0.8,
            linestyle="--",
        )
        ax.plot(
            pred_indices,
            mu_preds,
            label="online one-step mean",
            color="tab:blue",
        )
        ax.fill_between(
            pred_indices,
            mu_preds - std_preds,
            mu_preds + std_preds,
            color="tab:blue",
            alpha=0.3,
            label=r"$\pm 1\sigma$",
        )
        mu_forecast = np.asarray(run["mu_forecast"]).flatten()
        std_forecast = np.asarray(run["std_forecast"]).flatten()
        ax.plot(
            forecast_range,
            mu_forecast,
            label="validation forecast",
            color="tab:blue",
            linestyle="--",
        )
        ax.fill_between(
            forecast_range,
            mu_forecast - std_forecast,
            mu_forecast + std_forecast,
            color="tab:blue",
            alpha=0.3,
        )
        ax.set_ylabel(run["name"])
        ax.grid(True, alpha=0.25, linewidth=0.5)
        mark_changepoints(ax, changepoints, with_labels=ax is axes[0])

    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
    )
    axes[-1].set_xlabel("Time step")
    fig.tight_layout()
    save_figure(fig, stem)
    return fig


PRIOR_COLOR = "#1f77b4"  # before y(t) is used
POSTERIOR_COLOR = "#7570b3"  # after y(t) is used
OBS_COLOR = "0.45"


def plot_prior_vs_posterior(
    runs: Sequence[Dict[str, object]],
    observations: np.ndarray,
    stem: str,
    noise_var: Optional[float] = None,
    changepoints: Optional[Sequence[Tuple[int, str]]] = None,
    zoom: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    One row per model: the prior of the hidden state at every step, the posterior of that
    same state, and the observation, each estimate with a ±1σ band. A final row puts the
    correction `posterior - prior` of every model on one axes.

    Prior and posterior are the same quantity either side of the update at that step, so
    the correction is what the observation actually bought. It is usually small, and
    deliberately so: the update moves the state by the Kalman gain
    `var_prior / (var_prior + var_noise)`, so a model that is *confident* moves very
    little even when it is badly wrong. That is why the last row exists - at data scale
    the posterior is drawn on top of the prior and the difference cannot be read off the
    first rows.

    Args:
        runs (Sequence[Dict[str, object]]): one dict per model, with keys `name`,
            `prior_mu`, `prior_std`, `posterior_mu`, `posterior_std`.
        observations (np.ndarray): the observations the models were scored against.
        stem (str): file stem to save under.
        noise_var (Optional[float]): observation-noise variance. If given, each row's
            title also carries that model's mean Kalman gain, which is what explains the
            size of its correction.
        changepoints (Optional[Sequence[Tuple[int, str]]]): marks for every row.
        zoom (Optional[Tuple[int, int]]): `(start, stop)` to restrict the x range to.
    """

    start, stop = zoom if zoom else (0, len(observations))
    steps = np.arange(start, stop)
    fig, axes = plt.subplots(
        len(runs) + 1,
        1,
        figsize=(DOUBLE_COL[0], 1.6 * len(runs) + 2.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2] * len(runs) + [1.4]},
    )
    state_axes = axes[:-1]
    ax_update = axes[-1]

    for ax, run in zip(state_axes, runs):
        ax.plot(
            steps,
            observations[start:stop],
            color=OBS_COLOR,
            linewidth=0.7,
            alpha=0.85,
            label="observation",
            zorder=1,
        )
        for key, color, label in (
            ("prior", PRIOR_COLOR, "prior (before $y_t$)"),
            ("posterior", POSTERIOR_COLOR, "posterior (after $y_t$)"),
        ):
            mu = np.asarray(run[f"{key}_mu"])[start:stop]
            std = np.asarray(run[f"{key}_std"])[start:stop]
            ax.fill_between(
                steps, mu - std, mu + std, color=color, alpha=0.25, linewidth=0
            )
            ax.plot(steps, mu, color=color, linewidth=1.0, label=label, zorder=2)

        title = run["name"]
        if noise_var is not None:
            prior_var = np.asarray(run["prior_std"]) ** 2
            gain = float(np.mean(prior_var / (prior_var + noise_var)))
            title = f"{title}   (mean Kalman gain {gain:.2f})"
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        mark_changepoints(ax, changepoints, with_labels=ax is state_axes[0])

    for run, color in zip(runs, ("tab:blue", "tab:orange", "tab:green")):
        correction = (
            np.asarray(run["posterior_mu"]) - np.asarray(run["prior_mu"])
        )[start:stop]
        ax_update.plot(steps, correction, color=color, linewidth=0.9, label=run["name"])
    ax_update.axhline(0.0, color="0.6", linewidth=0.6)
    ax_update.set_title("what the update moved: posterior $-$ prior", fontsize=9,
                        loc="left")
    ax_update.set_ylabel("Correction")
    ax_update.set_xlabel("Test time step")
    ax_update.grid(True, alpha=0.2, linewidth=0.5)
    ax_update.legend(loc="lower left", ncol=3, frameon=False, fontsize=7.5)
    mark_changepoints(ax_update, changepoints)

    state_axes[0].legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False,
        fontsize=7.5,
    )
    fig.tight_layout()
    save_figure(fig, stem)
    return fig


def plot_error_comparison(
    runs: Sequence[Dict[str, object]],
    observations: np.ndarray,
    stem: str,
    window: int,
    changepoints: Optional[Sequence[Tuple[int, str]]] = None,
    noise_floor: Optional[float] = None,
) -> plt.Figure:
    """
    Rolling mean squared error of the online predictions of every model, on one axes.

    This is where the comparison is actually settled: the prediction rows show that both
    models track the signal, this shows which one is closer and for how long.
    """

    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    for run, color in zip(runs, ("tab:blue", "tab:orange", "tab:green")):
        pred_indices = np.asarray(run["pred_indices"])
        errors = (observations[pred_indices] - np.asarray(run["mu_preds"]).flatten()) ** 2
        rolling = np.convolve(errors, np.ones(window) / window, mode="valid")
        ax.plot(
            pred_indices[window - 1 :],
            rolling,
            label=run["name"],
            color=color,
        )
    if noise_floor is not None:
        ax.axhline(
            noise_floor,
            color="black",
            linewidth=0.8,
            linestyle=":",
            label="noise floor",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Time step")
    ax.set_ylabel(f"Online MSE, {window}-step rolling mean")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    mark_changepoints(ax, changepoints, with_labels=True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(runs) + 1,
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, stem)
    return fig


def plot_state_records(
    records: Dict[str, Dict[str, np.ndarray]],
    states_name: Sequence[str],
    smooth_lag: int,
    stem: str,
    changepoints: Optional[Sequence[Tuple[int, str]]] = None,
) -> plt.Figure:
    """
    One panel per hidden state with its t+1 prediction, its t+1 filtered value and its
    lag-`smooth_lag` smoothed value. The prediction and the smoothed value carry a
    ±1σ band.
    """

    num_time_steps = len(records[states_name[0]]["predict_mu"])
    time_index = np.arange(num_time_steps)
    fig, axes = plt.subplots(
        len(states_name),
        1,
        figsize=(DOUBLE_COL[0], 1.6 * len(states_name) + 0.6),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, states_name):
        record = records[name]
        ax.plot(
            time_index,
            record["predict_mu"],
            label=r"$t+1$ prediction",
            color="tab:blue",
        )
        ax.fill_between(
            time_index,
            record["predict_mu"] - record["predict_std"],
            record["predict_mu"] + record["predict_std"],
            color="tab:blue",
            alpha=0.2,
        )
        ax.plot(
            time_index,
            record["filter_mu"],
            label=r"$t+1$ filtered",
            color="tab:orange",
            linewidth=0.6,
        )
        ax.plot(
            time_index,
            record["smooth_mu"],
            label=f"smoothed (lag {smooth_lag})",
            color="tab:green",
        )
        ax.fill_between(
            time_index,
            record["smooth_mu"] - record["smooth_std"],
            record["smooth_mu"] + record["smooth_std"],
            color="tab:green",
            alpha=0.2,
        )
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        mark_changepoints(ax, changepoints, with_labels=ax is axes[0])

    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    axes[-1].set_xlabel("Time step")
    fig.tight_layout()
    save_figure(fig, stem)
    return fig


def plot_decomposition(
    records: Dict[str, Dict[str, np.ndarray]],
    component_names: Sequence[str],
    observations: np.ndarray,
    stem: str,
    colors: Optional[Sequence[str]] = None,
    changepoints: Optional[Sequence[Tuple[int, str]]] = None,
) -> plt.Figure:
    """
    The t+1 predictions of the named states, their sum, and the observations. The sum is
    what the model predicts for the observation, so it should lie on top of the data.
    """

    if colors is None:
        colors = ["tab:orange", "tab:blue", "tab:purple", "tab:brown"]

    time_index = np.arange(len(observations))
    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    ax.plot(
        time_index,
        observations,
        label="observations",
        color="tab:red",
        alpha=0.5,
    )

    total = np.zeros_like(time_index, dtype=np.float32)
    for name, color in zip(component_names, colors):
        component = records[name]["predict_mu"]
        ax.plot(time_index, component, label=name, color=color)
        total = total + component

    ax.plot(
        time_index,
        total,
        label=" + ".join(component_names),
        color="black",
        linestyle="--",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$t+1$ prediction")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    mark_changepoints(ax, changepoints, with_labels=True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(component_names) + 2,
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, stem)
    return fig


def plot_decomposition_rows(
    records: Dict[str, Dict[str, np.ndarray]],
    states_name: Sequence[str],
    observation_component_names: Sequence[str],
    observations: np.ndarray,
    stem: str,
    colors: Optional[Sequence[str]] = None,
    changepoints: Optional[Sequence[Tuple[int, str]]] = None,
) -> plt.Figure:
    """Plot an additive decomposition with one state per row.

    The first row compares the observations with the reconstructed observation prior. The
    remaining rows show the one-step-ahead prior of every requested state with its marginal
    uncertainty. ``observation_component_names`` is separate from ``states_name`` because
    transition states such as ``trend`` should be displayed but are not directly added to
    the observation.

    This follows the stacked-state convention used by :func:`canari.plot_states` in the
    examples while retaining the absolute online prediction indices stored in ``records``.
    """

    if not states_name:
        raise ValueError("states_name must contain at least one state.")
    if not observation_component_names:
        raise ValueError("observation_component_names must contain at least one state.")

    missing = [
        name
        for name in set(states_name) | set(observation_component_names)
        if name not in records
    ]
    if missing:
        raise ValueError(f"states missing from decomposition records: {sorted(missing)}")

    if colors is None:
        colors = ["tab:orange", "tab:green", "tab:blue", "tab:purple"]
    if len(colors) < len(states_name):
        raise ValueError("colors must provide at least one color per state.")

    observations = np.asarray(observations).flatten()
    num_time_steps = len(observations)
    if any(
        len(records[name]["predict_mu"]) != num_time_steps for name in states_name
    ):
        raise ValueError("state records and observations must have the same length.")

    time_index = np.arange(num_time_steps)
    fig, axes = plt.subplots(
        len(states_name) + 1,
        1,
        figsize=(DOUBLE_COL[0], 1.35 * (len(states_name) + 1) + 0.55),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25] + [1.0] * len(states_name)},
    )
    axes = np.atleast_1d(axes)

    reconstructed = np.zeros(num_time_steps, dtype=np.float32)
    reconstructed_var = np.zeros(num_time_steps, dtype=np.float32)
    for name in observation_component_names:
        reconstructed += records[name]["predict_mu"]
        reconstructed_var += records[name]["predict_std"] ** 2
    reconstructed_std = np.sqrt(reconstructed_var)

    axes[0].plot(
        time_index,
        observations,
        color="tab:red",
        alpha=0.55,
        linewidth=0.8,
        label="observation",
    )
    axes[0].plot(
        time_index,
        reconstructed,
        color="black",
        linewidth=1.0,
        label=" + ".join(observation_component_names),
    )
    axes[0].fill_between(
        time_index,
        reconstructed - reconstructed_std,
        reconstructed + reconstructed_std,
        color="black",
        alpha=0.12,
        label=r"latent $\pm 1\sigma$",
    )
    axes[0].set_ylabel("observation")
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    for ax, name, color in zip(axes[1:], states_name, colors):
        mu = records[name]["predict_mu"]
        std = records[name]["predict_std"]
        ax.plot(time_index, mu, color=color, linewidth=1.0)
        ax.fill_between(
            time_index,
            mu - std,
            mu + std,
            color=color,
            alpha=0.2,
        )
        if name in {"trend", "acceleration"}:
            ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.6)
        ax.set_ylabel(name)

    for index, ax in enumerate(axes):
        ax.grid(True, alpha=0.25, linewidth=0.5)
        mark_changepoints(ax, changepoints, with_labels=index == 0)

    axes[-1].set_xlabel("Time step")
    fig.align_ylabels(axes)
    fig.tight_layout()
    save_figure(fig, stem)
    return fig


def plot_parameter_diagnostics(
    history: Dict[str, Dict[str, list]],
    diagnostic_indices: Sequence[int],
    ylabel: str,
    stem: str,
    skip_layers: Sequence[str] = ("SLinear.1",),
    changepoints: Optional[Sequence[Tuple[int, str]]] = None,
) -> plt.Figure:
    """
    Per-layer parameter-change history over the online windows, weights and bias summed.
    """

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    for layer, parts in history.items():
        if layer in skip_layers:
            continue
        total = np.array(parts["weights"]) + np.array(parts["bias"])
        ax.plot(
            np.array(diagnostic_indices[: len(total)]),
            total,
            label=f"{layer} (w+b)",
        )
    ax.set_xlabel("Window end index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    mark_changepoints(ax, changepoints, with_labels=True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, stem)
    return fig


def plot_performance_grid(
    look_back_lengths: Sequence[int],
    smoothing_windows: Sequence[int],
    online_mse: np.ndarray,
    validation_mse: np.ndarray,
    stem: str,
) -> plt.Figure:
    """
    Plot two annotated heatmaps for an online-LSTM hyperparameter sweep.

    Rows are smoothing-window lengths and columns are LSTM look-back lengths. Each
    panel has its own color scale because online one-step and multi-step validation
    errors can have materially different ranges; the cell labels retain the exact
    comparison across panels.
    """

    look_back_lengths = list(look_back_lengths)
    smoothing_windows = list(smoothing_windows)
    expected_shape = (len(smoothing_windows), len(look_back_lengths))
    online_mse = np.asarray(online_mse, dtype=float)
    validation_mse = np.asarray(validation_mse, dtype=float)
    if online_mse.shape != expected_shape or validation_mse.shape != expected_shape:
        raise ValueError(
            "Performance grids must have shape "
            f"{expected_shape}; got {online_mse.shape} and {validation_mse.shape}."
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    panels = (
        (axes[0], online_mse, "Online one-step MSE"),
        (axes[1], validation_mse, "Validation forecast MSE"),
    )

    for ax, values, title in panels:
        image = ax.imshow(values, aspect="auto", cmap="Blues")
        finite_values = values[np.isfinite(values)]
        midpoint = (
            (float(np.min(finite_values)) + float(np.max(finite_values))) / 2
            if finite_values.size
            else 0.0
        )

        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                # Significant digits keep both ordinary values and very small errors
                # distinguishable (fixed decimals would render several cells as 0.0000).
                label = f"{value:.3g}" if np.isfinite(value) else "--"
                ax.text(
                    col,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if np.isfinite(value) and value > midpoint
                        else "#222222"
                    ),
                    fontsize=8,
                )

        if finite_values.size:
            best_row, best_col = np.unravel_index(
                np.nanargmin(values),
                values.shape,
            )
            # A white keyline keeps the gold circle visible on both ends of the blue
            # scale. The repeated circular marker also works without relying on color.
            ax.scatter(
                best_col,
                best_row,
                s=1700,
                facecolors="none",
                edgecolors="white",
                linewidths=4,
                zorder=4,
            )
            ax.scatter(
                best_col,
                best_row,
                s=1700,
                facecolors="none",
                edgecolors="#E17C05",
                linewidths=2,
                zorder=5,
            )

        ax.set_title(title)
        ax.set_xlabel("LSTM look-back length")
        ax.set_xticks(np.arange(len(look_back_lengths)), look_back_lengths)
        ax.set_yticks(np.arange(len(smoothing_windows)), smoothing_windows)
        ax.tick_params(which="minor", bottom=False, left=False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("MSE")

    axes[0].set_ylabel("Smoothing window length (D)")
    axes[1].set_ylabel("")
    fig.suptitle("Online LSTM performance grid (standardized scale; lower is better)")
    save_figure(fig, stem)
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    return fig
