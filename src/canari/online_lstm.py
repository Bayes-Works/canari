"""Fixed-lag online learning for LSTM-based Canari models."""

import copy
from typing import Dict, Optional, Tuple

import numpy as np

from canari.data_struct import StatesHistory


class _LookBackBuffer:
    """Keep smoothed LSTM outputs that later overlapping windows need."""

    def __init__(self, look_back_len: int, num_steps: int):
        self.look_back_len = look_back_len
        self.pad = look_back_len - 1
        self.mu = np.zeros(num_steps + self.pad, dtype=np.float32)
        self.var = np.ones(num_steps + self.pad, dtype=np.float32)

    def seed(self, mu: np.ndarray, var: np.ndarray) -> None:
        num_values = min(self.pad, len(mu))
        if num_values:
            self.mu[self.pad - num_values : self.pad] = mu[-num_values:]
            self.var[self.pad - num_values : self.pad] = var[-num_values:]

    def store(self, time_step: int, mu: np.ndarray, var: np.ndarray) -> None:
        self.mu[time_step + self.pad] = mu[0]
        self.var[time_step + self.pad] = var[0]

    def get(self, time_step: int) -> Tuple[np.ndarray, np.ndarray]:
        stop = time_step + self.pad
        start = stop - self.look_back_len
        return self.mu[start:stop], self.var[start:stop]


def _slice_data(
    data: Dict[str, np.ndarray], start: int, end: int
) -> Dict[str, np.ndarray]:
    return {"x": data["x"][start:end], "y": data["y"][start:end]}


def _validate_inputs(model, data, start: int, end: int, window_len: int) -> None:
    if model.lstm_net is None:
        raise ValueError("online_lstm_filter requires an LstmNetwork component")
    if not model.lstm_net.smooth:
        raise ValueError("online_lstm_filter requires LstmNetwork(smoother=True)")
    if window_len < 1:
        raise ValueError("window_len must be at least 1")
    if start < window_len:
        raise ValueError("start must be greater than or equal to window_len")
    if not start < end <= len(data["y"]):
        raise ValueError("expected start < end <= len(data['y'])")


def _new_states_history(states_name) -> StatesHistory:
    history = StatesHistory()
    history.initialize(states_name)
    return history


def _append_last_state(target: StatesHistory, source: StatesHistory) -> None:
    for name in (
        "mu_prior",
        "var_prior",
        "mu_posterior",
        "var_posterior",
        "mu_smooth",
        "var_smooth",
        "cov_states",
    ):
        values = getattr(source, name)
        if values:
            getattr(target, name).append(values[-1].copy())


def _append_all_states(target: StatesHistory, source: StatesHistory) -> None:
    for name in (
        "mu_prior",
        "var_prior",
        "mu_posterior",
        "var_posterior",
        "mu_smooth",
        "var_smooth",
        "cov_states",
    ):
        getattr(target, name).extend(value.copy() for value in getattr(source, name))


def _smooth_model_window(model) -> Tuple[np.ndarray, np.ndarray]:
    for time_step in reversed(range(len(model.states.mu_smooth) - 1)):
        model.rts_smoother(time_step)
    mu, var = model.lstm_net.smoother()
    return np.asarray(mu, dtype=np.float32), np.asarray(var, dtype=np.float32)


def _smooth_skf_window(skf) -> Tuple[np.ndarray, np.ndarray]:
    skf.smoother()
    mu, var = skf.lstm_net.smoother()
    return np.asarray(mu, dtype=np.float32), np.asarray(var, dtype=np.float32)


def _prepare_buffer(model, num_steps: int, seed) -> _LookBackBuffer:
    buffer = _LookBackBuffer(model.lstm_net.lstm_look_back_len, num_steps)
    if seed is not None:
        buffer.seed(*seed)
    return buffer


def _restart_skf(skf) -> None:
    """Reset SKF and LSTM memory while keeping the pretrained parameters."""

    skf.load_initial_states()
    skf._set_same_states_transition_models()
    skf.marginal_prob["norm"] = skf.norm_model_prior_prob
    skf.marginal_prob["abnorm"] = 1 - skf.norm_model_prior_prob
    skf.lstm_output_history.initialize(skf.lstm_net.lstm_look_back_len)

    lstm_states = skf.lstm_net.get_lstm_states()
    zero_states = {
        layer: tuple(np.zeros_like(value).tolist() for value in values)
        for layer, values in lstm_states.items()
    }
    skf.lstm_net.set_lstm_states(zero_states)


def _warm_up_model(model, data, end: int):
    if end == 0:
        return None

    model.lstm_net.num_samples = end
    model.lstm_net.eval()
    _, _, states = model.filter(_slice_data(data, 0, end), train_lstm=False)
    seed = (
        states.get_mean("lstm", "posterior"),
        states.get_std("lstm", "posterior") ** 2,
    )
    model.lstm_net.smoother()
    model.lstm_net.set_lstm_states(model.lstm_net.get_lstm_states(end - 1))
    return seed


def online_filter_model(
    model,
    data: Dict[str, np.ndarray],
    start: int,
    window_len: int,
    end: Optional[int] = None,
):
    """Run fixed-lag online LSTM learning and return one prediction per step."""

    end = len(data["y"]) if end is None else end
    _validate_inputs(model, data, start, end, window_len)

    first_window = start - window_len
    seed = _warm_up_model(model, data, first_window)
    num_steps = end - start
    buffer = _prepare_buffer(model, num_steps + window_len, seed)
    states_history = _new_states_history(model.states_name)
    means, stds = [], []

    model.lstm_net.num_samples = window_len + 1
    for step in range(num_steps):
        window_start = first_window + step
        window_end = window_start + window_len + 1
        model.lstm_net.train()
        mean, std, states = model.filter(
            _slice_data(data, window_start, window_end), train_lstm=True
        )
        means.append(mean[-1])
        stds.append(std[-1])
        _append_last_state(states_history, states)

        filtered_lstm_states = copy.deepcopy(model.lstm_net.get_lstm_states())
        smooth_mu, smooth_var = _smooth_model_window(model)
        buffer.store(step, smooth_mu, smooth_var)
        if step < num_steps - 1:
            model.set_memory(states=states, time_step=1)
            look_back_mu, look_back_var = buffer.get(step + 1)
            model.lstm_output_history.set(look_back_mu, look_back_var)
            model.lstm_net.set_lstm_states(model.lstm_net.get_lstm_states(0))
        else:
            model.set_memory(states=states, time_step=window_len + 1)
            model.lstm_net.set_lstm_states(filtered_lstm_states)

    return np.asarray(means).flatten(), np.asarray(stds).flatten(), states_history


def online_filter_skf(
    skf,
    data: Dict[str, np.ndarray],
    window_len: int,
    end: Optional[int] = None,
):
    """Run fixed-lag online LSTM learning and return the complete SKF history."""

    end = len(data["y"]) if end is None else end
    _validate_inputs(skf, data, window_len, end, window_len)
    _restart_skf(skf)

    num_windows = end - window_len
    buffer = _prepare_buffer(skf, end, seed=None)
    states_history = _new_states_history(skf.states_name)
    anomaly_probabilities = []

    skf.lstm_net.num_samples = window_len + 1
    for step in range(num_windows):
        window_start = step
        window_end = window_start + window_len + 1
        skf.lstm_net.train()
        probabilities, states = skf.filter(
            _slice_data(data, window_start, window_end),
            train_lstm=True,
            reset_memory=False,
        )
        if step == 0:
            anomaly_probabilities.extend(probabilities)
            _append_all_states(states_history, states)
        else:
            anomaly_probabilities.append(probabilities[-1])
            _append_last_state(states_history, states)

        filtered_lstm_states = copy.deepcopy(skf.lstm_net.get_lstm_states())
        smooth_mu, smooth_var = _smooth_skf_window(skf)
        buffer.store(step, smooth_mu, smooth_var)
        if step < num_windows - 1:
            skf.set_memory(states=states, time_step=1)
            look_back_mu, look_back_var = buffer.get(step + 1)
            skf.lstm_output_history.set(look_back_mu, look_back_var)
            skf.lstm_net.set_lstm_states(skf.lstm_net.get_lstm_states(0))
        else:
            skf.set_memory(states=states, time_step=window_len + 1)
            skf.lstm_net.set_lstm_states(filtered_lstm_states)

    return np.asarray(anomaly_probabilities).flatten(), states_history
