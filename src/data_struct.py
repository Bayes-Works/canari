from dataclasses import dataclass, field
import numpy as np


@dataclass
class LstmOutputHistory:
    mu: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)

    def initialize(self, look_back_len: int):
        self.mu = 0.1 * np.ones(look_back_len, dtype=np.float32)
        self.var = 1 * np.ones(look_back_len, dtype=np.float32)


@dataclass
class SmootherStates:
    mu_prior: np.ndarray = field(init=False)
    var_prior: np.ndarray = field(init=False)
    mu_posterior: np.ndarray = field(init=False)
    var_posterior: np.ndarray = field(init=False)
    mu_smooth: np.ndarray = field(init=False)
    var_smooth: np.ndarray = field(init=False)
    cov_states: np.ndarray = field(init=False)

    def initialize(self, num_time_steps: int, num_states: int) -> None:
        self.mu_prior = np.zeros((num_time_steps, num_states), dtype=np.float32)
        self.var_prior = np.zeros(
            (num_time_steps, num_states, num_states), dtype=np.float32
        )
        self.mu_posterior = np.zeros((num_time_steps, num_states), dtype=np.float32)
        self.var_posterior = np.zeros(
            (num_time_steps, num_states, num_states), dtype=np.float32
        )
        self.mu_smooth = np.zeros((num_time_steps, num_states), dtype=np.float32)
        self.var_smooth = np.zeros(
            (num_time_steps, num_states, num_states), dtype=np.float32
        )
        self.cov_states = np.zeros(
            (num_time_steps, num_states, num_states), dtype=np.float32
        )


@dataclass
class SmootherStatesSKF:
    var_smooth: np.ndarray = field(init=False)
    cov_states: np.ndarray = field(init=False)

    def initialize(self, num_time_steps: int, num_states: int) -> None:
        self.mu_smooth = np.zeros((num_time_steps, num_states), dtype=np.float32)
        self.var_smooth = np.zeros(
            (num_time_steps, num_states, num_states), dtype=np.float32
        )
