from dataclasses import dataclass, field
import numpy as np


@dataclass
class LstmOutputHistory:
    mu: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)

    def initialize(self, look_back_len: int):
        self.mu = np.zeros(look_back_len, dtype=np.float32)
        self.var = 0.1 * np.ones(look_back_len, dtype=np.float32)


@dataclass
class SmootherStates:
    mu_priors: np.ndarray = field(init=False)
    var_priors: np.ndarray = field(init=False)
    mu_posteriors: np.ndarray = field(init=False)
    var_posteriors: np.ndarray = field(init=False)
    mu_smooths: np.ndarray = field(init=False)
    var_smooths: np.ndarray = field(init=False)
    cov_states: np.ndarray = field(init=False)

    def initialize(self, num_time_steps: int, num_states: int) -> None:
        self.mu_priors = np.zeros((num_time_steps, num_states), dtype=np.float32)
        self.var_priors = np.zeros(
            (num_time_steps, num_states, num_states), dtype=np.float32
        )
        self.mu_posteriors = np.zeros((num_time_steps, num_states), dtype=np.float32)
        self.var_posteriors = np.zeros(
            (num_time_steps, num_states, num_states), dtype=np.float32
        )
        self.mu_smooths = np.zeros((num_time_steps, num_states), dtype=np.float32)
        self.var_smooths = np.zeros(
            (num_time_steps, num_states, num_states), dtype=np.float32
        )
        self.cov_states = np.zeros(
            (num_time_steps, num_states, num_states), dtype=np.float32
        )
