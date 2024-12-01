from dataclasses import dataclass, field
import numpy as np


@dataclass
class LstmOutputHistory:
    mu: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)

    def initialize(self, look_back_len: int):
        self.mu = 0 * np.ones(look_back_len, dtype=np.float32)
        self.var = 0 * np.ones(look_back_len, dtype=np.float32)


@dataclass
class StatesHistory:
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
class MarginalProbability:
    norm: np.ndarray = field(init=False)
    abnorm: np.ndarray = field(init=False)

    def initialize(self, look_back_len: int):
        self.norm = 0 * np.ones(look_back_len, dtype=np.float32)
        self.abnorm = 0 * np.ones(look_back_len, dtype=np.float32)


def initialize_transition():
    """
    Create a dictionary for model transition
    """
    return {
        "norm_norm": None,
        "abnorm_abnorm": None,
        "norm_abnorm": None,
        "abnorm_norm": None,
    }


def initialize_marginal():
    """
    Create a dictionary for models
    """
    return {
        "norm": None,
        "abnorm": None,
    }
