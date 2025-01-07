from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class LstmOutputHistory:
    mu: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)

    def initialize(self, look_back_len: int):
        self.mu = 0 * np.ones(look_back_len, dtype=np.float32)
        self.var = 1 * np.ones(look_back_len, dtype=np.float32)


@dataclass
class StatesHistory:
    mu_prior: List[np.ndarray] = field(init=False)
    var_prior: List[np.ndarray] = field(init=False)
    mu_posterior: List[np.ndarray] = field(init=False)
    var_posterior: List[np.ndarray] = field(init=False)
    mu_smooth: List[np.ndarray] = field(init=False)
    var_smooth: List[np.ndarray] = field(init=False)
    cov_states: List[np.ndarray] = field(init=False)
    states_name: List[str] = field(init=False)

    def initialize(self, states_name: List[str]) -> None:
        self.mu_prior = []
        self.var_prior = []
        self.mu_posterior = []
        self.var_posterior = []
        self.mu_smooth = []
        self.var_smooth = []
        self.cov_states = []
        self.states_name = states_name


def initialize_marginal_prob_history(num_time_steps):
    """
    Create a dictionary saving marginal probability
    """
    return {
        "norm": np.zeros(num_time_steps, dtype=np.float32),
        "abnorm": np.zeros(num_time_steps, dtype=np.float32),
    }


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
