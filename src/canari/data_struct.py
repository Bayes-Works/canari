from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class LstmOutputHistory:
    """LSTM output history"""

    mu: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)

    def initialize(self, look_back_len: int):
        self.mu = np.zeros(look_back_len, dtype=np.float32)
        self.var = np.ones(look_back_len, dtype=np.float32)

    def update(self, mu_lstm, var_lstm):
        """Update lstm output history"""

        self.mu = np.roll(self.mu, -1)
        self.var = np.roll(self.var, -1)
        self.mu[-1] = mu_lstm.item()
        self.var[-1] = var_lstm.item()


@dataclass
class StatesHistory:
    """States history: prior, posterior, smoother and cross-covariance matrix"""

    mu_prior: List[np.ndarray] = field(init=False)
    var_prior: List[np.ndarray] = field(init=False)
    mu_posterior: List[np.ndarray] = field(init=False)
    var_posterior: List[np.ndarray] = field(init=False)
    mu_smooth: List[np.ndarray] = field(init=False)
    var_smooth: List[np.ndarray] = field(init=False)
    cov_states: List[np.ndarray] = field(init=False)
    states_name: List[str] = field(init=False)

    def initialize(self, states_name: List[str]):
        self.mu_prior = []
        self.var_prior = []
        self.mu_posterior = []
        self.var_posterior = []
        self.mu_smooth = []
        self.var_smooth = []
        self.cov_states = []
        self.states_name = states_name

    def get_mean(
        self,
        states_type: Optional[str] = "posterior",
        states_name: Optional[list[str]] = "all",
    ) -> dict[str, np.ndarray]:
        """Get mean values for hidden states"""
        mean = {}

        if states_name == "all":
            states_name = self.states_name

        if states_type == "prior":
            values = np.array(self.mu_prior)
        elif states_type == "posterior":
            values = np.array(self.mu_posterior)
        elif states_type == "smooth":
            values = np.array(self.mu_smooth)
        else:
            raise ValueError(
                f"Incorrect states_types, should choose among 'prior', 'posterior', or 'smooth' ."
            )

        for state in states_name:
            idx = self.states_name.index(state)
            mean[state] = values[:, idx].flatten()

        return mean

    def get_std(
        self,
        states_type: Optional[str] = "posterior",
        states_name: Optional[list[str]] = "all",
    ) -> dict:
        """Get mean values for hidden states"""

        standard_deviation = {}

        if states_name == "all":
            states_name = self.states_name

        if states_type == "prior":
            values = np.array(self.var_prior)
        elif states_type == "posterior":
            values = np.array(self.var_posterior)
        elif states_type == "smooth":
            values = np.array(self.var_smooth)
        else:
            raise ValueError(
                f"Incorrect states_types, should choose among 'prior', 'posterior', or 'smooth' ."
            )

        for state in states_name:
            idx = self.states_name.index(state)
            standard_deviation[state] = values[:, idx, idx] ** 0.5

        return standard_deviation
