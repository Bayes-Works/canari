"""
Module for managing historical values for LSTM' output, cell and hidden states.

This module provides two main data classes:

- `LstmOutputHistory`: Maintains a rolling history of LSTM mean and variance outputs.
- `StatesHistory`: Tracks prior, posterior, and smoothed estimates of hidden states across time,
  including their means, variances, and covariances.

Example:
    >>> lstm_history = LstmOutputHistory()
    >>> lstm_history.initialize(look_back_len=10)
    >>> lstm_history.update(mu_lstm=np.array([0.5]), var_lstm=np.array([0.2]))
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class LstmOutputHistory:
    """
    Container for managing a rolling history of LSTM output mean and variance.

    This class keeps track of the LSTM's predicted output statistics over a fixed look-back
    window. New predictions shift the window forward, maintaining a history of values.

    Attributes:
        mu (np.ndarray): Rolling array storing the LSTM mean outputs.
        var (np.ndarray): Rolling array storing the LSTM variance outputs.
    """

    mu: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)

    def initialize(self, look_back_len: int):
        """
        Initialize the history arrays with a specified look-back length.

        Args:
            look_back_len (int): Number of time steps to keep in history.
        """
        self.mu = np.zeros(look_back_len, dtype=np.float32)
        self.var = np.ones(look_back_len, dtype=np.float32)

    def update(self, mu_lstm, var_lstm):
        """
        Update the rolling window with new LSTM outputs.

        Removes the oldest values and inserts the newest ones at the end of the arrays.

        Args:
            mu_lstm (np.ndarray or float): Latest LSTM mean output.
            var_lstm (np.ndarray or float): Latest LSTM variance output.
        """
        self.mu = np.roll(self.mu, -1)
        self.var = np.roll(self.var, -1)
        self.mu[-1] = mu_lstm.item()
        self.var[-1] = var_lstm.item()


@dataclass
class StatesHistory:
    """
    Tracks historical estimates of hidden states in a filtering/smoothing system.

    Stores the evolution of prior, posterior, and smoothed state distributions over time,
    along with their covariance matrices. Useful for diagnostic analysis in state-space
    models like Kalman filters or probabilistic RNNs.

    Attributes:
        mu_prior (List[np.ndarray]): Mean of the prior state estimates.
        var_prior (List[np.ndarray]): Variance of the prior state estimates.
        mu_posterior (List[np.ndarray]): Mean of the posterior state estimates.
        var_posterior (List[np.ndarray]): Variance of the posterior state estimates.
        mu_smooth (List[np.ndarray]): Mean of the smoothed state estimates.
        var_smooth (List[np.ndarray]): Variance of the smoothed state estimates.
        cov_states (List[np.ndarray]): Full cross-covariance matrices for the states.
        states_name (List[str]): Names of the tracked hidden states.
    """

    mu_prior: List[np.ndarray] = field(init=False)
    var_prior: List[np.ndarray] = field(init=False)
    mu_posterior: List[np.ndarray] = field(init=False)
    var_posterior: List[np.ndarray] = field(init=False)
    mu_smooth: List[np.ndarray] = field(init=False)
    var_smooth: List[np.ndarray] = field(init=False)
    cov_states: List[np.ndarray] = field(init=False)
    states_name: List[str] = field(init=False)

    def initialize(self, states_name: List[str]):
        """
        Initialize the internal storage for state means, variances, and names.

        Args:
            states_name (List[str]): List of hidden state names to be tracked.
        """
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
        """
        Retrieve the time series of mean values for the specified hidden states.

        Args:
            states_type (str, optional): Type of state estimate to return ('prior', 'posterior', 'smooth').
                Defaults to "posterior".
            states_name (list[str] or str, optional): Subset of state names to extract, or "all" for all.
                Defaults to "all".

        Returns:
            dict[str, np.ndarray]: Dictionary mapping state names to 1D arrays of means over time.
        """
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
                f"Incorrect states_type: choose from 'prior', 'posterior', or 'smooth'."
            )

        for state in states_name:
            idx = self.states_name.index(state)
            mean[state] = values[:, idx].flatten()

        return mean

    def get_std(
        self,
        states_type: Optional[str] = "posterior",
        states_name: Optional[list[str]] = "all",
    ) -> dict[str, np.ndarray]:
        """
        Retrieve the time series of standard deviation values for the specified hidden states.

        Args:
            states_type (str, optional): Type of state estimate to return ('prior', 'posterior', 'smooth').
                Defaults to "posterior".
            states_name (list[str] or str, optional): Subset of state names to extract, or "all" for all.
                Defaults to "all".

        Returns:
            dict[str, np.ndarray]: Dictionary mapping state names to 1D arrays of standard deviations over time.
        """
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
                f"Incorrect states_type: choose from 'prior', 'posterior', or 'smooth'."
            )

        for state in states_name:
            idx = self.states_name.index(state)
            standard_deviation[state] = values[:, idx, idx] ** 0.5

        return standard_deviation
