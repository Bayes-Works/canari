"""
This module manages historical values for LSTM' output, cell and hidden states.

It provides two data classes:

- `LstmOutputHistory`: Maintain a rolling history of the LSTM mean and variance outputs.
- `StatesHistory`: Save prior, posterior, and smoothed estimates of hidden states over time.

"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from pytagi import Normalizer


@dataclass
class LstmOutputHistory:
    """
    Container for saving a rolling history of LSTM output means and variances
    over a fixed lookback window.

    Examples:
        >>> lstm_history = LstmOutputHistory()
        >>> lstm_history.initialize(look_back_len=10)
        >>> lstm_history.update(mu_lstm=np.array([0.5]), var_lstm=np.array([0.2]))
        >>> lstm_history.set(mu=np.array([0.1, 0.2, 0.3]), var=np.array([0.01, 0.02, 0.03]))

    Attributes:
        mu (np.ndarray): Rolling array storing the LSTM mean outputs.
        var (np.ndarray): Rolling array storing the LSTM variance outputs.

    """

    mu: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)

    def initialize(self, look_back_len: int):
        """
        Initialize :attr:`mu` and :attr:`var` with a specified lookback length.

        Args:
            look_back_len (int): Number of time steps to keep in history.
        """
        self.mu = np.zeros(look_back_len)
        self.var = np.ones(look_back_len)

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

    def set(self, mu: np.ndarray, var: np.ndarray):
        """
        Set the entire rolling window directly with new values for `mu` and `var`.

        Args:
            mu (np.ndarray): Array of mean values to set.
            var (np.ndarray): Array of variance values to set.

        Raises:
            ValueError: If the shapes of `mu` and `var` do not match the initialized window size.
        """
        self.mu = mu.copy()
        self.var = var.copy()


@dataclass
class StatesHistory:
    """
    Save estimates of hidden states over time.

    Stores the evolution of prior, posterior, and smoothed values for hidden states over time,
    along with cross-covariances between hidden states at two consecutive timesteps `t` and `t+1`.

    Attributes:
        mu_prior (List[np.ndarray]): Mean of the prior hidden states.
        var_prior (List[np.ndarray]): Covariance matrix for the prior hiddens states.
        mu_posterior (List[np.ndarray]): Mean of the posterior hidden states.
        var_posterior (List[np.ndarray]): Covariance matrix for the posterior hidden states.
        mu_smooth (List[np.ndarray]): Mean of the smoothed estimates for hidden states.
        var_smooth (List[np.ndarray]): Covariance matrix for the smoothed estimates
                                        for hidden states.
        cov_states (List[np.ndarray]):  Cross-covariance matrix for the hidden states
                                        at two consecutive timesteps `t` and `t+1`.
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
        Initialize `mu_prior`, `var_prior`, `mu_posterior`, `var_posterior`,
        `mu_smooth`, `var_smooth`, and `cov_states` as empty lists.

        Args:
            states_name (List[str]): List of hidden state names.
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
        states_name: str,
        states_type: Optional[str] = "posterior",
        standardization: Optional[bool] = True,
        scale_const_mean: Optional[float] = 0,
        scale_const_std: Optional[float] = 1,
    ) -> np.ndarray:
        """
        Retrieve the mean values over time for a specified hidden states and for either
        a) the prior predicted value, b) the posterior updated values after the filter step,
        or c) the posterior updated values after the smoother step (smoothed estimates).

        Args:
            states_name (str): Name of hidden state to extract.
            states_type (str, optional): Type of states to return ('prior', 'posterior', 'smooth').
                Defaults to "posterior".
            standardization (bool, optional): Get the standardized values for hidden states.
                                                Defaults to True.
            scale_const_mean (float, optional): Mean used for unstandardization.
            scale_const_std (float, optional): Standard deviation used for unstandardization.

        Returns:
            np.ndarray: 1D arrays of means over time.
        """

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

        idx = self.states_name.index(states_name)
        mean = values[:, idx].flatten()

        if not standardization:
            scale_const_mean = scale_const_mean if states_name == "level" else 0
            mean = Normalizer.unstandardize(mean, scale_const_mean, scale_const_std)

        return mean

    def get_std(
        self,
        states_name: str,
        states_type: Optional[str] = "posterior",
        standardization: Optional[bool] = True,
        scale_const_std: Optional[float] = 1,
    ) -> np.ndarray:
        """
        Retrieve the standard deviation values over time for a specified hidden states and
        for either a) the prior predicted value, b) the posterior updated values after the
        filter step, or c) the posterior updated values after the smoother step
        (smoothed estimates).

        Args:
            states_name (str): Name of hidden state to extract.
            states_type (str, optional): Type of states to return ('prior', 'posterior', 'smooth').
                Defaults to "posterior".
            standardization (bool, optional): Get the standardized values for hidden states.
                                    Defaults to True.
            scale_const_std (float, optional): Standard deviation used for unstandardization.

        Returns:
            np.ndarray: 1D arrays of standard deviations over time.
        """

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

        idx = self.states_name.index(states_name)
        standard_deviation = values[:, idx, idx] ** 0.5

        if not standardization:
            standard_deviation = Normalizer.unstandardize_std(
                standard_deviation, scale_const_std
            )

        return standard_deviation


@dataclass
class OutputHistory:
    """
    Save output history mean and variance
    """

    mu: List[float] = field(init=False)
    var: List[float] = field(init=False)

    def initialize(self):
        """
        Initialize `mu` and `var`  as empty lists.
        """

        self.mu = []
        self.var = []

    def save_output_history(self, mu: float, var: float):
        """
        Add `mu` and `var` to the saved history
        """

        self.mu.append(mu.item())
        self.var.append(var.item())
