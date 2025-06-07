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
    over a fixed lookback window. New predictions shift the window forward,
    saving the most recent predictions, and discard the oldest ones.

    Examples:
        >>> lstm_history = LstmOutputHistory()
        >>> lstm_history.initialize(look_back_len=10)
        >>> lstm_history.update(mu_lstm=np.array([0.5]), var_lstm=np.array([0.2]))

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
class LstmEmbedding:
    """
    Container for saving the LSTM embedding mean and variance.

    Attributes:
        mu (np.ndarray): Mean of the LSTM embedding matrix (nb_ts x embedding_dim).
        var (np.ndarray): Variance of the LSTM embedding matrix (nb_ts x embedding_dim).
    """

    mu: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)

    def initialize(self, embedding_dim: int, nb_ts: int = 1):
        """
        Initialize `mu` and `var` using Bayesian-appropriate initialization strategies.

        Args:
            embedding_dim (int): Dimension of the LSTM embedding.
            nb_ts (int): Number of time steps. Defaults to 1.
        """
        self.mu = np.random.normal(0, 1, (nb_ts, embedding_dim)).astype(np.float32)
        self.var = np.ones((nb_ts, embedding_dim), dtype=np.float32)

    def update(
        self,
        delta_mu: np.ndarray,
        delta_var: np.ndarray,
        ts_idx: int = 0,
        learning_rate: float = 1,
    ):
        """
        Update the LSTM embedding mean and variance for a specific time series.

        Args:
            delta_mu (np.ndarray): Change in the mean of the LSTM embedding.
            delta_var (np.ndarray): Change in the variance of the LSTM embedding.
            i (int): Index of the time series to update.
            learning_rate (float): Learning rate for the update. Defaults to 1.
        """
        self.mu[ts_idx] += learning_rate * delta_mu * self.mu[ts_idx]
        self.var[ts_idx] += (
            learning_rate * self.var[ts_idx] * delta_var * self.var[ts_idx]
        )
    def __getitem__(self, ts_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the mean and variance for a specific time series index.

        Args:
            ts_idx (int): Index of the time series.

        Returns:
            tuple[np.ndarray, np.ndarray]: Mean and variance arrays for the specified index.
        """
        return self.mu[ts_idx], self.var[ts_idx]


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
    ) -> dict[str, np.ndarray]:
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
    ) -> dict[str, np.ndarray]:
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
