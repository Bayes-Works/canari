"""
Baseline components for Canari

This module defines three state-space components for modeling local level,
local trend, and local acceleration dynamics. Each class inherits from `BaseComponent`.
"""

from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent


class LocalLevel(BaseComponent):
    """
    Local level component for modeling baselines with a constant level over time.

    Args:
        std_error (Optional[float]): Process noise standard deviation. Default is 0.0.
        mu_states (Optional[np.ndarray]): Initial mean values.
        var_states (Optional[np.ndarray]): Initial covariance matrix.

    Raises:
        ValueError: If dimensions of `mu_states` or `var_states` do not match expected number of states.

    Examples:
        >>> level = LocalLevel(std_error=0.5)
        >>> print(level.component_name)
        local level
        >>> level.transition_matrix
        array([[1]])
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        mu_states: Optional[np.ndarray] = None,
        var_states: Optional[np.ndarray] = None,
    ):
        self.std_error = std_error
        self._mu_states = mu_states
        self._var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "local level"

    def initialize_num_states(self):
        self._num_states = 1

    def initialize_states_name(self):
        self._states_name = ["local level"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[1]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = np.array([[self.std_error**2]])

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self._num_states, 1))
        elif len(self._mu_states) == self._num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
        else:
            raise ValueError(
                "Incorrect mu_states dimension for the local level component."
            )

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self._num_states, 1))
        elif len(self._var_states) == self._num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(
                "Incorrect var_states dimension for the local level component."
            )


class LocalTrend(BaseComponent):
    """
    Local trend component for modeling baselines with a constant speed over time.
    This component captures a latent linear trend with both intercept and slope terms.

    Args:
        std_error (Optional[float]): Process noise standard deviation. Default is 0.0.
        mu_states (Optional[np.ndarray]): Initial mean values.
        var_states (Optional[np.ndarray]): Initial covariance matrix.

    Raises:
        ValueError: If dimensions of `mu_states` or `var_states` do not match expected number of states.

    Examples:
        >>> trend = LocalTrend(std_error=0.2)
        >>> trend.states_name
        ['local level', 'local trend']
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        mu_states: Optional[np.ndarray] = None,
        var_states: Optional[np.ndarray] = None,
    ):
        self.std_error = std_error
        self._mu_states = mu_states
        self._var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "local trend"

    def initialize_num_states(self):
        self._num_states = 2

    def initialize_states_name(self):
        self._states_name = ["local level", "local trend"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[1, 1], [0, 1]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1, 0]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = self.std_error**2 * np.array(
            [[1 / 3, 1 / 2], [1 / 2, 1 / 3]]
        )

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self._num_states, 1))
        elif len(self._mu_states) == self._num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
        else:
            raise ValueError(
                "Incorrect mu_states dimension for the local trend component."
            )

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self._num_states, 1))
        elif len(self._var_states) == self._num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(
                "Incorrect var_states dimension for the local trend component."
            )


class LocalAcceleration(BaseComponent):
    """
    Local acceleration component for modeling baselines with a constant acceleration over time.
    This component models three hidden states: level, slope, and acceleration in order to capture a curvature baseline.

    Args:
        std_error (Optional[float]): Process noise standard deviation. Default is 0.0.
        mu_states (Optional[np.ndarray]): Initial mean values.
        var_states (Optional[np.ndarray]): Initial covariance matrix.

    Raises:
        ValueError: If dimensions of `mu_states` or `var_states` do not match expected number of states.

    Examples:
        >>> acc = LocalAcceleration(std_error=0.3)
        >>> acc.transition_matrix
        array([[1. , 1. , 0.5],
               [0. , 1. , 1. ],
               [0. , 0. , 1. ]])
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        mu_states: Optional[np.ndarray] = None,
        var_states: Optional[np.ndarray] = None,
    ):
        self.std_error = std_error
        self._mu_states = mu_states
        self._var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "local acceleration"

    def initialize_num_states(self):
        self._num_states = 3

    def initialize_states_name(self):
        self._states_name = ["local level", "local trend", "local acceleration"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 1]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1, 0, 0]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = self.std_error**2 * np.array(
            [[1 / 20, 1 / 8, 1 / 6], [1 / 8, 1 / 3, 0.5], [1 / 6, 0.5, 1]]
        )

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self._num_states, 1))
        elif len(self._mu_states) == self._num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
        else:
            raise ValueError(
                "Incorrect mu_states dimension for the local acceleration component."
            )

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self._num_states, 1))
        elif len(self._var_states) == self._num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(
                "Incorrect var_states dimension for the local acceleration component."
            )
