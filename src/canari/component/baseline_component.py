"""
This module defines three components for modelling baselines including `LocalLevel`, `LocalTrend`, and `LocalAcceleration`.
"""

from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent


class LocalLevel(BaseComponent):
    """
    `LocalLevel` class, inheriting from Canari's `BaseComponent`.
    It models baselines with a locally constant level (zero speed) over time. It has one hidden state
    `level`.

    Args:
        std_error (Optional[float]): Standard deviation of the process noise. Defaults: initialized to zeros 0.
        mu_states (Optional[list[float]]): Initial mean of the hidden states. Defaults:
            initialized to zeros.
        var_states (Optional[list[float]]): Initial variance of the hidden states. Defaults:
            initialized to zeros.

    Examples:
        >>> from canari.component import LocalLevel
        >>> # With known mu_states and var_states
        >>> level = LocalLevel(mu_states=[1], var_states=[1],std_error=0.5)
        >>> # With default mu_states and var_states
        >>> level = LocalLevel(std_error=0.5)
        >>> level.component_name
        'local level'
        >>> level.states_name
        ['level']
        >>> level.transition_matrix
        array([[1]])
        >>> level.observation_matrix
        array([[1]])
        >>> level.process_noise_matrix
        >>> level.mu_states
        >>> level.var_states
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        mu_states: Optional[list[float]] = None,
        var_states: Optional[list[float]] = None,
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
        self._states_name = ["level"]

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
    `LocalTrend` class, inheriting from Canari's `BaseComponent`.
    It models baselines with a locally constant speed over time (linear level). It has two
    hidden states `level` and `trend`.

    Args:
        std_error (Optional[float]): Standard deviation of the process noise. Defaults: initialized to 0.
        mu_states (Optional[list[float]]): Initial mean of the hidden state. Defaults:
            initialized to zeros.
        var_states (Optional[list[float]]): Initial variance of the hidden state. Defaults:
            initialized to zeros.

    Examples:
        >>> from canari.component import LocalTrend
        >>> # With known mu_states and var_states
        >>> local_trend = LocalAcceleration(mu_states=[1, 2], var_states=[1, 2], std_error=0.3)
        >>> # With default mu_states and var_states
        >>> local_trend = LocalTrend(std_error=0.2)
        >>> level.component_name
        'local trend'
        >>> local_trend.states_name
        ['level', 'trend']
        >>> local_trend.transition_matrix
        array([[1, 1],
               [0, 1]])
        >>> local_trend.observation_matrix
        array([[1, 0]])
        >>> local_trend.process_noise_matrix
        >>> local_trend.mu_states
        >>> local_trend.var_states

    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        mu_states: Optional[list[float]] = None,
        var_states: Optional[list[float]] = None,
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
        self._states_name = ["level", "trend"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[1, 1], [0, 1]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1, 0]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = self.std_error**2 * np.array(
            [[1 / 4, 1 / 2], [1 / 2, 1]]
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
    `LocalAcceleration` class, inheriting from Canari's `BaseComponent`.
    It models baselines with a locally constant acceleration (linear speed and curvature level) over time.
    It has three hidden states `level`, `trend`, and `acceleration`.

    Args:
        std_error (Optional[float]): Standard deviation of the process noise. Defaults: initialized to 0.
        mu_states (Optional[list[float]]): Initial mean of the hidden state. Defaults:
            initialized to zeros.
        var_states (Optional[list[float]]): Initial variance of the hidden state. Defaults:
            initialized to zeros.

    Examples:
        >>> from canari.component import LocalAcceleration
        >>> # With known mu_states and var_states
        >>> local_acc = LocalAcceleration(mu_states = [1, 2, 3], var_states = [1, 2, 3], std_error=0.3)
        >>> # With default mu_states and var_states
        >>> local_acc = LocalAcceleration(std_error=0.3)
        >>> local_acc.transition_matrix
        array([[1. , 1. , 0.5],
               [0. , 1. , 1. ],
               [0. , 0. , 1. ]])
        >>> local_acc.observation_matrix
        array([[1, 0, 0]])
        >>> local_acc.process_noise_matrix
        >>> local_acc.mu_states
        >>> local_acc.var_states
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        mu_states: Optional[list[float]] = None,
        var_states: Optional[list[float]] = None,
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
        self._states_name = ["level", "trend", "acceleration"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 1]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1, 0, 0]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = self.std_error**2 * np.array(
            [[1 / 4, 0.5, 0.5], [0.5, 1, 1], [0.5, 1, 1]]
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
