"""
Periodic Component for Canari

This module defines the `Periodic` component, which models a cyclic behavior using
a second-order harmonic state-space model. It inherits from `BaseComponent` and sets
up sinusoidal transitions with a fixed period.
"""

from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent


class Periodic(BaseComponent):
    """
    Periodic component for modeling cyclic behavior with a fixed period.

    The periodic component represents a two-dimensional latent process with fixed
    frequency determined by the `period`. This is useful for capturing seasonality
    or other repeating phenomena.

    Args:
        std_error (Optional[float]): Process noise standard deviation. Default is 0.0.
        period (float): Length of one full cycle of the periodic component.
        mu_states (Optional[np.ndarray]): Initial mean values.
        var_states (Optional[np.ndarray]): Initial covariance matrix.

    Raises:
        ValueError: If the shape of `mu_states` or `var_states` does not match the number of states.

    Examples:
        >>> p = Periodic(std_error=0.1, period=52)
        >>> p.states_name
        ['periodic 1', 'periodic 2']
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        period: float = 0.0,
        mu_states: Optional[np.ndarray] = None,
        var_states: Optional[np.ndarray] = None,
    ):
        self.std_error = std_error
        self.period = period
        self._mu_states = mu_states
        self._var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "periodic"

    def initialize_num_states(self):
        self._num_states = 2

    def initialize_states_name(self):
        self._states_name = ["periodic 1", "periodic 2"]

    def initialize_transition_matrix(self):
        """
        Initializes the transition matrix to represent circular rotation dynamics.

        This corresponds to:
            [[cos(w), sin(w)],
             [-sin(w), cos(w)]]
        where w = 2π / period
        """
        w = 2 * np.pi / self.period
        self._transition_matrix = np.array(
            [[np.cos(w), np.sin(w)], [-np.sin(w), np.cos(w)]]
        )

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1, 0]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = self.std_error**2 * np.eye(2)

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self._num_states, 1))
        elif len(self._mu_states) == self._num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
        else:
            raise ValueError(
                "Incorrect mu_states dimension for the periodic component."
            )

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self._num_states, 1))
        elif len(self._var_states) == self._num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(
                "Incorrect var_states dimension for the periodic component."
            )
