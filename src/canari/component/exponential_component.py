"""
This module defines one component "exponential".
"""

from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent


class Exponential(BaseComponent):
    """
    `Exponential` class, inheriting from Canari's `BaseComponent`.
    It models exponential growth with a locally constant speed over time (linear level), which simulates the abscissa scale,
    and a constant amplitude, which simulates the ordinate scale.

    Args:
        std_error (Optional[float]): Standard deviation of the process noise. Defaults: initialized to 0.
        mu_states (Optional[list[float]]): Initial mean of the hidden state. Defaults:
            initialized to zeros.
        var_states (Optional[list[float]]): Initial variance of the hidden state. Defaults:
            initialized to zeros.

    Examples:
        >>> from canari.component import Exponential
        >>> # With known mu_states and var_states
        >>> exponential = Exponential(mu_states=[0, 0.15, 10, 0, 0], var_states=[0.04, 0.01, 1, 0, 0], std_error=0.3)
        >>> # With default mu_states and var_states
        >>> exponential = Exponential(std_error=0.2)
        >>> exponential.component_name
        'exp'
        >>> exponential.states_name
        ['latent level', 'latent trend', 'exp scale factor', 'exp', 'scaled exp']
        >>> exponential.transition_matrix
        array([ [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],)
        >>> exponential.observation_matrix
        array([[0, 0, 0, 0, 1]])
        >>> exponential.process_noise_matrix
        >>> exponential.mu_states
        >>> exponential.var_states
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
        self._component_name = "exp"

    def initialize_num_states(self):
        self._num_states = 5

    def initialize_states_name(self):
        self._states_name = [
            "latent level",  # latent level
            "latent trend",  # latent trend
            "exp scale factor",  # exp scale factor
            "exp",  # exp
            "scaled exp",  # scaled exp
        ]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[0, 0, 0, 0, 1]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = self.std_error**2 * np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self._num_states, 1))
        elif len(self._mu_states) == self._num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
            self._mu_states[0] = -self._mu_states[1]
        else:
            raise ValueError(
                "Incorrect mu_states dimension for the exponential component."
            )

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self._num_states, 1))
        elif len(self._var_states) == self._num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(
                "Incorrect var_states dimension for the exponential component."
            )
