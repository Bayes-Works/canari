"""
This module defines one component "exponential".
"""

from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent


class Exponential(BaseComponent):
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
            "exp level",
            "exp trend",
            "exp amplitude",
            "exp",
            "exp with amplitude",
        ]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
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
