from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent


class Periodic(BaseComponent):
    """
    `Periodic` class, inheriting from Canari's `BaseComponent`.
    It models a cyclic behavior with a fixed period using a Fourrier-form harmonic.
    It has two hidden states.

    Args:
        period (float): Length of one full cycle of the periodic component
          (number of time steps).
        std_error (Optional[float]): Standard deviation of the process noise. Defaults to 0.0.
        mu_states (Optional[list[float]]): Initial mean of the hidden state. Defaults:
            initialized to zeros.
        var_states (Optional[list[float]]): Initial variance of the hidden state. Defaults:
            initialized to zeros.

    Examples:
        >>> from canari.component import Periodic
        >>> # With known parameters and default mu_states and var_states
        >>> periodic = Periodic(std_error=0.1, period=52)
        >>> # With known mu_states and var_states
        >>> periodic = Periodic(mu_states=[0., 0.], var_states=[1., 1.], std_error=0.1, period=52)
        >>> periodic.states_name
        ['periodic 1', 'periodic 2']
        >>> periodic.mu_states
        >>> periodic.var_states
        >>> periodic.transition_matrix
        >>> periodic.observation_matrix
        >>> periodic.process_noise_matrix
    """

    def __init__(
        self,
        period: float,
        std_error: Optional[float] = 0.0,
        mu_states: Optional[list[float]] = None,
        var_states: Optional[list[float]] = None,
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
        Transition matrix:
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
