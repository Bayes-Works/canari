"""
Bounded Autoregression Component for Canari

This module defines a `BoundedAutoregression` class, inheriting from Canari's `BaseComponent`.
It represents residuals following a univariate AR(1) process with optional constains defined by 
a coefficient (gamma), which scales the standard deviation of the stationary AR.
"""

from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent


class BoundedAutoregression(BaseComponent):
    """
    Bounded autoregression component for modeling residual following a univariate AR(1) process.

    Parameters:
        std_error ([float]): Known standard deviation of the process noise.
        phi ([float]): Known autoregressive coefficient.
        gamma (Optional[float]): Coefficient to scale the standard deviation of the stationary AR. If none, no constraint is applied.
        mu_states (Optional[np.ndarray]): Initial mean values.
        var_states (Optional[np.ndarray]): Initial covariance matrix.

    Behavior:
        - Adds 1 extra states, X^{BAR} if gamma is provided. Otherwise no constraint is applied.
        - Initializes transition, observation, and noise matrices accordingly.

    Raises:
        ValueError: If dimensions of `mu_states` or `var_states` do not match expected number of states.

    Examples:
        >>> from my_module.autoregression import Autoregression
        >>> bar = BoundedAutoregression(std_error=1, phi=0.75, gamma=0.5)
        >>> print(ar.mu_states)
        >>> print(ar.var_states)
        >>> print(ar.states_name)
        ['autoregression', 'bounded_autoregression']
        >>> bar = BoundedAutoregression(std_error=1, phi=0.75)
    """

    def __init__(
        self,
        std_error: float,
        phi: float,
        gamma: Optional[float] = None,
        mu_states: Optional[np.ndarray] = None,
        var_states: Optional[np.ndarray] = None,
        formular: Optional[str] = 'onlineAR_similar',
    ):
        self.std_error = std_error
        self.phi = phi
        self.gamma = gamma
        self._mu_states = mu_states
        self._var_states = var_states
        self.formular = formular
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "bounded autoregression"

    def initialize_num_states(self):
        self._num_states = 2
        if self.gamma is None:
            self._num_states = 1

    def initialize_states_name(self):
        self._states_name = ["autoregression", "bounded autoregression"]
        if self.gamma is None:
            self._states_name = ["autoregression"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[0, self.phi], [0, 0]])
        if self.formular == "BAR_paper":
            self._transition_matrix = np.array([[self.phi, 0], [0, 0]])
        if self.gamma is None:
            self._transition_matrix = np.array([[self.phi]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1, 0]])
        if self.formular == "BAR_paper":
            self._observation_matrix = np.array([[0, 1]])
        if self.gamma is None:
            self._observation_matrix = np.array([[1]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = np.array([[self.std_error**2, 0], [0, 0]])
        if self.gamma is None:
            self._process_noise_matrix = np.array([[self.std_error**2]])

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self._num_states, 1))
        elif len(self._mu_states) == self._num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
        else:
            raise ValueError(
                "Incorrect mu_states dimension for the autoregression component."
            )

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self._num_states, 1))
        elif len(self._var_states) == self._num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(
                "Incorrect var_states dimension for the autoregression component."
            )
