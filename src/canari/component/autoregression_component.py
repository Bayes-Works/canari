"""
Autoregression Component for Canari

This module defines an `Autoregression` class, inheriting from Canari's `BaseComponent`.
It represents residuals following a univariate AR(1) process with optional learning of
the autoregressive coefficient (`phi`) and process noise standard deviation (`std_error`).
"""

from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent


class Autoregression(BaseComponent):
    """
    Autoregression component for modeling residual following a univariate AR(1) process.

    Parameters:
        std_error (Optional[float]): Known standard deviation of the process noise. If None, it will be learned using AGVI.
        phi (Optional[float]): Known autoregressive coefficient. If None, it will be learned using Gaussian Multiplicative Approximation (GMA).
        mu_states (Optional[np.ndarray]): Initial mean values.
        var_states (Optional[np.ndarray]): Initial covariance matrix.

    Behavior:
        - Adds 2 extra states if `phi` is None.
        - Adds 3 extra states if `std_error` is None.
        - Initializes transition, observation, and noise matrices accordingly.

    Raises:
        ValueError: If dimensions of `mu_states` or `var_states` do not match expected number of states.

    Examples:
        >>> from my_module.autoregression import Autoregression
        >>> ar = Autoregression(std_error=None, phi=None)
        >>> print(ar.mu_states)
        >>> print(ar.var_states)
        >>> print(ar.states_name)
        ['autoregression', 'phi', 'phi_autoregression', 'AR_error', 'W2', 'W2bar']
        >>> ar = Autoregression(phi=0.9, std_error=0.1)
        >>> ar = Autoregression()
    """

    def __init__(
        self,
        std_error: Optional[float] = None,
        phi: Optional[float] = None,
        mu_states: Optional[np.ndarray] = None,
        var_states: Optional[np.ndarray] = None,
    ):
        self.std_error = (
            std_error  # When std_error is None, use AGVI to learn the process error
        )
        self.phi = phi  # When phi is None, use GMA to learn
        self._mu_states = mu_states
        self._var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "autoregression"

    def initialize_num_states(self):
        self._num_states = 1
        if self.phi is None:
            self._num_states += 2
        if self.std_error is None:
            self._num_states += 3

    def initialize_states_name(self):
        self._states_name = ["autoregression"]
        if self.phi is None:
            self._states_name.append("phi")
            self._states_name.append("phi_autoregression")  # phi^{AR} times X^{AR}
        if self.std_error is None:
            self._states_name.append(
                "AR_error"
            )  # Process error of AR (W variable in AGVI)
            self._states_name.append("W2")  # Square of the process error
            self._states_name.append("W2bar")  # Expected value of W2

    def initialize_transition_matrix(self):
        if self.phi is None:
            self._transition_matrix = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]])
        else:
            self._transition_matrix = np.array([[self.phi]])
        if self.std_error is None:
            self._transition_matrix = np.block(
                [
                    [self._transition_matrix, np.zeros((self._num_states - 3, 3))],
                    [np.zeros((3, self._num_states))],
                ]
            )

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1]])
        if self.phi is None:
            self._observation_matrix = np.hstack(
                (self._observation_matrix, np.zeros((1, 2)))
            )
        if self.std_error is None:
            self._observation_matrix = np.hstack(
                (self._observation_matrix, np.zeros((1, 3)))
            )

    def initialize_process_noise_matrix(self):
        if self.std_error is not None:
            self._process_noise_matrix = np.array([[self.std_error**2]])
        else:
            self._process_noise_matrix = np.array([[self._mu_states[-1]]])
        if self.phi is None:
            self._process_noise_matrix = np.block(
                [[self._process_noise_matrix, np.zeros((1, 2))], [np.zeros((2, 3))]]
            )
        if self.std_error is None:
            self._process_noise_matrix = np.block(
                [
                    [self._process_noise_matrix, np.zeros((self._num_states - 3, 3))],
                    [np.zeros((3, self._num_states))],
                ]
            )

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
