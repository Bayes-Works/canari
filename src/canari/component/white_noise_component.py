"""
White Noise Component for Canari

This module defines the `WhiteNoise` component, a special case of a state-space model
with no dynamics. It captures uncorrelated, zero-mean Gaussian errors as a latent process.
"""

from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent


class WhiteNoise(BaseComponent):
    """
    White noise component for modeling i.i.d. Gaussian process errors.

    This component represents a latent variable that has no memory or autoregression.

    Args:
        std_error (Optional[float]): Standard deviation of the white noise process. Default is 0.0.

    Examples:
        >>> wn = WhiteNoise(std_error=0.5)
        >>> wn.transition_matrix
        array([[0]])
        >>> wn.process_noise_matrix
        array([[0.25]])
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
    ):
        self.std_error = std_error
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "white noise"

    def initialize_num_states(self):
        self._num_states = 1

    def initialize_states_name(self):
        self._states_name = ["white noise"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[0]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = np.array([[self.std_error**2]])

    def initialize_mu_states(self):
        self._mu_states = np.zeros((self._num_states, 1))

    def initialize_var_states(self):
        self._var_states = np.zeros((self._num_states, 1))
