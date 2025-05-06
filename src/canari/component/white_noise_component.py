from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent


class WhiteNoise(BaseComponent):
    """
    `WhiteNoise` class, inheriting from Canari's `BaseComponent`.
    It is used to model zero-mean i.i.d. Gaussian errors.

    Args:
        std_error (Optional[float]): Standard deviation of the error. Default is 0.05.

    Examples:
        >>> from canari.component import WhiteNoise
        >>> white_noise = WhiteNoise(std_error=0.5)
        >>> white_noise.transition_matrix
        array([[0]])
        >>> white_noise.process_noise_matrix
        array([[0.25]])
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.05,
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
