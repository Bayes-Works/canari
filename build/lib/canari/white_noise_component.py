import numpy as np
from typing import Optional
from canari.base_component import BaseComponent


class WhiteNoise(BaseComponent):
    """
    Autoregression component
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
        self.mu_states = np.zeros((self.num_states, 1))

    def initialize_var_states(self):
        self.var_states = np.zeros((self.num_states, 1))
