import numpy as np
from typing import Optional
from src.base_component import BaseComponent


class Autoregression(BaseComponent):
    """
    Autoregression component
    """

    def __init__(
        self,
        std_error: Optional[float] = None,
        phi: Optional[float] = None,
        mu_states: Optional[np.ndarray] = None,
        var_states: Optional[np.ndarray] = None,
    ):
        self.std_error = std_error  # When std_error is None, use AGVI to learn the process error
        self.phi = phi              # When phi is None, use GMA to learn
        self.mu_states = mu_states
        self.var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "autoregression"

    def initialize_num_states(self):
        if self.phi is None:
            self._num_states = 2
        else:
            self._num_states = 1

    def initialize_states_name(self):
        if self.phi is None:
            self._states_name = ["phi", "autoregression"] # When phi is not provided, it is a hidden state to learn
        else:
            self._states_name = ["autoregression"]

    def initialize_transition_matrix(self):
        if self.phi is None:
            self._transition_matrix = np.array([[1, 0], [0, 1]])
        else:
            self._transition_matrix = np.array([[self.phi]])

    def initialize_observation_matrix(self):
        if self.phi is None:
            self._observation_matrix = np.array([[0, 1]])
        else:
            self._observation_matrix = np.array([[1]])

    def initialize_process_noise_matrix(self):
        if self.phi is None:
            self._process_noise_matrix = self.std_error**2 * np.array(
                [[0, 0], [0, 1]]
            )
        else:
            self._process_noise_matrix = np.array([[self.std_error**2]])

    def initialize_mu_states(self):
        if self.mu_states is None:
            self.mu_states = np.zeros((self.num_states, 1))
        elif len(self.mu_states) == self.num_states:
            self.mu_states = np.atleast_2d(self.mu_states).T
        else:
            raise ValueError(
                f"Incorrect mu_states dimension for the autoregression component."
            )

    def initialize_var_states(self):
        if self.var_states is None:
            self.var_states = np.zeros((self.num_states, 1))
        elif len(self.var_states) == self.num_states:
            self.var_states = np.atleast_2d(self.var_states).T
        else:
            raise ValueError(
                f"Incorrect var_states dimension for the autoregression component."
            )
