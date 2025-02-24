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
        self._num_states = 1
        if self.phi is None:
            self._num_states += 1
        if self.std_error is None:
            self._num_states += 3

    def initialize_states_name(self):
        self._states_name = ["autoregression"]
        if self.phi is None:
            self._states_name.append("phi")
        if self.std_error is None:
            self._states_name.append("AR_error")
            self._states_name.append("W2")
            self._states_name.append("W2bar")

    def initialize_transition_matrix(self):
        if self.phi is None:
            self._transition_matrix = np.array([[1, 0], [0, 1]])
        else:
            self._transition_matrix = np.array([[self.phi]])
        if self.std_error is None:
            self._transition_matrix = np.block(
                [[self._transition_matrix, np.zeros((self.num_states - 3, 3))],
                 [np.zeros((3, self.num_states))]]
            )

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1]])
        if self.phi is None:
            self._observation_matrix = np.hstack((self._observation_matrix, np.zeros((1, 1))))
        if self.std_error is None:
            # Add zero at the end of the observation matrix
            self._observation_matrix = np.hstack((self._observation_matrix, np.zeros((1, 3))))

    def initialize_process_noise_matrix(self):
        if self.std_error is not None:
            self._process_noise_matrix = np.array([[self.std_error**2]])
        else:
            self._process_noise_matrix = np.array([[self.mu_states[-1]]]) 
        if self.phi is None:
            self._process_noise_matrix = np.block(
                [[self._process_noise_matrix, np.zeros((1, 1))],
                 [np.zeros((1, 2))]]
            )
        if self.std_error is None:
            self._process_noise_matrix = np.block(
                [[self._process_noise_matrix, np.zeros((self.num_states - 3, 3))],
                 [np.zeros((3, self.num_states))]]
            )

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
