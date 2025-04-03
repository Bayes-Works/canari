from typing import Optional
import numpy as np
from canari.base_component import BaseComponent


class Periodic(BaseComponent):
    """
    Period component
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        period: float = 0.0,
        mu_states: Optional[np.ndarray] = None,
        var_states: Optional[np.ndarray] = None,
    ):
        self.std_error = std_error
        self.period = period
        self.mu_states = mu_states
        self.var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "periodic"

    def initialize_num_states(self):
        self._num_states = 2

    def initialize_states_name(self):
        self._states_name = ["periodic 1", "periodic 2"]

    def initialize_transition_matrix(self):
        w = 2 * np.pi / self.period
        self._transition_matrix = np.array(
            [[np.cos(w), np.sin(w)], [-np.sin(w), np.cos(w)]]
        )

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1, 0]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = self.std_error**2 * np.array([[1, 0], [0, 1]])

    def initialize_mu_states(self):
        if self.mu_states is None:
            self.mu_states = np.zeros((self.num_states, 1))
        elif len(self.mu_states) == self.num_states:
            self.mu_states = np.atleast_2d(self.mu_states).T
        else:
            raise ValueError(
                f"Incorrect mu_states dimension for the periodic component."
            )

    def initialize_var_states(self):
        if self.var_states is None:
            self.var_states = np.zeros((self.num_states, 1))
        elif len(self.var_states) == self.num_states:
            self.var_states = np.atleast_2d(self.var_states).T
        else:
            raise ValueError(
                f"Incorrect var_states dimension for the periodic component."
            )
