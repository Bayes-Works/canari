from typing import Optional
import numpy as np
from canari.base_component import BaseComponent


class LocalLevel(BaseComponent):
    """
    Local level component
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        mu_states: Optional[np.ndarray] = None,
        var_states: Optional[np.ndarray] = None,
    ):
        self.std_error = std_error
        self.mu_states = mu_states
        self.var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "local level"

    def initialize_num_states(self):
        self._num_states = 1

    def initialize_states_name(self):
        self._states_name = ["local level"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[1]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = np.array([[self.std_error**2]])

    def initialize_mu_states(self):
        if self.mu_states is None:
            self.mu_states = np.zeros((self.num_states, 1))
        elif len(self.mu_states) == self.num_states:
            self.mu_states = np.atleast_2d(self.mu_states).T
        else:
            raise ValueError(
                f"Incorrect mu_states dimension for the local level component."
            )

    def initialize_var_states(self):
        if self.var_states is None:
            self.var_states = np.zeros((self.num_states, 1))
        elif len(self.var_states) == self.num_states:
            self.var_states = np.atleast_2d(self.var_states).T
        else:
            raise ValueError(
                f"Incorrect var_states dimension for the local level component."
            )


class LocalTrend(BaseComponent):
    """
    Local level component
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        mu_states: Optional[np.ndarray] = None,
        var_states: Optional[np.ndarray] = None,
    ):
        self.std_error = std_error
        self.mu_states = mu_states
        self.var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "local trend"

    def initialize_num_states(self):
        self._num_states = 2

    def initialize_states_name(self):
        self._states_name = ["local level", "local trend"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[1, 1], [0, 1]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1, 0]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = self.std_error**2 * np.array(
            [[1 / 3, 1 / 2], [1 / 2, 1 / 3]]
        )

    def initialize_mu_states(self):
        if self.mu_states is None:
            self.mu_states = np.zeros((self.num_states, 1))
        elif len(self.mu_states) == self.num_states:
            self.mu_states = np.atleast_2d(self.mu_states).T
        else:
            raise ValueError(
                f"Incorrect mu_states dimension for the local trend component."
            )

    def initialize_var_states(self):
        if self.var_states is None:
            self.var_states = np.zeros((self.num_states, 1))
        elif len(self.var_states) == self.num_states:
            self.var_states = np.atleast_2d(self.var_states).T
        else:
            raise ValueError(
                f"Incorrect var_states dimension for the local trend component."
            )


class LocalAcceleration(BaseComponent):
    """
    Local level component
    """

    def __init__(
        self,
        std_error: Optional[float] = 0.0,
        mu_states: Optional[np.ndarray] = None,
        var_states: Optional[np.ndarray] = None,
    ):
        self.std_error = std_error
        self.mu_states = mu_states
        self.var_states = var_states
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "local acceleration"

    def initialize_num_states(self):
        self._num_states = 3

    def initialize_states_name(self):
        self._states_name = ["local level", "local trend", "local acceleration"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[1, 1, 0.5], [0, 1, 1], [0, 0, 1]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1, 0, 0]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = self.std_error**2 * np.array(
            [[1 / 20, 1 / 8, 1 / 6], [1 / 8, 1 / 3, 0.5], [1 / 6, 0.5, 1]]
        )

    def initialize_mu_states(self):
        if self.mu_states is None:
            self.mu_states = np.zeros((self.num_states, 1))
        elif len(self.mu_states) == self.num_states:
            self.mu_states = np.atleast_2d(self.mu_states).T
        else:
            raise ValueError(
                f"Incorrect mu_states dimension for the local acceleration component."
            )

    def initialize_var_states(self):
        if self.var_states is None:
            self.var_states = np.zeros((self.num_states, 1))
        elif len(self.var_states) == self.num_states:
            self.var_states = np.atleast_2d(self.var_states).T
        else:
            raise ValueError(
                f"Incorrect var_states dimension for the local acceleration component."
            )
