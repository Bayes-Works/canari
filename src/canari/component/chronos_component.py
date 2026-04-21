from typing import Optional
import numpy as np
import pytagi
from pytagi.nn import Sequential, LSTM, Linear, SLSTM, SLinear
from canari.component.base_component import BaseComponent


class Chronos(BaseComponent):
    """
    `Chronos` class, inheriting from Canari's `BaseComponent`.

    Args:
        look_back_len (Optional[int]): Number of past LSTM's outputs used as input features.
            Defaults to 1.
    """

    def __init__(
        self,
        look_back_len: Optional[int] = 1,
    ):
        self.look_back_len = look_back_len
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "chronos"

    def initialize_num_states(self):
        self._num_states = 1

    def initialize_states_name(self):
        self._states_name = ["chronos"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[0]])

    def initialize_observation_matrix(self):
        self._observation_matrix = np.array([[1]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = np.array([[0]])

    def initialize_mu_states(self):
        if self._mu_states is None:
            self._mu_states = np.zeros((self.num_states, 1))
        elif len(self._mu_states) == self.num_states:
            self._mu_states = np.atleast_2d(self._mu_states).T
        else:
            raise ValueError(f"Incorrect mu_states dimension for the chronos component.")

    def initialize_var_states(self):
        if self._var_states is None:
            self._var_states = np.zeros((self.num_states, 1))
        elif len(self._var_states) == self.num_states:
            self._var_states = np.atleast_2d(self._var_states).T
        else:
            raise ValueError(f"Incorrect var_states dimension for the chronos component.")
