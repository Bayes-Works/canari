from typing import Optional
import numpy as np
from canari.component.base_component import BaseComponent


class Intervention(BaseComponent):
    """
    `Intervention` class, inheriting from Canari's `BaseComponent`.
    It is used to add a hidden state for intervention.

    Args:
        interv_state_index (int): Index of hidden state make intervention.
        contribute_to_obs (bool, optinal): the intervention hidden state contribute
                                            to the observation or not. Defaults to False.

    Examples:
        >>> from canari.component import Intervention
        >>> intervention = Intervention(interv_state_index=1)
        >>> intervention.transition_matrix
        array([[1]])
        >>> intervention.process_noise_matrix
        array([[0]])
    """

    def __init__(
        self,
        interv_state_index: int,
        contribute_to_obs: Optional[bool] = False
    ):
        self.interv_state_index = interv_state_index
        self.contribute_to_obs = contribute_to_obs
        super().__init__()

    def initialize_component_name(self):
        self._component_name = "intervention"

    def initialize_num_states(self):
        self._num_states = 1

    def initialize_states_name(self):
        self._states_name = ["intervention"]

    def initialize_transition_matrix(self):
        self._transition_matrix = np.array([[1]])

    def initialize_observation_matrix(self):
        if self.contribute_to_obs:
            self._observation_matrix = np.array([[1]])
        else:
            self._observation_matrix = np.array([[0]])

    def initialize_process_noise_matrix(self):
        self._process_noise_matrix = np.array([[0]])

    def initialize_mu_states(self):
        self._mu_states = np.zeros((self._num_states, 1))

    def initialize_var_states(self):
        self._var_states = np.zeros((self._num_states, 1))
