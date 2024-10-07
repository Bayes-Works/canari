import numpy as np
from typing import Optional, List
from pytagi.nn import Sequential
from base_model import BaseModel
from base_component import BaseComponent

from common import block_diag


class LstmSsm(BaseModel):
    """
    LSTM/SSM model
    """

    def __init__(self, *components: BaseComponent):
        super().__init__()
        self.model_construction(*components)
        for component in components:
            if component.component_name == "Lstm":
                self._lstm = Sequential(*component.layers)

    def model_construction(self, *components):
        """model_construction"""
        self.construct_global_matrix(*components)
        self.construct_global_states(*components)

    def construct_global_matrix(self, *components):
        """construct_global_matrix"""
        transition_matrices = [component.transition_matrix for component in components]
        self._transition_matrix = block_diag(*transition_matrices)
        process_noise_matrices = [
            component.process_noise_matrix for component in components
        ]
        self._process_noise_matrix = block_diag(*process_noise_matrices)
        global_observation_matrix = np.array([])
        for component in components:
            global_observation_matrix = np.concatenate(
                (global_observation_matrix, component.observation_matrix[0, :]), axis=0
            )
        self._observation_matrix = global_observation_matrix

    def construct_global_states(self, *components):
        """construct_global_states"""
        self._mu_states = np.vstack([component.mu_states for component in components])
        self._var_states = np.vstack([component.var_states for component in components])
        self._component_name = ", ".join(
            [component.component_name for component in components]
        )
        self._states_name = [
            state for component in components for state in component.states_name
        ]
        self._num_states = sum(component.num_states for component in components)

    def search_parameters(self):
        pass

    def detect_anomaly(self, time_series_data: np.ndarray):
        pass

    def save(self, filename: str):
        pass

    def load(self, filename: str):
        pass
