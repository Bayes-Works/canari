import numpy as np
from typing import Optional, List
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
        self._lstm = components[1]

    def model_construction(self, *components):
        """model_construction"""
        self.construct_global_matrix(*components)
        self.construct_global_states(*components)
        self.component_name = ", ".join(
            [component.component_name for component in components]
        )
        self.states_names = [
            state for component in components for state in component.states_name
        ]
        self.num_states = sum(component.num_states for component in components)

    def construct_global_matrix(self, *components):
        """construct_global_matrix"""
        transition_matrices = [component.transition_matrix for component in components]
        self.transition_matrix = block_diag(*transition_matrices)
        process_noise_matrices = [
            component.process_noise_matrix for component in components
        ]
        self.process_noise_matrix = block_diag(*process_noise_matrices)
        global_observation_matrix = np.array([])
        for component in components:
            global_observation_matrix = np.concatenate(
                (global_observation_matrix, component.observation_matrix[0, :]), axis=0
            )
        self.observation_matrix = global_observation_matrix

    def construct_global_states(self, *components):
        """construct_global_states"""
        self.mu_states = np.vstack([component.mu_states for component in components])
        self.var_states = np.vstack([component.var_states for component in components])

    def search_parameters(self):
        pass

    def detect_anomaly(self, time_series_data: np.ndarray):
        pass

    def save(self, filename: str):
        pass

    def load(self, filename: str):
        pass
