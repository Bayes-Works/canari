import numpy as np
from typing import Optional, List
from pytagi.nn import Sequential
from base_component import BaseComponent

from common import block_diag


class Model:
    """
    LSTM/SSM model
    """

    def __init__(
        self,
        components: List[BaseComponent],
        std_observation_error: Optional[float] = 0.0,
    ):
        super().__init__()
        self.define_model(components, std_observation_error)
        for component in components:
            if component.component_name == "lstm":
                self._net = Sequential(*component.layers)

    def define_model(self, components, std_observation_error):
        """
        Define_model
        """

        self.assemble_matrices(components, std_observation_error)
        self.assemble_states(components)

    def search_parameters(self):
        """Grid-search model's parameters"""
        pass

    def save(self, filename: str):
        """Save model"""
        pass

    def load(self, filename: str):
        """Load model"""
        pass

    def assemble_matrices(self, components, std_observation_error):
        """
        Assemble_matrices
        """

        # Global transition matrix
        transition_matrices = [component.transition_matrix for component in components]
        self._transition_matrix = block_diag(*transition_matrices)

        # Global process_noise_matrix
        process_noise_matrices = [
            component.process_noise_matrix for component in components
        ]
        self._process_noise_matrix = block_diag(*process_noise_matrices)

        # Glolabl observation noise matrix
        global_observation_matrix = np.array([])
        for component in components:
            global_observation_matrix = np.concatenate(
                (global_observation_matrix, component.observation_matrix[0, :]), axis=0
            )
        self._observation_matrix = global_observation_matrix
        self._observation_noise_matrix = np.array([[std_observation_error**2]])

    def assemble_states(self, components):
        """
        Assemble_states
        """

        self._mu_states = np.vstack([component.mu_states for component in components])
        self._var_states = np.vstack([component.var_states for component in components])
        self._component_name = ", ".join(
            [component.component_name for component in components]
        )
        self._states_name = [
            state for component in components for state in component.states_name
        ]
        self._num_states = sum(component.num_states for component in components)
