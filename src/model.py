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
        *components: BaseComponent,
    ):
        super().__init__()
        self.components = components
        self.define_model()
        for component in components:
            if component.component_name == "lstm":
                self._net = Sequential(*component.layers)

    def define_model(self):
        """
        Define_model
        """

        self.assemble_matrices()
        self.assemble_states()

    def search_parameters(self):
        """Grid-search model's parameters"""
        pass

    def save(self, filename: str):
        """Save model"""
        pass

    def load(self, filename: str):
        """Load model"""
        pass

    def assemble_matrices(self):
        """
        Assemble_matrices
        """

        # Global transition matrix
        transition_matrices = [
            component.transition_matrix for component in self.components
        ]
        self._transition_matrix = block_diag(*transition_matrices)

        # Global process_noise_matrix
        process_noise_matrices = [
            component.process_noise_matrix for component in self.components
        ]
        self._process_noise_matrix = block_diag(*process_noise_matrices)

        # Glolabl observation noise matrix
        global_observation_matrix = np.array([])
        for component in self.components:
            global_observation_matrix = np.concatenate(
                (global_observation_matrix, component.observation_matrix[0, :]), axis=0
            )
        self._observation_matrix = np.atleast_2d(global_observation_matrix)

    def assemble_states(self):
        """
        Assemble_states
        """

        self._mu_states = np.vstack(
            [component.mu_states for component in self.components]
        )
        self._var_states = np.vstack(
            [component.var_states for component in self.components]
        )
        self._component_name = ", ".join(
            [component.component_name for component in self.components]
        )
        self._states_name = [
            state for component in self.components for state in component.states_name
        ]
        self._num_states = sum(component.num_states for component in self.components)
