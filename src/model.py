import numpy as np
from typing import Optional, List, Tuple
from collections import deque
from pytagi.nn import Sequential
from src.base_component import BaseComponent
import src.common as common
from src.data_struct import LstmInput, LstmStates, SmootherStates


class Model:
    """
    LSTM/SSM model
    """

    def __init__(
        self,
        *components: BaseComponent,
    ):
        self.components = list(components)
        self.assemble_components()
        self.lstm_net = None
        self.smoother_states = None

    def assemble_components(self):
        """
        Define_model
        """

        self.assemble_matrices()
        self.assemble_states()

    def assemble_matrices(self):
        """
        Assemble_matrices
        """

        # Global transition matrix
        transition_matrices = [
            component.transition_matrix for component in self.components
        ]
        self.transition_matrix = common.block_diag(*transition_matrices)

        # Global process_noise_matrix
        process_noise_matrices = [
            component.process_noise_matrix for component in self.components
        ]
        self.process_noise_matrix = common.block_diag(*process_noise_matrices)

        # Glolabl observation noise matrix
        global_observation_matrix = np.array([])
        for component in self.components:
            global_observation_matrix = np.concatenate(
                (global_observation_matrix, component.observation_matrix[0, :]), axis=0
            )
        self.observation_matrix = np.atleast_2d(global_observation_matrix)

    def assemble_states(self):
        """
        Assemble_states
        """

        self.mu_states = np.vstack(
            [component.mu_states for component in self.components]
        )
        self.var_states = np.vstack(
            [component.var_states for component in self.components]
        )
        self.var_states = np.diagflat(self.var_states)
        self.component_name = ", ".join(
            [component.component_name for component in self.components]
        )
        self.states_name = [
            state for component in self.components for state in component.states_name
        ]
        self.num_states = sum(component.num_states for component in self.components)

    @staticmethod
    def prepare_lstm_input(lstm_input, sample):
        mu_lstm_input = np.concatenate((lstm_input.mu, sample[1:]))
        var_lstm_input = np.concatenate(
            (
                lstm_input.var,
                np.zeros(len(sample) - 1, dtype=np.float32),
            )
        )
        return mu_lstm_input, var_lstm_input

    def forward(
        self,
        mu_lstm_pred: Optional[np.ndarray] = None,
        var_lstm_pred: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        One step prediction in states-space model
        """

        mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = common.forward(
            self.mu_states,
            self.var_states,
            self.transition_matrix,
            self.process_noise_matrix,
            self.observation_matrix,
            mu_lstm_pred,
            var_lstm_pred,
            self.lstm_states_index,
        )
        self._mu_states_prior = mu_states_prior
        self._var_states_prior = var_states_prior
        return mu_obs_pred, var_obs_pred

    def backward(
        self,
        obs: float,
        mu_obs_pred: np.ndarray,
        var_obs_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step in states-space model
        """

        delta_mu_states, delta_var_states = common.backward(
            obs,
            mu_obs_pred,
            var_obs_pred,
            self._var_states_prior,
            self.observation_matrix,
        )
        return delta_mu_states, delta_var_states

    def rts_smoother(self, time_step):
        """
        RTS smoother
        """

        (
            self.smoother_states.mu_smooths[time_step - 1],
            self.smoother_states.var_smooths[time_step - 1],
        ) = common.rts_smoother(
            self.smoother_states.mu_priors[time_step],
            self.smoother_states.var_priors[time_step],
            self.smoother_states.mu_smooths[time_step],
            self.smoother_states.var_smooths[time_step],
            self.smoother_states.mu_posteriors[time_step - 1],
            self.smoother_states.var_posteriors[time_step - 1],
            self.smoother_states.cov_states[time_step],
        )

    def update_states(
        self,
        delta_mu_states: np.ndarray,
        delta_var_states: np.ndarray,
    ) -> None:
        """
        Estimate the posterirors for the states
        """

        mu_states_posterior = self.mu_states + delta_mu_states
        var_states_posterior = self.var_states + delta_var_states
        self._mu_states_posterior = mu_states_posterior
        self._var_states_posterior = var_states_posterior

    def update_lstm_output_history(self, mu_lstm_pred, var_lstm_pred):
        self.lstm_output_history.mu = np.roll(self.lstm_output_history.mu, -1)
        self.lstm_output_history.var = np.roll(self.lstm_output_history.var, -1)
        self.lstm_output_history.mu[-1] = mu_lstm_pred
        self.lstm_output_history.var[-1] = var_lstm_pred

    def update_lstm_param(
        self,
        delta_mu_lstm: np.ndarray,
        delta_var_lstm: np.ndarray,
    ):
        """
        update lstm's prameters
        """

        self.lstm_net.input_delta_z_buffer.delta_mu = delta_mu_lstm
        self.lstm_net.input_delta_z_buffer.delta_var = delta_var_lstm
        self.lstm_net.backward()
        self.lstm_net.step()

    def save_for_smoother(self):
        """ "
        Save variables for smoother
        """

        self.smoother_states.mu_priors.append(self._mu_states_prior)
        self.smoother_states.var_priors.append(
            (self._var_states_prior + self._var_states_prior.T) / 2
        )
        self.smoother_states.mu_posteriors.append(self._mu_states_posterior)
        self.smoother_states.var_posteriors.append(self._var_states_posterior)
        self.smoother_states.cov_states.append(self.transition_matrix @ self.var_states)

    def initialize_lstm_network(self):
        """
        Initialize_lstm_network from lstm_component
        """

        lstm_component = next(
            (
                component
                for component in self.components
                if component.component_name == "lstm"
            ),
            None,
        )
        if lstm_component:
            self.lstm_net = lstm_component.initialize_lstm_network()
            self.lstm_net.update_param = self.update_lstm_param
            self.lstm_states = LstmStates()
            self.lstm_output_history = LstmInput(lstm_component.look_back_len)
            self.lstm_states_index = self.states_name.index("lstm")
            self._lstm_look_back_len = lstm_component.look_back_len
        else:
            raise ValueError(f"No LstmNetwork component found")

    def initialize_states_with_smoother_estimates(self):
        """
        Set the model initial hidden states = the smoothed estimates
        """

        self.mu_states = self.smoother_states.mu_smooths[0]
        self.var_states = self.smoother_states.var_smooths[0]

    def reset_lstm_output_history(self):
        self.lstm_output_history = LstmInput(self._lstm_look_back_len)

    def initialize_smoother_states(self):
        self.smoother_states = SmootherStates()

    def initialize_smoother_buffers(self):
        """
        Set the smoothed estimates at the last time step = posterior
        """
        self.smoother_states.mu_smooths = np.zeros_like(
            self.smoother_states.mu_posteriors
        )
        self.smoother_states.var_smooths = np.zeros_like(
            self.smoother_states.var_posteriors
        )
        self.smoother_states.mu_smooths[-1] = self.smoother_states.mu_posteriors[-1]
        self.smoother_states.var_smooths[-1] = self.smoother_states.var_posteriors[-1]

    def forecast(self, data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast for whole time series data
        """

        mu_obs_preds = []
        var_obs_preds = []
        mu_lstm_pred = None
        var_lstm_pred = None

        for obs in data:
            if self.lstm_net:
                mu_lstm_input, var_lstm_input = self.prepare_lstm_input(
                    self.lstm_output_history, obs
                )
                mu_lstm_pred, var_lstm_pred = self.lstm_net.forward(
                    mu_x=mu_lstm_input, var_x=var_lstm_input
                )

            mu_obs_pred, var_obs_pred = self.forward(mu_lstm_pred, var_lstm_pred)

            if self.lstm_net:
                self.update_lstm_output_history(mu_lstm_pred, var_lstm_pred)

            self.mu_states = self._mu_states_prior
            self.var_states = self._var_states_prior
            mu_obs_preds.append(mu_obs_pred)
            var_obs_preds.append(var_obs_pred)
        return np.array(mu_obs_preds).flatten(), np.array(var_obs_preds).flatten()

    def filter(
        self,
        data,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter for whole time series data
        """

        mu_obs_preds = []
        var_obs_preds = []
        mu_lstm_pred = None
        var_lstm_pred = None

        for obs in data:
            if self.lstm_net:
                mu_lstm_input, var_lstm_input = self.prepare_lstm_input(
                    self.lstm_output_history, obs
                )
                mu_lstm_pred, var_lstm_pred = self.lstm_net.forward(
                    mu_x=mu_lstm_input, var_x=var_lstm_input
                )

            mu_obs_pred, var_obs_pred = self.forward(mu_lstm_pred, var_lstm_pred)
            delta_mu_states, delta_var_states = self.backward(
                obs[0], mu_obs_pred, var_obs_pred
            )
            self.update_states(delta_mu_states, delta_var_states)

            if self.lstm_net:
                delta_mu_lstm = delta_mu_states[self.lstm_states_index] / var_lstm_pred
                delta_var_lstm = (
                    delta_var_states[self.lstm_states_index] / var_lstm_pred**2
                )
                self.lstm_net.update_param(delta_mu_lstm, delta_var_lstm)
                self.update_lstm_output_history(mu_lstm_pred, var_lstm_pred)

            if self.smoother_states:
                self.save_for_smoother()

            self.mu_states = self._mu_states_posterior
            self.var_states = self._var_states_posterior
            mu_obs_preds.append(mu_obs_pred)
            var_obs_preds.append(var_obs_pred)
        return np.array(mu_obs_preds).flatten(), np.array(var_obs_preds).flatten()

    def smoother(self, data):
        """
        Smoother for whole time series
        """

        num_time_steps = len(data)
        self.initialize_smoother_states()

        # Filter
        mu_obs_preds, var_obs_preds = self.filter(data)

        # Smoother
        self.initialize_smoother_buffers()
        for time_step in reversed(range(1, num_time_steps)):
            self.rts_smoother(time_step)

        return np.array(mu_obs_preds).flatten(), np.array(var_obs_preds).flatten()

    def lstm_train(
        self,
        train_data: np.ndarray,
        validation_data: np.ndarray,
    ):
        """
        Train LstmNetwork
        """

        train_data = np.float32(train_data)
        validation_data = np.float32(validation_data)
        if self.lstm_net is None:
            self.initialize_lstm_network()

        self.smoother(train_data)
        mu_validation_preds, var_validation_preds = self.forecast(validation_data)
        self.reset_lstm_output_history()
        self.initialize_states_with_smoother_estimates()

        return (
            np.array(mu_validation_preds).flatten(),
            np.array(var_validation_preds).flatten(),
        )
