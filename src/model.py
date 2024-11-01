import numpy as np
import copy
from typing import Optional, List, Tuple, Dict
from collections import deque
from pytagi.nn import Sequential
from src.base_component import BaseComponent
import src.common as common
from src.data_struct import LstmOutputHistory, SmootherStates


class Model:
    """
    LSTM/SSM model
    """

    def __init__(
        self,
        *components: BaseComponent,
    ):
        self.components = list(components)
        self.define_model()
        self.lstm_net = None
        self.lstm_states_index = None
        self.smoother_states = None

    @staticmethod
    def prepare_lstm_input(
        lstm_output_history: LstmOutputHistory, input_covariates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input for lstm network, concatenate lstm output history and the input covariates
        """
        mu_lstm_input = np.concatenate((lstm_output_history.mu, input_covariates))
        var_lstm_input = np.concatenate(
            (
                lstm_output_history.var,
                np.zeros(len(input_covariates), dtype=np.float32),
            )
        )
        return mu_lstm_input, var_lstm_input

    def define_model(self):
        """
        Assemble components
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
        self.transition_matrix = common.create_block_diag(*transition_matrices)

        # Global process_noise_matrix
        process_noise_matrices = [
            component.process_noise_matrix for component in self.components
        ]
        self.process_noise_matrix = common.create_block_diag(*process_noise_matrices)

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

    def auto_initialize_baseline_states(self, y: np.ndarray):
        """
        Automatically initialize baseline states from data
        """
        t = np.arange(len(y))
        y = y.flatten()
        t_no_nan = t[~np.isnan(y)]
        y_no_nan = y[~np.isnan(y)]
        coefficients = np.polyfit(t_no_nan, y_no_nan, 2)
        for i, _state_name in enumerate(self.states_name):
            if _state_name == "local level":
                self.mu_states[i] = coefficients[2]
                self.var_states[i, i] = 1
            elif _state_name == "local trend":
                self.mu_states[i] = coefficients[1]
                self.var_states[i, i] = (0.2 * abs(coefficients[1])) ** 2
            elif _state_name == "local acceleration":
                self.mu_states[i] = 2 * coefficients[0]
                self.var_states[i, i] = (0.2 * abs(coefficients[1])) ** 2

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
            self.smoother_states.mu_smooth[time_step - 1],
            self.smoother_states.var_smooth[time_step - 1],
        ) = common.rts_smoother(
            self.smoother_states.mu_prior[time_step],
            self.smoother_states.var_prior[time_step],
            self.smoother_states.mu_smooth[time_step],
            self.smoother_states.var_smooth[time_step],
            self.smoother_states.mu_posterior[time_step - 1],
            self.smoother_states.var_posterior[time_step - 1],
            self.smoother_states.cov_states[time_step],
        )

    def estimate_posterior_states(
        self,
        delta_mu_states: np.ndarray,
        delta_var_states: np.ndarray,
    ):
        """
        Estimate the posterirors for the states
        """

        self._mu_states_posterior = self._mu_states_prior + delta_mu_states
        self._var_states_posterior = self._var_states_prior + delta_var_states

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
        update lstm network's parameters
        """

        self.lstm_net.input_delta_z_buffer.delta_mu = delta_mu_lstm
        self.lstm_net.input_delta_z_buffer.delta_var = delta_var_lstm
        self.lstm_net.backward()
        self.lstm_net.step()

    def initialize_states_with_smoother_estimates(self):
        """
        Set the model initial hidden states = the smoothed estimates
        """

        self.mu_states = self.smoother_states.mu_smooth[0]
        self.var_states = np.diag(np.diag(self.smoother_states.var_smooth[0]))

    def reset_lstm_output_history(self):
        self.lstm_output_history = LstmOutputHistory()
        self.lstm_output_history.initialize(self._lstm_look_back_len)

    def initialize_smoother_states(self, num_time_steps: int):
        self.smoother_states = SmootherStates()
        self.smoother_states.initialize(num_time_steps + 1, self.num_states)
        self.smoother_states.mu_prior[0] = self.mu_states.flatten()
        self.smoother_states.var_prior[0] = self.var_states
        self.smoother_states.mu_posterior[0] = self.mu_states.flatten()
        self.smoother_states.var_posterior[0] = self.var_states

    def save_for_smoother(self, time_step):
        """ "
        Save states' priors, posteriors and cross-covariances for smoother
        """

        self.smoother_states.mu_prior[time_step] = self._mu_states_prior.flatten()
        self.smoother_states.var_prior[time_step] = (
            self._var_states_prior + self._var_states_prior.T
        ) * 0.5
        self.smoother_states.mu_posterior[time_step] = (
            self._mu_states_posterior.flatten()
        )
        self.smoother_states.var_posterior[time_step] = self._var_states_posterior
        self.smoother_states.cov_states[time_step] = (
            self.var_states @ self.transition_matrix.T
        )

    def initialize_smoother_buffers(self):
        """
        Set the smoothed estimates at the last time step = posterior
        """
        self.smoother_states.mu_smooth = np.zeros_like(
            self.smoother_states.mu_posterior
        )
        self.smoother_states.var_smooth = np.zeros_like(
            self.smoother_states.var_posterior
        )
        self.smoother_states.mu_smooth[-1] = self.smoother_states.mu_posterior[-1]
        self.smoother_states.var_smooth[-1] = self.smoother_states.var_posterior[-1]

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
            self._lstm_look_back_len = lstm_component.look_back_len
            self.lstm_states_index = self.states_name.index("lstm")
            self.lstm_net.update_param = self.update_lstm_param
            self.reset_lstm_output_history()
        else:
            raise ValueError(f"No LstmNetwork component found")

    def forecast(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast for whole time series data
        """

        mu_obs_preds = []
        var_obs_preds = []
        mu_lstm_pred = None
        var_lstm_pred = None

        for x in data["x"]:
            if self.lstm_net:
                mu_lstm_input, var_lstm_input = self.prepare_lstm_input(
                    self.lstm_output_history, x
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
        data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter for whole time series data
        """

        mu_obs_preds = []
        var_obs_preds = []
        mu_lstm_pred = None
        var_lstm_pred = None

        for time_step, (x, y) in enumerate(zip(data["x"], data["y"])):
            if self.lstm_net:
                mu_lstm_input, var_lstm_input = self.prepare_lstm_input(
                    self.lstm_output_history, x
                )
                mu_lstm_pred, var_lstm_pred = self.lstm_net.forward(
                    mu_x=mu_lstm_input, var_x=var_lstm_input
                )

            mu_obs_pred, var_obs_pred = self.forward(mu_lstm_pred, var_lstm_pred)
            delta_mu_states, delta_var_states = self.backward(
                y, mu_obs_pred, var_obs_pred
            )
            self.estimate_posterior_states(delta_mu_states, delta_var_states)

            if self.lstm_net:
                delta_mu_lstm = delta_mu_states[self.lstm_states_index] / var_lstm_pred
                delta_var_lstm = (
                    delta_var_states[self.lstm_states_index, self.lstm_states_index]
                    / var_lstm_pred**2
                )
                self.lstm_net.update_param(delta_mu_lstm, delta_var_lstm)
                self.update_lstm_output_history(mu_lstm_pred, var_lstm_pred)

            if self.smoother_states:
                self.save_for_smoother(time_step + 1)

            self.mu_states = self._mu_states_posterior
            self.var_states = self._var_states_posterior
            mu_obs_preds.append(mu_obs_pred)
            var_obs_preds.append(var_obs_pred)
        return np.array(mu_obs_preds).flatten(), np.array(var_obs_preds).flatten()

    def smoother(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smoother for whole time series
        """

        num_time_steps = len(data["y"])
        self.initialize_smoother_states(num_time_steps)

        # Filter
        mu_obs_preds, var_obs_preds = self.filter(data)

        # Smoother
        self.initialize_smoother_buffers()
        for time_step in reversed(range(1, num_time_steps + 1)):
            self.rts_smoother(time_step)

        return np.array(mu_obs_preds).flatten(), np.array(var_obs_preds).flatten()

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, SmootherStates]:
        """
        Train LstmNetwork
        """

        if self.lstm_net is None:
            self.initialize_lstm_network()

        self.smoother(train_data)
        mu_validation_preds, var_validation_preds = self.forecast(validation_data)

        self.reset_lstm_output_history()
        self.initialize_states_with_smoother_estimates()

        return (
            np.array(mu_validation_preds).flatten(),
            np.array(var_validation_preds).flatten(),
            self.smoother_states,
        )
