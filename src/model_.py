import numpy as np
from typing import Optional, List, Tuple
from collections import deque
from pytagi.nn import Sequential
from src.base_component import BaseComponent
from src.common import block_diag, forward, backward, rts_smoother


class Model:
    """
    LSTM/SSM model
    """

    def __init__(
        self,
        *components: BaseComponent,
    ):
        self._mu_states_priors = []
        self._var_states_priors = []
        self._mu_states_posteriors = []
        self._var_states_posteriors = []
        self._mu_states_smooths = []
        self._var_states_smooths = []
        self._cov_states = []
        self._save_for_smoother = False

        self._net = None
        self._lstm_indice = None
        self._use_lstm_prediction = False
        self._mu_lstm_input = None
        self._var_lstm_input = None
        self._update_lstm_param = None

        self.components = list(components)
        self.define_model()

    def define_model(self):
        """
        Define_model
        """

        self.assemble_matrices()
        self.assemble_states()

    def filter(
        self,
        time_series_data,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter for whole time series data
        """

        mu_obs_preds = []
        var_obs_preds = []
        mu_lstm_pred = None
        var_lstm_pred = None

        for obs in time_series_data:
            if self._use_lstm_prediction:
                mu_lstm_input = np.concatenate((self._mu_lstm_input, obs[1:]))
                var_lstm_input = np.concatenate(
                    (
                        self._var_lstm_input,
                        np.zeros(self._num_features - 1, dtype=np.float32),
                    )
                )
                mu_lstm_pred, var_lstm_pred = self._net(
                    mu_x=mu_lstm_input, var_x=var_lstm_input
                )

            mu_obs_pred, var_obs_pred = self.forward(mu_lstm_pred, var_lstm_pred)
            delta_mu_states, delta_var_states = self.backward(
                obs[0], mu_obs_pred, var_obs_pred
            )

            self.update_states(delta_mu_states, delta_var_states)
            if self._update_lstm_param:
                self.update_lstm_param(delta_mu_states, delta_var_states, var_lstm_pred)

            if self._use_lstm_prediction:
                self._mu_lstm_input[:-1] = self._mu_lstm_input[1:]
                self._mu_lstm_input[-1] = mu_lstm_pred
                self._var_lstm_input[:-1] = self._var_lstm_input[1:]
                self._var_lstm_input[-1] = var_lstm_pred

                # self._mu_lstm_input.popleft()
                # self._mu_lstm_input.append(mu_lstm_pred)
                # self._var_lstm_input.popleft()
                # self._var_lstm_input.append(var_lstm_pred)

            mu_obs_preds.append(mu_obs_pred)
            var_obs_preds.append(var_obs_pred)

        return np.array(mu_obs_preds).flatten(), np.array(var_obs_preds).flatten()

    def forecast(self, time_series_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast for whole time series
        """

        self._save_for_smoother = False
        mu_obs_preds = []
        var_obs_preds = []
        mu_lstm_pred = None
        var_lstm_pred = None

        for obs in time_series_data:
            if self._use_lstm_prediction:
                mu_lstm_input = np.concatenate((self._mu_lstm_input, obs[1:]))
                var_lstm_input = np.concatenate(
                    (
                        self._var_lstm_input,
                        np.zeros(self._num_features - 1, dtype=np.float32),
                    )
                )
                mu_lstm_pred, var_lstm_pred = self._net(
                    mu_x=mu_lstm_input, var_x=var_lstm_input
                )

            mu_obs_pred, var_obs_pred = self.forward(mu_lstm_pred, var_lstm_pred)

            if self._use_lstm_prediction:
                self._mu_lstm_input[:-1] = self._mu_lstm_input[1:]
                self._mu_lstm_input[-1] = mu_lstm_pred
                self._var_lstm_input[:-1] = self._var_lstm_input[1:]
                self._var_lstm_input[-1] = var_lstm_pred

            mu_obs_preds.append(mu_obs_pred)
            var_obs_preds.append(var_obs_pred)

        return np.array(mu_obs_preds).flatten(), np.array(var_obs_preds).flatten()

    def smoother(self, time_series_data):
        """
        Smoother for whole time series
        """

        self._save_for_smoother = True

        # Filter
        mu_obs_preds, var_obs_preds = self.filter(time_series_data)

        # Smoother
        self._mu_states_smooths = np.zeros_like(self._mu_states_priors)
        self._var_states_smooths = np.zeros_like(self._var_states_priors)
        self._mu_states_smooths[-1] = self._mu_states_posteriors[-1]
        self._var_states_smooths[-1] = self._var_states_posteriors[-1]

        num_time_steps = len(self._mu_states_priors)
        for i in reversed(range(1, num_time_steps)):
            self._mu_states_smooths[i - 1], self._var_states_smooths[i - 1] = (
                rts_smoother(
                    self._mu_states_priors[i],
                    self._var_states_priors[i],
                    self._mu_states_smooths[i],
                    self._var_states_smooths[i],
                    self._mu_states_posteriors[i - 1],
                    self._var_states_posteriors[i - 1],
                    self._cov_states[i],
                )
            )

        self.mu_states = self._mu_states_smooths[0]
        self.var_states = self._var_states_smooths[0]

        return np.array(mu_obs_preds).flatten(), np.array(var_obs_preds).flatten()

    def lstm_train(
        self,
        train_data: np.ndarray,
        validation_data: np.ndarray,
        num_epoch: int,
    ):
        """
        Train LstmNetwork
        """

        train_data = np.float32(train_data)
        validation_data = np.float32(validation_data)
        if self._net is None:
            self.initialize_lstm_network()

        for epoch in range(num_epoch):
            self.smoother(train_data)
            mu_validation_preds, var_validation_preds = self.forecast(validation_data)

        return (
            np.array(mu_validation_preds).flatten(),
            np.array(var_validation_preds).flatten(),
        )

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
        self.transition_matrix = block_diag(*transition_matrices)

        # Global process_noise_matrix
        process_noise_matrices = [
            component.process_noise_matrix for component in self.components
        ]
        self.process_noise_matrix = block_diag(*process_noise_matrices)

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

    def initialize_lstm_network(self):
        """
        Initilize lstm network
        """

        lstm_component = next(
            (
                component
                for component in self.components
                if component.component_name == "lstm"
            ),
            None,
        )
        self._net = Sequential(*lstm_component.layers)
        self._use_lstm_prediction = True
        self._update_lstm_param = True
        self._mu_lstm_input = np.zeros(lstm_component.look_back_len, dtype=np.float32)
        self._var_lstm_input = np.zeros(lstm_component.look_back_len, dtype=np.float32)

        # self._mu_lstm_input = deque(
        #     np.zeros(lstm_component.look_back_len, dtype=np.float32)
        # )
        # self._var_lstm_input = deque(
        #     np.ones(lstm_component.look_back_len, dtype=np.float32)
        # )

        self._lstm_indice = self.states_name.index("lstm")
        self._num_features = lstm_component.num_features

    def forward(
        self,
        mu_lstm_pred: Optional[np.ndarray] = None,
        var_lstm_pred: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step in states-space model for one sample
        """

        mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = forward(
            self.mu_states,
            self.var_states,
            self.transition_matrix,
            self.process_noise_matrix,
            self.observation_matrix,
            mu_lstm_pred,
            var_lstm_pred,
            self._lstm_indice,
        )

        if self._save_for_smoother:
            self._mu_states_priors.append(mu_states_prior)
            self._var_states_priors.append(var_states_prior)
            self._cov_states.append(self.transition_matrix @ self.var_states)

        self.mu_states = mu_states_prior
        self.var_states = var_states_prior

        return mu_obs_pred, var_obs_pred

    def backward(
        self,
        obs: float,
        mu_obs_pred: np.ndarray,
        var_obs_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step in states-space model for one sample
        """

        delta_mu_states, delta_var_states = backward(
            obs,
            mu_obs_pred,
            var_obs_pred,
            self.var_states,
            self.observation_matrix,
        )
        return delta_mu_states, delta_var_states

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
        self.mu_states = mu_states_posterior
        self.var_states = var_states_posterior
        if self._save_for_smoother:
            self._mu_states_posteriors.append(mu_states_posterior)
            self._var_states_posteriors.append(var_states_posterior)

    def update_lstm_param(
        self,
        delta_mu_states: np.ndarray,
        delta_var_states: np.ndarray,
        var_lstm_pred: np.ndarray,
    ):
        """
        update lstm's prameters
        """

        delta_mean_lstm = delta_mu_states[self._lstm_indice] / var_lstm_pred
        delta_var_lstm = (
            delta_var_states[self._lstm_indice, self._lstm_indice] / var_lstm_pred**2
        )

        # # update lstm network
        self._net.input_delta_z_buffer.delta_mu = delta_mean_lstm
        self._net.input_delta_z_buffer.delta_var = delta_var_lstm
        self._net.backward()
        self._net.step()
