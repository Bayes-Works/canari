import numpy as np
from typing import Optional, List, Tuple
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

        self._save_for_smoother = None

        self.components = list(components)
        self.define_model()

    def define_model(self):
        """
        Define_model
        """

        self.assemble_matrices()
        self.observation_noise_matrix = np.array([[0]])
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
        for obs in time_series_data:
            mu_obs_pred, var_obs_pred = self.forward()
            delta_mu_states, delta_var_states = self.backward(
                obs, mu_obs_pred, var_obs_pred
            )
            self.update_states(delta_mu_states, delta_var_states)
            mu_obs_preds.append(mu_obs_pred)
            var_obs_pred.append(var_obs_pred)

        return mu_obs_preds, var_obs_preds

    def forecast(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast for whole time series
        """

        mu_obs_preds = []
        var_obs_preds = []
        for _ in range(horizon):
            mu_obs_pred, var_obs_pred = self.forward()
            mu_obs_preds.append(mu_obs_pred)
            var_obs_pred.append(var_obs_pred)

        return mu_obs_preds, var_obs_preds

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

        for i in range(time_series_data - 1, 0, -1):
            self._mu_states_smooths[i - 1], self._var_states_smooths[i - 1] = (
                rts_smoother(
                    self._mu_states_priors[i],
                    self._var_states_priors[i],
                    self._mu_states_smooths[i],
                    self._var_states_smooths[i],
                    self._mu_states_posteriors[i - 1],
                    self._var_states_posteriors[i - 1],
                    self._cov_states[i - 1],
                )
            )

        self.mu_states = self._mu_states_smooths[0]
        self.var_states = self._var_states_smooths[0]

        return mu_obs_preds, var_obs_preds

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
        self.component_name = ", ".join(
            [component.component_name for component in self.components]
        )
        self.states_name = [
            state for component in self.components for state in component.states_name
        ]
        self.num_states = sum(component.num_states for component in self.components)

    def forward(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step in states-space model for one sample
        """

        mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = forward(
            self.mu_states,
            self.var_states,
            self.transition_matrix,
            self.process_noise_matrix,
            self.observation_matrix,
            self.observation_noise_matrix,
        )
        if self._save_for_smoother is not None:
            self._mu_states_priors = self._mu_states_priors.append(mu_states_prior)
            self._var_states_priors = self._mu_states_priors.append(var_states_prior)
            self._cov_states = self.transition_matrix @ self.var_states
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
        save_for_smoother: Optional[bool] = False,
    ) -> None:
        mu_states_posterior = self.mu_states + delta_mu_states
        var_states_posterior = self.var_states + delta_var_states
        self.mu_states = mu_states_posterior
        self.var_states = var_states_posterior
        if self._save_for_smoother is not None:
            self._mu_states_posteriors = self._mu_states_posteriors.append(
                mu_states_posterior
            )
            self._var_states_posteriors = self._mu_states_posteriors.append(
                var_states_posterior
            )
