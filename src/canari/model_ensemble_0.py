"""
Emsemble of model
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from canari.model import Model
from canari.data_struct import StatesHistory
import canari.common as common


class ModelEnsemble:
    """
    Ensemble of model for modelling dependencies.
    """

    def __init__(self, main_model: Model, dependent_model: Model):
        self.main_model = main_model
        self.dep_model = dependent_model
        self._initialize_model()
        self.states = StatesHistory()

    def _initialize_model(self):
        """
        Set up the model by assembling matrices, initializing states,
        configuring LSTM and autoregressive modules if included.
        """

        self._assemble_matrices()
        self._assemble_states()

    def _assemble_matrices(self):
        """
        Assemble global matrices:
            - Transition matrix
            - Process noise matrix
            - Observation matrix
        from all components in the model.
        """

        # Assemble transition matrices
        self.transition_matrix = common.create_block_diag(
            self.main_model.transition_matrix, self.dep_model.transition_matrix
        )

        # Assemble process noise matrices
        self.process_noise_matrix = common.create_block_diag(
            self.main_model.process_noise_matrix, self.dep_model.process_noise_matrix
        )

        # Assemble observation matrices
        self.observation_matrix = common.create_block_diag(
            self.main_model.observation_matrix, self.dep_model.observation_matrix
        )

    def _assemble_states(self):
        """
        Concatenate state means and variances from all components.
        """

        self.mu_states = np.vstack(
            (self.main_model.mu_states, self.dep_model.mu_states)
        )
        self.var_states = common.create_block_diag(
            self.main_model.var_states, self.dep_model.var_states
        )
        main_states_names = [f"{name} main" for name in self.main_model.states_name]
        dep_states_names = [f"{name} dep" for name in self.dep_model.states_name]

        self.states_name = main_states_names + dep_states_names
        self.num_states = self.main_model.num_states + self.dep_model.num_states

    def get_states_index(self, states_name: str):
        """
        Retrieve index of a state in the state vector.

        Args:
            states_name (str): The name of the state.

        Returns:
            int or None: Index of the state, or None if not found.

        Examples:
            >>> lstm_index = model.get_states_index("lstm")
            >>> level_index = model.get_states_index("level")
        """

        index = (
            self.states_name.index(states_name)
            if states_name in self.states_name
            else None
        )
        return index

    def initialize_states_history(self):
        """
        Reinitialize prior, posterior, and smoothed values for hidden states in
        :attr:`~canari.model.Model.states` with empty lists.
        """

        self.states.initialize(self.states_name)

    def _save_states_history(self):
        """
        Save current prior, posterior hidden states, and cross-covariaces between hidden states
        at two consecutive time steps for later use in Kalman's smoother.
        """

        self.states.mu_prior.append(self.mu_states_prior)
        self.states.var_prior.append(self.var_states_prior)
        self.states.mu_posterior.append(self.mu_states_posterior)
        self.states.var_posterior.append(self.var_states_posterior)
        cov_states = self.var_states @ self.transition_matrix.T
        self.states.cov_states.append(cov_states)
        self.states.mu_smooth.append(self.mu_states_posterior)
        self.states.var_smooth.append(self.var_states_posterior)

    def set_states(
        self,
        new_mu_states: np.ndarray,
        new_var_states: np.ndarray,
    ):
        """
        Set new values for states, i.e., :attr:`~canari.model.Model.mu_states` and
        :attr:`~canari.model.Model.var_states`

        Args:
            new_mu_states (np.ndarray): Mean values to be set.
            new_var_states (np.ndarray): Covariance matrix to be set.
        """

        self.mu_states = new_mu_states.copy()
        self.var_states = new_var_states.copy()

    def _set_posterior_states(
        self,
        new_mu_states: np.ndarray,
        new_var_states: np.ndarray,
    ):
        """
        Set values the posterior hidden states, i.e.,
        :attr:`~canari.model.Model.mu_states_posterior` and
        :attr:`~canari.model.Model.var_states_posterior`

        Args:
            new_mu_states (np.ndarray): Posterior state means.
            new_var_states (np.ndarray): Posterior state variances.
        """

        self.mu_states_posterior = new_mu_states.copy()
        self.var_states_posterior = new_var_states.copy()

    def initialize_states_with_smoother_estimates(self):
        """
        Set hidden states :attr:`~canari.model.Model.mu_states` and
        :attr:`~canari.model.Model.var_states` using the smoothed estimates for hidden states
        at the first time step `t=1` stored in :attr:`~canari.model.Model.states`. This new hidden
        states act as the inital hidden states at `t=0` in the next epoch.
        """

        self.mu_states = self.states.mu_smooth[0].copy()
        self.var_states = np.diag(np.diag(self.states.var_smooth[0])).copy()
        if "level" in self.states_name and hasattr(self, "_mu_local_level"):
            local_level_index = self.get_states_index("level")
            self.mu_states[local_level_index] = self._mu_local_level

    def set_memory(self, states: StatesHistory, time_step: int):
        """
        Set :attr:`~canari.model.Model.mu_states`, :attr:`~canari.model.Model.var_states`, and
        :attr:`~canari.model.Model.lstm_output_history` with smoothed estimates from a specific
        time steps stored in :class:`~canari.model.Model.states`. This is to prepare for the next
        analysis by ensuring the continuity of these variables, e.g., if the next analysis starts
        from time step `t`, should set the memory to the time step `t`.

        If `t=0`, also set the means and variances for cell and hidden states of
        :attr:`~canari.model.Model.lstm_net` to zeros. If `t` is not 0, need to set cell and hidden
        states outside in the code using `Model.lstm_net.set_lstm_states(lstm_cell_hidden_states)`.

        Args:
            states (StatesHistory): Full history of hidden states over time.
            time_step (int): Index of timestep to restore.

        Examples:
            >>> # If the next analysis starts from the beginning of the time series
            >>> model.set_memory(states=model.states, time_step=0))
            >>> # If the next analysis starts from t = 200
            >>> model.set_memory(states=model.states, time_step=200))
        """

        if time_step == 0:
            self.initialize_states_with_smoother_estimates()
            # Dependent
            self.dep_model.lstm_output_history.initialize(
                self.dep_model.lstm_net.lstm_look_back_len
            )
            lstm_states = self.dep_model.lstm_net.get_lstm_states()
            for key in lstm_states:
                old_tuple = lstm_states[key]
                new_tuple = tuple(
                    np.zeros_like(np.array(v)).tolist() for v in old_tuple
                )
                lstm_states[key] = new_tuple
            self.dep_model.lstm_net.set_lstm_states(lstm_states)
            # Main
            self.main_model.lstm_output_history.initialize(
                self.main_model.lstm_net.lstm_look_back_len
            )
            lstm_states = self.main_model.lstm_net.get_lstm_states()
            for key in lstm_states:
                old_tuple = lstm_states[key]
                new_tuple = tuple(
                    np.zeros_like(np.array(v)).tolist() for v in old_tuple
                )
                lstm_states[key] = new_tuple
            self.main_model.lstm_net.set_lstm_states(lstm_states)

    def forward(
        self,
        input_covariates: Optional[np.ndarray] = None,
        mu_lstm_pred: Optional[np.ndarray] = None,
        var_lstm_pred: Optional[np.ndarray] = None,
    ):
        """ """
        # LSTM dependent model
        lstm_states_dep_index = self.get_states_index("lstm dep")
        x_dep = input_covariates[self.dep_model.input_col]
        mu_lstm_input_dep, var_lstm_input_dep = common.prepare_lstm_input(
            self.dep_model.lstm_output_history, x_dep
        )
        mu_lstm_pred_dep, var_lstm_pred_dep = self.dep_model.lstm_net.forward(
            mu_x=np.float32(mu_lstm_input_dep), var_x=np.float32(var_lstm_input_dep)
        )

        # LSTM main model
        lstm_states_index = self.get_states_index("lstm main")
        mu_x_main = np.concatenate((input_covariates, mu_lstm_pred_dep.flatten()))
        var_x_main = np.concatenate(
            (np.zeros_like(input_covariates), var_lstm_pred_dep.flatten())
        )
        mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
            self.main_model.lstm_output_history, mu_x_main, var_x_main
        )
        mu_lstm_pred, var_lstm_pred = self.main_model.lstm_net.forward(
            mu_x=np.float32(mu_lstm_input), var_x=np.float32(var_lstm_input)
        )

        # State-space model prediction:
        _, _, mu_states_prior, var_states_prior = common.forward(
            self.mu_states,
            self.var_states,
            self.transition_matrix,
            self.process_noise_matrix,
            self.observation_matrix,
        )

        mu_states_prior[lstm_states_dep_index] = mu_lstm_pred_dep
        var_states_prior[lstm_states_dep_index, lstm_states_dep_index] = (
            var_lstm_pred_dep
        )
        mu_states_prior[lstm_states_index] = mu_lstm_pred
        var_states_prior[lstm_states_index, lstm_states_index] = var_lstm_pred

        mu_obs_pred, var_obs_pred = common.calc_observation(
            mu_states_prior, var_states_prior, self.observation_matrix
        )

        self.mu_states_prior = mu_states_prior
        self.var_states_prior = var_states_prior
        self.mu_obs_predict = mu_obs_pred
        self.var_obs_predict = var_obs_pred

        return (
            mu_obs_pred,
            var_obs_pred,
            mu_states_prior,
            var_states_prior,
        )

    def backward(
        self,
        obs: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ """
        delta_mu_states, delta_var_states = common.backward(
            obs.reshape(-1, 1),
            self.mu_obs_predict,
            self.var_obs_predict,
            self.var_states_prior,
            self.observation_matrix,
        )
        delta_mu_states = np.nan_to_num(delta_mu_states, nan=0.0)
        delta_var_states = np.nan_to_num(delta_var_states, nan=0.0)
        mu_states_posterior = self.mu_states_prior + delta_mu_states
        var_states_posterior = self.var_states_prior + delta_var_states
        self.mu_states_posterior = mu_states_posterior
        self.var_states_posterior = var_states_posterior

        return (
            delta_mu_states,
            delta_var_states,
            mu_states_posterior,
            var_states_posterior,
        )

    def rts_smoother(
        self,
        time_step: int,
        matrix_inversion_tol: Optional[float] = 1e-12,
        tol_type: Optional[str] = "relative",  # relative of absolute
    ):
        """
        Apply RTS smoothing equations for a specity timestep. As a result of this function,
        the smoothed estimates for hidden states at the specific time step will be updated in
        :attr:`states`.

        This function is used at the one-time-step level. Recall :meth:`~canari.common.rts_smoother`
        from :class:`~canari.common`.

        Args:
            time_step (int): Target smoothing index.
            matrix_inversion_tol (float): Numerical stability threshold for matrix
                                            pseudoinversion (pinv). Defaults to 1E-12.
        """

        (
            self.states.mu_smooth[time_step],
            self.states.var_smooth[time_step],
        ) = common.rts_smoother(
            self.states.mu_prior[time_step + 1],
            self.states.var_prior[time_step + 1],
            self.states.mu_smooth[time_step + 1],
            self.states.var_smooth[time_step + 1],
            self.states.mu_posterior[time_step],
            self.states.var_posterior[time_step],
            self.states.cov_states[time_step + 1],
            matrix_inversion_tol,
            tol_type,
        )

    def smoother(
        self,
        matrix_inversion_tol: Optional[float] = 1e-12,
        tol_type: Optional[str] = "relative",  # relative of absolute
    ) -> StatesHistory:
        """
        Run the Kalman smoother over an entire time series data, i.e., repeatly apply the
        RTS smoothing equation over multiple time steps.

        This function is used at the entire-dataset-level. Recall repeatedly the function
        :meth:`rts_smoother` at one-time-step level from :class:`~canari.model.Model`.

        Args:
            matrix_inversion_tol (float): Numerical stability threshold for matrix
                                            pseudoinversion (pinv). Defaults to 1E-12.

        Returns:
            StatesHistory:
                :attr:`states`: The history of hidden states over time.

        Examples:
            >>> mu_preds_train, std_preds_train, states = model.filter(train_set)
            >>> states = model.smoother()
        """

        num_time_steps = len(self.states.mu_smooth)
        for time_step in reversed(range(0, num_time_steps - 1)):
            self.rts_smoother(time_step, matrix_inversion_tol, tol_type)

        return self.states

    def filter(
        self,
        data: Dict[str, np.ndarray],
    ):
        """
        Filter
        """

        mu_obs_preds = []
        std_obs_preds = []
        self.initialize_states_history()

        for x, y in zip(data["x"], data["y"]):
            mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = self.forward(
                x
            )
            (
                delta_mu_states,
                delta_var_states,
                mu_states_posterior,
                var_states_posterior,
            ) = self.backward(y)

            #
            lstm_index_dep = self.get_states_index("lstm dep")
            delta_mu_lstm_dep = np.array(
                delta_mu_states[lstm_index_dep]
                / var_states_prior[lstm_index_dep, lstm_index_dep]
            )
            delta_var_lstm_dep = np.array(
                delta_var_states[lstm_index_dep, lstm_index_dep]
                / var_states_prior[lstm_index_dep, lstm_index_dep] ** 2
            )
            self.dep_model.lstm_net.update_param(
                np.float32(delta_mu_lstm_dep), np.float32(delta_var_lstm_dep)
            )
            self.dep_model.lstm_output_history.update(
                mu_states_posterior[lstm_index_dep],
                var_states_posterior[lstm_index_dep, lstm_index_dep],
            )
            #
            lstm_index = self.get_states_index("lstm main")
            delta_mu_lstm = np.array(
                delta_mu_states[lstm_index] / var_states_prior[lstm_index, lstm_index]
            )
            delta_var_lstm = np.array(
                delta_var_states[lstm_index, lstm_index]
                / var_states_prior[lstm_index, lstm_index] ** 2
            )
            self.main_model.lstm_net.update_param(
                np.float32(delta_mu_lstm), np.float32(delta_var_lstm)
            )
            self.main_model.lstm_output_history.update(
                mu_states_posterior[lstm_index],
                var_states_posterior[lstm_index, lstm_index],
            )

            self._save_states_history()
            self.set_states(mu_states_posterior, var_states_posterior)
            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)

        return (
            np.array(mu_obs_preds),
            np.array(std_obs_preds),
            self.states,
        )

    def forecast(
        self, data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """ """
        mu_obs_preds = []
        std_obs_preds = []

        for x in data["x"]:
            mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = self.forward(
                x
            )

            lstm_index_dep = self.get_states_index("lstm dep")
            self.dep_model.lstm_output_history.update(
                mu_states_prior[lstm_index_dep],
                var_states_prior[lstm_index_dep, lstm_index_dep],
            )
            #
            lstm_index = self.get_states_index("lstm main")
            self.main_model.lstm_output_history.update(
                mu_states_prior[lstm_index],
                var_states_prior[lstm_index, lstm_index],
            )

            self._set_posterior_states(mu_states_prior, var_states_prior)
            self._save_states_history()
            self.set_states(mu_states_prior, var_states_prior)
            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)
        return (
            np.array(mu_obs_preds),
            np.array(std_obs_preds),
            self.states,
        )

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
        white_noise_decay: Optional[bool] = True,
        white_noise_max_std: Optional[float] = 5,
        white_noise_decay_factor: Optional[float] = 0.9,
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """
        Train LSTM
        """

        self.filter(train_data)
        self.smoother()
        mu_validation_preds, std_validation_preds, _ = self.forecast(validation_data)

        return (
            np.array(mu_validation_preds),
            np.array(std_validation_preds),
            self.states,
            # np.concatenate(mu_validation_preds, axis=1).T,
            # np.concatenate(std_validation_preds, axis=1).T,
            # self.states,
        )
