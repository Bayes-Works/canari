import numpy as np
import copy
from typing import Optional, List, Tuple, Dict
from collections import deque
from pytagi.nn import Sequential
import pytagi.metric as metric
from src.base_component import BaseComponent
import src.common as common
from src.data_struct import LstmOutputHistory, StatesHistory
from src.common import GMA


class Model:
    """
    LSTM/SSM model
    """

    def __init__(
        self,
        *components: BaseComponent,
    ):
        if not components:
            pass
        else:
            self.components = list(components)
            self.define_model()
            self.initialize_lstm_network()
            self.initialize_autoregression_component()
            self.initialize_early_stop()
            self.states = StatesHistory()

    def __deepcopy__(self, memo):
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            if k in ["lstm_net"]:
                v = None
            setattr(obj, k, copy.deepcopy(v, memo))
            pass
        return obj

    def initialize_early_stop(self):
        """
        Initialize for early stopping
        """

        self.early_stop_metric = None
        self.early_stop_metric_history = []
        self.early_stop_lstm_param = None
        self.early_stop_init_mu_states = None
        self.early_stop_init_var_states = None
        self.optimal_epoch = 0
        self._current_epoch = 0
        self.stop_training = False

    def define_model(self):
        """
        Assemble components
        """

        self._assemble_matrices()
        self._assemble_states()

    def _assemble_matrices(self):
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

    def _assemble_states(self):
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
        if "lstm" in self.states_name:
            self.lstm_states_index = self.states_name.index("lstm")
        else:
            self.lstm_states_index = None

    def initialize_autoregression_component(self):
        """
        Initialize autoregression component
        """
        autoregression_component = next(
            (
                component
                for component in self.components
                if component.component_name == "autoregression"
            ),
            None,
        )

        if "autoregression" in self.states_name:
            self.autoregression_index = self.states_name.index("autoregression")
            self.mu_W2bar = None
            self.var_W2bar = None
            self.mu_W2_prior = None
            self.var_W2_prior = None
            if "phi" in self.states_name:
                self.phi_index = self.states_name.index("phi")
            if "AR_error" in self.states_name:
                self.ar_error_index = self.states_name.index("AR_error")
                self.W2_index = self.states_name.index("W2")
                self.W2bar_index = self.states_name.index("W2bar")
                self.mu_W2bar = autoregression_component.mu_states[-1]
                self.var_W2bar = autoregression_component.var_states[-1]

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
            self.lstm_output_history = LstmOutputHistory()
            self.lstm_output_history.initialize(self._lstm_look_back_len)
        else:
            self.lstm_net = None

    def update_lstm_param(
        self,
        delta_mu_lstm: np.ndarray,
        delta_var_lstm: np.ndarray,
    ):
        """
        update lstm network's parameters
        """

        self.lstm_net.input_delta_z_buffer.delta_mu = delta_mu_lstm
        self.lstm_net.input_delta_z_buffer.delta_var = [delta_var_lstm]
        self.lstm_net.backward()
        self.lstm_net.step()

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
                self.mu_states[i] = np.nanmean(y_no_nan)
                if self.var_states[i, i] == 0:
                    self.var_states[i, i] = 1e-2
            elif _state_name == "local trend":
                self.mu_states[i] = coefficients[1]
                if self.var_states[i, i] == 0:
                    self.var_states[i, i] = 1e-2
            elif _state_name == "local acceleration":
                self.mu_states[i] = coefficients[0]
                if self.var_states[i, i] == 0:
                    self.var_states[i, i] = 1e-5

        self._mu_local_level = np.nanmean(y_no_nan)

    def estimate_posterior_states(
        self,
        delta_mu_states: np.ndarray,
        delta_var_states: np.ndarray,
    ):
        """
        Estimate the posterirors for the states
        """
        mu_states_posterior = self.mu_states_prior + delta_mu_states
        var_states_posterior = self.var_states_prior + delta_var_states
        return mu_states_posterior, var_states_posterior

    def set_posterior_states(
        self,
        new_mu_states: np.ndarray,
        new_var_states: np.ndarray,
    ):
        """
        Set the posterirors for the states
        """

        self.mu_states_posterior = new_mu_states.copy()
        self.var_states_posterior = new_var_states.copy()

    def update_lstm_output_history(self, mu_lstm_pred, var_lstm_pred):
        self.lstm_output_history.mu = np.roll(self.lstm_output_history.mu, -1)
        self.lstm_output_history.var = np.roll(self.lstm_output_history.var, -1)
        self.lstm_output_history.mu[-1] = mu_lstm_pred.item()
        self.lstm_output_history.var[-1] = var_lstm_pred.item()

    def set_states(
        self,
        new_mu_states: np.ndarray,
        new_var_states: np.ndarray,
    ):
        """
        Assign new states
        """

        self.mu_states = new_mu_states.copy()
        self.var_states = new_var_states.copy()

    def initialize_states_with_smoother_estimates(self):
        """
        Set the model initial hidden states = the smoothed estimates
        """

        self.mu_states = self.states.mu_smooth[0].copy()
        self.var_states = np.diag(np.diag(self.states.var_smooth[0])).copy()
        if "local level" in self.states_name and hasattr(self, "_mu_local_level"):
            local_level_index = self.states_name.index("local level")
            self.mu_states[local_level_index] = self._mu_local_level

    def initialize_lstm_output_history(self):
        self.lstm_output_history.initialize(self._lstm_look_back_len)

    def initialize_states_history(self):
        self.states.initialize(self.states_name)

    def save_states_history(self):
        """
        Save states' priors, posteriors and cross-covariances for smoother
        """
        self.states.mu_prior.append(self.mu_states_prior)
        self.states.var_prior.append(
            (self.var_states_prior + self.var_states_prior.T) * 0.5
        )
        self.states.mu_posterior.append(self.mu_states_posterior)
        self.states.var_posterior.append(self.var_states_posterior)
        if "AR_error" in self.states_name:
            self.ar_error_index = self.states_name.index("AR_error")
            self.W2_index = self.states_name.index("W2")
            self.W2bar_index = self.states_name.index("W2bar")
            cov_states = self.var_states @ self.transition_matrix.T
            # Set covariance between W and other states to zero
            cov_states[self.ar_error_index, :] = np.zeros_like(
                cov_states[self.ar_error_index, :]
            )
            cov_states[:, self.ar_error_index] = np.zeros_like(
                cov_states[:, self.ar_error_index]
            )
            cov_states[self.W2_index, :] = np.zeros_like(cov_states[self.W2_index, :])
            cov_states[:, self.W2_index] = np.zeros_like(cov_states[:, self.W2_index])
            cov_states[self.W2bar_index, :] = np.zeros_like(
                cov_states[self.W2bar_index, :]
            )
            cov_states[:, self.W2bar_index] = np.zeros_like(
                cov_states[:, self.W2bar_index]
            )
            self.states.cov_states.append(cov_states)
        else:
            self.states.cov_states.append(self.var_states @ self.transition_matrix.T)
        self.states.mu_smooth.append([])
        self.states.var_smooth.append([])

    def initialize_smoother_buffers(self):
        """
        Set the smoothed estimates at the last time step = posterior
        """

        self.states.mu_smooth[-1] = self.states.mu_posterior[-1].copy()
        self.states.var_smooth[-1] = self.states.var_posterior[-1].copy()

    def create_compatible_model(self, target_model) -> None:
        """
        Create compatiable model by padding zero to states and matrices
        Using in SKF
        """

        pad_row = np.zeros((self.num_states)).flatten()
        pad_col = np.zeros((target_model.num_states)).flatten()
        for i, state in enumerate(target_model.states_name):
            if state not in self.states_name:
                self.mu_states = common.pad_matrix(
                    self.mu_states, i, pad_row=np.zeros(1)
                )
                self.var_states = common.pad_matrix(
                    self.var_states, i, pad_row, pad_col
                )
                self.transition_matrix = common.pad_matrix(
                    self.transition_matrix, i, pad_row, pad_col
                )
                self.process_noise_matrix = common.pad_matrix(
                    self.process_noise_matrix, i, pad_row, pad_col
                )
                self.observation_matrix = common.pad_matrix(
                    self.observation_matrix, i, pad_col=np.zeros(1)
                )
                self.num_states += 1
                self.states_name.insert(i, state)
                self.index_pad_state = i
        if "lstm" in self.states_name:
            self.lstm_states_index = target_model.states_name.index("lstm")
        else:
            self.lstm_states_index = None

    def online_AR_forward_modification(self) -> None:
        """
        Online AR forward modification
        """

        ar_index = self.states_name.index("autoregression")
        if "phi" in self.states_name:
            phi_index = self.states_name.index("phi")

            # GMA operations
            self.mu_states, self.var_states = GMA(
                self.mu_states,
                self.var_states,
                index1=phi_index,
                index2=ar_index,
                replace_index=ar_index,
            ).get_results()

            # Cap phi_AR if it is bigger than 1: for numerical stability in BAR later
            self.mu_states[phi_index] = (
                0.9999 if self.mu_states[phi_index] >= 1 else self.mu_states[phi_index]
            )

        if "AR_error" in self.states_name:
            ar_error_index = self.states_name.index("AR_error")
            W2_index = self.states_name.index("W2")
            W2bar_index = self.states_name.index("W2bar")

            # Forward path to compute the moments of W
            # # W2bar
            self.mu_states[W2bar_index] = self.mu_W2bar
            self.var_states[W2bar_index, W2bar_index] = self.var_W2bar

            # # From W2bar to W2
            self.mu_W2_prior = self.mu_W2bar
            self.var_W2_prior = 3 * self.var_W2bar + 2 * self.mu_W2bar**2
            self.mu_states[W2_index] = self.mu_W2_prior
            self.var_states[W2_index, W2_index] = self.var_W2_prior

            # # From W2 to W
            self.mu_states[ar_error_index] = 0
            self.var_states[ar_error_index, :] = np.zeros_like(
                self.var_states[ar_error_index, :]
            )
            self.var_states[:, ar_error_index] = np.zeros_like(
                self.var_states[:, ar_error_index]
            )
            self.var_states[ar_error_index, ar_error_index] = self.mu_W2bar
            self.var_states[ar_error_index, ar_index] = self.mu_W2bar
            self.var_states[ar_index, ar_error_index] = self.mu_W2bar

            # Replace the process error variance in self.process_noise_matrix
            self.process_noise_matrix[ar_index, ar_index] = self.mu_W2bar

    def online_AR_backward_modification(
        self,
        mu_states_posterior,
        var_states_posterior,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Online AR backwar modification
        """

        ar_error_index = self.states_name.index("AR_error")
        W2_index = self.states_name.index("W2")
        W2bar_index = self.states_name.index("W2bar")

        # Backward path to update W2 and W2bar
        # # From W to W2
        mu_W2_posterior = (
            mu_states_posterior[ar_error_index] ** 2
            + var_states_posterior[ar_error_index, ar_error_index]
        )
        var_W2_posterior = (
            2 * var_states_posterior[ar_error_index, ar_error_index] ** 2
            + 4
            * var_states_posterior[ar_error_index, ar_error_index]
            * mu_states_posterior[ar_error_index] ** 2
        )
        mu_states_posterior[W2_index] = mu_W2_posterior
        var_states_posterior[W2_index, :] = np.zeros_like(
            var_states_posterior[W2_index, :]
        )
        var_states_posterior[:, W2_index] = np.zeros_like(
            var_states_posterior[:, W2_index]
        )
        var_states_posterior[W2_index, W2_index] = var_W2_posterior

        # # From W2 to W2bar
        K = self.var_W2bar / self.var_W2_prior
        self.mu_W2bar = self.mu_W2bar + K * (mu_W2_posterior - self.mu_W2_prior)
        self.var_W2bar = self.var_W2bar + K**2 * (var_W2_posterior - self.var_W2_prior)
        mu_states_posterior[W2bar_index] = self.mu_W2bar
        var_states_posterior[W2bar_index, :] = np.zeros_like(
            var_states_posterior[W2bar_index, :]
        )
        var_states_posterior[:, W2bar_index] = np.zeros_like(
            var_states_posterior[:, W2bar_index]
        )
        var_states_posterior[W2bar_index, W2bar_index] = self.var_W2bar

        return mu_states_posterior, var_states_posterior

    def forward(
        self,
        input_covariates: Optional[np.ndarray] = None,
        mu_lstm_pred: Optional[np.ndarray] = None,
        var_lstm_pred: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        One step prediction in states-space model
        """

        # LSTM prediction:
        if self.lstm_net and input_covariates is not None:
            mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                self.lstm_output_history, input_covariates
            )
            mu_lstm_pred, var_lstm_pred = self.lstm_net.forward(
                mu_x=np.float32(mu_lstm_input), var_x=np.float32(var_lstm_input)
            )

        # Autoregression
        if "autoregression" in self.states_name:
            self.online_AR_forward_modification()

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
        self.mu_states_prior = mu_states_prior
        self.var_states_prior = var_states_prior
        self.mu_obs_prior = mu_obs_pred
        self.var_obs_prior = var_obs_pred

        return mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior

    def backward(
        self,
        obs: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Update step in states-space model
        """

        delta_mu_states, delta_var_states = common.backward(
            obs,
            self.mu_obs_prior,
            self.var_obs_prior,
            self.var_states_prior,
            self.observation_matrix,
        )
        delta_mu_states = np.nan_to_num(delta_mu_states, nan=0.0)
        delta_var_states = np.nan_to_num(delta_var_states, nan=0.0)
        mu_states_posterior, var_states_posterior = self.estimate_posterior_states(
            delta_mu_states, delta_var_states
        )

        if "AR_error" in self.states_name:
            mu_states_posterior, var_states_posterior = (
                self.online_AR_backward_modification(
                    mu_states_posterior,
                    var_states_posterior,
                )
            )

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
    ):
        """
        RTS smoother
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
        )

    def forecast(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forecast for whole time series data
        """

        data = common.set_default_input_covariates(data)
        mu_obs_preds = []
        std_obs_preds = []
        lstm_index = self.lstm_states_index

        for x in data["x"]:
            mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = self.forward(
                x
            )

            if self.lstm_net:
                self.update_lstm_output_history(
                    mu_states_prior[lstm_index],
                    var_states_prior[lstm_index, lstm_index],
                )
            self.set_posterior_states(mu_states_prior, var_states_prior)
            self.save_states_history()
            self.set_states(mu_states_prior, var_states_prior)
            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)
        return np.array(mu_obs_preds).flatten(), np.array(std_obs_preds).flatten()

    def filter(
        self,
        data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter for whole time series data
        """

        data = common.set_default_input_covariates(data)
        lstm_index = self.lstm_states_index
        mu_obs_preds = []
        std_obs_preds = []
        self.initialize_states_history()

        for x, y in zip(data["x"], data["y"]):
            mu_obs_pred, var_obs_pred, _, var_states_prior = self.forward(x)
            (
                delta_mu_states,
                delta_var_states,
                mu_states_posterior,
                var_states_posterior,
            ) = self.backward(y)

            if self.lstm_net:
                delta_mu_lstm = np.array(
                    delta_mu_states[lstm_index]
                    / var_states_prior[lstm_index, lstm_index]
                )
                delta_var_lstm = np.array(
                    delta_var_states[lstm_index, lstm_index]
                    / var_states_prior[lstm_index, lstm_index] ** 2
                )
                self.lstm_net.update_param(
                    np.float32(delta_mu_lstm), np.float32(delta_var_lstm)
                )
                self.update_lstm_output_history(
                    mu_states_posterior[lstm_index],
                    var_states_posterior[lstm_index, lstm_index],
                )

            self.save_states_history()
            self.set_states(mu_states_posterior, var_states_posterior)
            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)

        return np.array(mu_obs_preds).flatten(), np.array(std_obs_preds).flatten()

    def smoother(self, data: Dict[str, np.ndarray]) -> None:
        """
        Smoother for whole time series
        """

        num_time_steps = len(data["y"])
        # Smoother
        self.initialize_smoother_buffers()
        for time_step in reversed(range(0, num_time_steps - 1)):
            self.rts_smoother(time_step)

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """
        Train LstmNetwork
        """

        if self.lstm_net is None:
            self.initialize_lstm_network()

        self.filter(train_data)
        self.smoother(train_data)
        mu_validation_preds, std_validation_preds = self.forecast(validation_data)
        self.initialize_lstm_output_history()
        self.initialize_states_with_smoother_estimates()
        return (
            np.array(mu_validation_preds).flatten(),
            np.array(std_validation_preds).flatten(),
            self.states,
        )

    def early_stopping(
        self,
        mode: Optional[str] = "max",
        patience: Optional[int] = 20,
        evaluate_metric: Optional[float] = None,
        skip_epoch: Optional[int] = 5,
    ) -> Tuple[bool, int, float, list]:

        if self._current_epoch == 0:
            if mode == "max":
                self.early_stop_metric = -np.inf
            elif mode == "min":
                self.early_stop_metric = np.inf

        self.early_stop_metric_history.append(evaluate_metric)

        # Check for improvement
        improved = False
        if (
            mode == "max"
            and evaluate_metric > self.early_stop_metric
            and self._current_epoch > skip_epoch
        ):
            improved = True
        elif (
            mode == "min"
            and evaluate_metric < self.early_stop_metric
            and self._current_epoch > skip_epoch
        ):
            improved = True

        # Update metric and parameters if there's an improvement
        if improved:
            self.early_stop_metric = copy.copy(evaluate_metric)
            self.early_stop_lstm_param = copy.copy(self.lstm_net.get_state_dict())
            self.early_stop_init_mu_states = copy.copy(self.mu_states)
            self.early_stop_init_var_states = copy.copy(self.var_states)
            self.optimal_epoch = copy.copy(self._current_epoch)

        self._current_epoch += 1

        # Check stop condition and assign optimal values
        if (self._current_epoch - self.optimal_epoch) >= patience:
            self.stop_training = True
            self.lstm_net.load_state_dict(self.early_stop_lstm_param)
            self.set_states(
                self.early_stop_init_mu_states, self.early_stop_init_var_states
            )

        return (
            self.stop_training,
            self.optimal_epoch,
            self.early_stop_metric,
            self.early_stop_metric_history,
        )
