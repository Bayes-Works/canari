"""
Hybrid LSTM-SSM model that combines Bayesian Long-short Term Memory (LSTM) Neural Networks
and State-Space Models (SSM).

This model supports a flexible architecture where multiple `components`
are assembled to define a structured state-space model.

On time series data, the model can:

    - Provide forecasts with associated uncertainties.
    - Decompose orginal time serires data into unobserved hidden states. Provide mean values and associate uncertainties for these hidden states.
    - Train its Bayesian LSTM network component.
    - Support forecasting, filtering, and smoothing operations.
    - Generate synthetic time series data, including synthetic anomaly injection.

References:
    Vuong, V.D., Nguyen, L.H. and Goulet, J.-A. (2025). `Coupling LSTM neural networks and
    state-space models through analytically tractable inference
    <https://www.sciencedirect.com/science/article/pii/S0169207024000335>`_.
    International Journal of Forecasting. Volume 41, Issue 1, Pages 128-140.

"""

import copy
from typing import Optional, List, Tuple, Dict
import numpy as np
from pytagi import Normalizer as normalizer
from canari.component.base_component import BaseComponent
import canari.common as common
from canari.data_struct import LstmOutputHistory, StatesHistory
from canari.common import GMA
from canari.data_process import DataProcess


class Model:
    """
    `Model` class for the Hybrid LSTM/SSM model.

    Args:
        *components (BaseComponent): One or more instances of classes derived from
                            :class:`~canari.component.base_component.BaseComponent`.

    Examples:
        >>> model = Model(LocalTrend(), Periodic(),WhiteNoise())
        >>> mu_pred, std_pred, _ = model.forecast(data)

    Attributes:
        components (Dict[str, BaseComponent]):
            Dictionary to save model components' configurations.
        num_states (int):
            Number of hidden states.
        states_names (list[str]):
            Names of hidden states.
        mu_states (np.ndarray):
            Mean vector for the hidden states :math:`x_t` at the time step `t`.
        var_states (np.ndarray):
            Covariance matrix for the hidden states :math:`x_t` at the time step `t`.
        mu_states_prior (np.ndarray):
            Prior mean vector for the hidden states :math:`x_{t+1}` at the time step `t+1`.
        var_states_prior (np.ndarray):
            Prior covariance matrix for the hidden states :math:`x_{t+1}` at the time step `t+1`.
        mu_states_posterior (np.ndarray):
            Posteriror mean vector for the hidden states :math:`x_{t+1}` at the time step `t+1`.
        var_states_posterior (np.ndarray):
            Posteriror covariance matrix for the hidden states :math:`x_{t+1}` at the time
            step `t+1`.
        states (StatesHistory):
            Container for storing prior, posterior, and smoothed values of hidden states over time.
        mu_obs_predict (np.ndarray):
            Means for predictions at a time step `t+1`.
        var_obs_predict (np.ndarray):
            Variances for predictions at a time step `t+1`.
        observation_matrix (np.ndarray):
            Global observation matrix constructed from all components.
        transition_matrix (np.ndarray):
            Global transition matrix constructed from all components.
        process_noise_matrix (np.ndarray):
            Global process noise matrix constructed from all components.

        # LSTM-related attributes: only being used when a :class:`~canari.component.lstm_component.LstmNetwork` component is found.

        lstm_net (:class:`pytagi.Sequential`):
            LSTM neural network that is generated from the
            :class:`~canari.component.lstm_component.LstmNetwork` component, if present.
            It is a :class:`pytagi.Sequential` instance.
        lstm_output_history (LstmOutputHistory):
            Container for saving a rolling history of LSTM output over a fixed look-back window.

        # Early stopping attributes: only being used when training a :class:`~canari.component.lstm_component.LstmNetwork` component.

        early_stop_metric (float):
            Best value of the metric being monitored.
        early_stop_metric_history (List[float]):
            Logged history of metric values across epochs.
        early_stop_lstm_param (Dict):
            LSTM's weight and bias parameters at the optimal epoch for :class:`pytagi.Sequential`.
        early_stop_init_mu_states (np.ndarray):
            Copy of `mu_states` at time step :math:`t=0` of the optimal epoch .
        early_stop_init_var_states (np.ndarray):
            Copy of `var_states` at time step :math:`t=0` of the optimal epoch .
        optimal_epoch (int):
            Epoch at which the monitored metric was best.
        stop_training (bool):
            Flag indicating whether training has been stopped due to
            early stopping or reaching maximum number of epoch.
    """

    def __init__(
        self,
        *components: BaseComponent,
    ):
        """
        Initialize the model from components.
        """

        self._initialize_attributes()
        self.components = {
            component.component_name: component for component in components
        }
        self._initialize_model()
        self.states = StatesHistory()

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the model while excluding the LSTM network.

        Args:
            memo (dict): Python deepcopy memoization dictionary.

        Returns:
            Model: Deep-copied instance without LSTM network state.
        """

        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            if k in ["lstm_net"]:
                v = None
            setattr(obj, k, copy.deepcopy(v, memo))
        return obj

    def _initialize_attributes(self):
        """
        Initialize default model attributes.
        """

        # General attributes
        self.components = {}
        self.num_states = 0
        self.states_name = []

        # State-space model matrices
        self.mu_states = None
        self.var_states = None
        self.mu_states_prior = None
        self.var_states_prior = None
        self.mu_states_posterior = None
        self.var_states_posterior = None
        self.mu_obs_predict = None
        self.var_obs_predict = None
        self.transition_matrix = None
        self.process_noise_matrix = None
        self.observation_matrix = None

        # LSTM-related attributes
        self.lstm_net = None
        self.lstm_output_history = LstmOutputHistory()

        # TODO: use internal variables
        # Autoregression-related attributes
        self.mu_W2bar = None
        self.var_W2bar = None
        self.mu_W2_prior = None
        self.var_W2_prior = None

        # Early stopping attributes
        self.early_stop_metric = None
        self.early_stop_metric_history = []
        self.early_stop_lstm_param = None
        self.early_stop_init_mu_states = None
        self.early_stop_init_var_states = None
        self.optimal_epoch = 0
        self._current_epoch = 0
        self.stop_training = False

    def _initialize_model(self):
        """
        Set up the model by assembling matrices, initializing states,
        configuring LSTM and autoregressive modules if included.
        """

        self._assemble_matrices()
        self._assemble_states()
        self._initialize_lstm_network()
        self._initialize_autoregression()

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
            *(component.transition_matrix for component in self.components.values())
        )

        # Assemble process noise matrices
        self.process_noise_matrix = common.create_block_diag(
            *(component.process_noise_matrix for component in self.components.values())
        )

        # Assemble observation matrices
        global_observation_matrix = np.array([])
        for component in self.components.values():
            global_observation_matrix = np.concatenate(
                (global_observation_matrix, component.observation_matrix[0, :]), axis=0
            )
        self.observation_matrix = np.atleast_2d(global_observation_matrix)

    def _assemble_states(self):
        """
        Concatenate state means and variances from all components.
        """

        self.mu_states = np.vstack(
            [component.mu_states for component in self.components.values()]
        )
        self.var_states = np.vstack(
            [component.var_states for component in self.components.values()]
        )
        self.var_states = np.diagflat(self.var_states)
        self.states_name = [
            state
            for component in self.components.values()
            for state in component.states_name
        ]
        self.num_states = sum(
            component.num_states for component in self.components.values()
        )

    def _initialize_lstm_network(self):
        """
        Initialize and configure LSTM network if there is a LstmNetwork component is used.
        """

        lstm_component = next(
            (
                component
                for component in self.components.values()
                if component.component_name == "lstm"
            ),
            None,
        )
        if lstm_component:
            self.lstm_net = lstm_component.initialize_lstm_network()
            self.lstm_net.update_param = self._update_lstm_param
            self.lstm_output_history.initialize(self.lstm_net.lstm_look_back_len)

    def _initialize_autoregression(self):
        """
        Initialize autoregression-related attributes.
        Only applicable when using the Autoregression component.
        """

        autoregression_component = next(
            (
                component
                for component in self.components.values()
                if component.component_name == "autoregression"
            ),
            None,
        )

        if "AR_error" in self.states_name:
            self.mu_W2bar = autoregression_component.mu_states[-1]
            self.var_W2bar = autoregression_component.var_states[-1]

    def _update_lstm_param(
        self,
        delta_mu_lstm: np.ndarray,
        delta_var_lstm: np.ndarray,
    ):
        """
        Update LSTM neural network's parameters.

        Args:
            delta_mu_lstm (np.ndarray): Delta mean update for LSTM's output.
            delta_var_lstm (np.ndarray): Delta variance for LSTM's output.
        """

        self.lstm_net.set_delta_z(np.array(delta_mu_lstm), np.array(delta_var_lstm))
        self.lstm_net.backward()
        self.lstm_net.step()

    def _white_noise_decay(
        self, epoch: int, white_noise_max_std: float, white_noise_decay_factor: float
    ):
        """
        TODO: remove epoch from input
        Apply exponential decay to white noise standard deviation over epochs.

        Args:
            epoch (int): Current training epoch.
            white_noise_max_std (float): Maximum allowed noise std.
            white_noise_decay_factor (float): Factor controlling decay rate.
        """

        noise_index = self.get_states_index("white noise")
        scheduled_sigma_v = white_noise_max_std * np.exp(
            -white_noise_decay_factor * epoch
        )
        if scheduled_sigma_v < self.components["white noise"].std_error:
            scheduled_sigma_v = self.components["white noise"].std_error
        self.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

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

    def _online_AR_forward_modification(self, mu_states_prior, var_states_prior):
        """
        Apply forward path autoregressive (AR) moment transformations.

        Updates prior state means and variances based on autoregressive error modeling,
        propagating uncertainty from W2bar to AR_error. These are used during the forward
        pass when AR components are present.

        Args:
            mu_states_prior (np.ndarray): Prior mean vector of the states.
            var_states_prior (np.ndarray): Prior variance-covariance matrix of the states.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated (mu_states_prior, var_states_prior).
        """

        if "AR_error" in self.states_name:
            ar_index = self.get_states_index("autoregression")
            ar_error_index = self.get_states_index("AR_error")
            W2_index = self.get_states_index("W2")
            W2bar_index = self.get_states_index("W2bar")

            # Forward path to compute the moments of W
            # # W2bar
            mu_states_prior[W2bar_index] = self.mu_W2bar
            var_states_prior[W2bar_index, W2bar_index] = self.var_W2bar

            # # From W2bar to W2
            self.mu_W2_prior = self.mu_W2bar
            self.var_W2_prior = 3 * self.var_W2bar + 2 * self.mu_W2bar**2
            mu_states_prior[W2_index] = self.mu_W2_prior
            var_states_prior[W2_index, W2_index] = self.var_W2_prior

            # # From W2 to W
            mu_states_prior[ar_error_index] = 0
            var_states_prior[ar_error_index, :] = np.zeros_like(
                var_states_prior[ar_error_index, :]
            )
            var_states_prior[:, ar_error_index] = np.zeros_like(
                var_states_prior[:, ar_error_index]
            )
            var_states_prior[ar_error_index, ar_error_index] = self.mu_W2bar
            var_states_prior[ar_error_index, ar_index] = self.mu_W2bar
            var_states_prior[ar_index, ar_error_index] = self.mu_W2bar
        return mu_states_prior, var_states_prior

    def _online_AR_backward_modification(
        self,
        mu_states_posterior,
        var_states_posterior,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backward AR moment updates during state-space filtering.

        Computes the posterior distribution of W2 and W2bar from AR_error states,
        and adjusts the autoregressive process noise accordingly. Also applies
        GMA transformations when "phi" is involved in the model.

        Args:
            mu_states_posterior (np.ndarray): Posterior mean vector of the states.
            var_states_posterior (np.ndarray): Posterior variance-covariance matrix of the states.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated (mu_states_posterior, var_states_posterior).
        """

        if "phi" in self.states_name:
            # GMA operations
            mu_states_posterior, var_states_posterior = GMA(
                mu_states_posterior,
                var_states_posterior,
                index1=self.get_states_index("phi"),
                index2=self.get_states_index("autoregression"),
                replace_index=self.get_states_index("phi_autoregression"),
            ).get_results()

        if "AR_error" in self.states_name:
            ar_index = self.get_states_index("autoregression")
            ar_error_index = self.get_states_index("AR_error")
            W2_index = self.get_states_index("W2")
            W2bar_index = self.get_states_index("W2bar")

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
            self.var_W2bar = self.var_W2bar + K**2 * (
                var_W2_posterior - self.var_W2_prior
            )
            mu_states_posterior[W2bar_index] = self.mu_W2bar
            var_states_posterior[W2bar_index, :] = np.zeros_like(
                var_states_posterior[W2bar_index, :]
            )
            var_states_posterior[:, W2bar_index] = np.zeros_like(
                var_states_posterior[:, W2bar_index]
            )
            var_states_posterior[W2bar_index, W2bar_index] = self.var_W2bar

            self.process_noise_matrix[ar_index, ar_index] = self.mu_W2bar

        return mu_states_posterior, var_states_posterior

    def _prepare_covariates_generation(
        self, initial_covariate, num_generated_samples: int, time_covariates: List[str]
    ):
        """
        Generate structured time-based covariates for simulation purposes.

        Each covariate (e.g., hour_of_day, day_of_week) is computed cyclically using
        modular arithmetic to simulate realistic calendar-based signals.

        Args:
            initial_covariate (int): Starting value for time-based covariates.
            num_generated_samples (int): Total number of steps to generate.
            time_covariates (List[str]): List of time covariate names to encode.

        Returns:
            np.ndarray: Encoded covariate matrix of shape (num_generated_samples, 1).
        """

        covariates_generation = np.arange(0, num_generated_samples).reshape(-1, 1)
        for time_cov in time_covariates:
            if time_cov == "hour_of_day":
                covariates_generation = (
                    initial_covariate + covariates_generation
                ) % 24 + 1
            elif time_cov == "day_of_week":
                covariates_generation = (
                    initial_covariate + covariates_generation
                ) % 7 + 1
            elif time_cov == "day_of_year":
                covariates_generation = (
                    initial_covariate + covariates_generation
                ) % 365 + 1
            elif time_cov == "week_of_year":
                covariates_generation = (
                    initial_covariate + covariates_generation
                ) % 52 + 1
            elif time_cov == "month_of_year":
                covariates_generation = (
                    initial_covariate + covariates_generation
                ) % 12 + 1
            elif time_cov == "quarter_of_year":
                covariates_generation = (
                    initial_covariate + covariates_generation
                ) % 4 + 1
        return covariates_generation

    def get_dict(self) -> dict:
        """
        Export model attributes into a serializable dictionary.

        Returns:
            dict: Serializable model dictionary containing neccessary attributes.
        """

        save_dict = {}
        save_dict["components"] = self.components
        save_dict["mu_states"] = self.mu_states
        save_dict["var_states"] = self.var_states
        if self.lstm_net:
            save_dict["lstm_network_params"] = self.lstm_net.state_dict()
        # TODO: do save_dict["states_name"], remove saving indexes
        if "phi" in self.states_name:
            save_dict["phi_index"] = self.states_name.index("phi")
        if "autoregression" in self.states_name:
            save_dict["autoregression_index"] = self.states_name.index("autoregression")
        if "W2bar" in self.states_name:
            save_dict["W2bar_index"] = self.states_name.index("W2bar")

        return save_dict

    @staticmethod
    def load_dict(save_dict: dict):
        """
        Load model from a dictionary.

        Args:
            save_dict (dict): Dictionary containing saved model structure and parameters.

        Returns:
            Model: An instance of :class:`~canari.model.Model` generated from the input dictionary.
        """

        components = list(save_dict["components"].values())
        model = Model(*components)
        model.set_states(save_dict["mu_states"], save_dict["var_states"])
        if model.lstm_net:
            model.lstm_net.load_state_dict(save_dict["lstm_network_params"])

        return model

    def get_states_index(self, states_name: str):
        """
        Retrieve index of a state in the state vector.

        Args:
            states_name (str): The name of the state.

        Returns:
            int or None: Index of the state, or None if not found.
        """

        index = (
            self.states_name.index(states_name)
            if states_name in self.states_name
            else None
        )
        return index

    def auto_initialize_baseline_states(self, data: np.ndarray):
        """
        Automatically assign initial means and variances for baseline hidden states (local level,
        local trend, and local acceleration) from input data using time series decomposition
        defined in :meth:`~canari.data_process.DataProcess.decompose_data`.

        Args:
            data (np.ndarray): Time series data.
        """

        trend, slope, _, _ = DataProcess.decompose_data(data.flatten())

        for i, _state_name in enumerate(self.states_name):
            if _state_name == "local level":
                self.mu_states[i] = trend[0]
                if self.var_states[i, i] == 0:
                    self.var_states[i, i] = 1e-2
            elif _state_name == "local trend":
                self.mu_states[i] = slope
                if self.var_states[i, i] == 0:
                    self.var_states[i, i] = 1e-2
            elif _state_name == "local acceleration":
                self.mu_states[i] = 0
                if self.var_states[i, i] == 0:
                    self.var_states[i, i] = 1e-5

        self._mu_local_level = trend[0]

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

    def initialize_states_with_smoother_estimates(self):
        """
        Set hidden states :attr:`~canari.model.Model.mu_states` and
        :attr:`~canari.model.Model.var_states` using the smoothed estimates for hidden states
        at the first time step `t=1` stored in :attr:`~canari.model.Model.states`. This new hidden
        states act as the inital hidden states at `t=0` in the next epoch.
        """

        self.mu_states = self.states.mu_smooth[0].copy()
        self.var_states = np.diag(np.diag(self.states.var_smooth[0])).copy()
        if "local level" in self.states_name and hasattr(self, "_mu_local_level"):
            local_level_index = self.get_states_index("local level")
            self.mu_states[local_level_index] = self._mu_local_level

    def initialize_states_history(self):
        """
        Reinitialize prior, posterior, and smoothed values for hidden states in
        :attr:`~canari.model.Model.states` with empty lists.
        """

        self.states.initialize(self.states_name)

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
            >>> model = Model()
            >>> # If the next analysis starts from the beginning of the time series
            >>> model.set_memory(states=model.states, time_step=0))
            >>> # If the next analysis starts from t = 200
            >>> model.set_memory(states=model.states, time_step=200))
        """

        if time_step == 0:
            self.initialize_states_with_smoother_estimates()
            if self.lstm_net:
                self.lstm_output_history.initialize(self.lstm_net.lstm_look_back_len)
                lstm_states = self.lstm_net.get_lstm_states()
                for key in lstm_states:
                    old_tuple = lstm_states[key]
                    new_tuple = tuple(
                        np.zeros_like(np.array(v)).tolist() for v in old_tuple
                    )
                    lstm_states[key] = new_tuple
                self.lstm_net.set_lstm_states(lstm_states)
        else:
            mu_states_to_set = states.mu_smooth[time_step - 1]
            var_states_to_set = states.var_smooth[time_step - 1]
            self.set_states(mu_states_to_set, var_states_to_set)
            if self.lstm_net:
                mu_lstm_to_set = states.get_mean(
                    states_name=["lstm"], states_type="smooth"
                )
                mu_lstm_to_set = mu_lstm_to_set["lstm"][
                    time_step - self.lstm_net.lstm_look_back_len : time_step
                ]
                std_lstm_to_set = states.get_std(
                    states_name=["lstm"], states_type="smooth"
                )
                std_lstm_to_set = std_lstm_to_set["lstm"][
                    time_step - self.lstm_net.lstm_look_back_len : time_step
                ]
                self.lstm_output_history.mu = mu_lstm_to_set
                self.lstm_output_history.var = std_lstm_to_set**2

    def forward(
        self,
        input_covariates: np.ndarray,
        mu_lstm_pred: Optional[np.ndarray] = None,
        var_lstm_pred: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Make a one-step-ahead prediction. Prediction step in Kalman filter.
        If no `input_covariates`
        for LSTM, use an empty `np.ndarray`. Recall :meth:`~canari.common.forward`
        from :class:`~canari.common`.

        This function is used at one-time-step level.

        Args:
            input_covariates (np.ndarray): Input covariates for LSTM at time `t`.
            mu_lstm_pred (Optional[np.ndarray]): Predicted mean from LSTM at time `t+1`, used when
                dont want LSTM to make predictions, but use LSTM predictions already have.
            var_lstm_pred (Optional[np.ndarray]): Predicted variance from LSTM at time `t+1`, used
                when dont want LSTM to make predictions, but use LSTM predictions already have.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing:

                - :attr:`~canari.model.Model.mu_obs_predict` (np.ndarray):
                    The predictive mean of the observation at `t+1`.
                - :attr:`~canari.model.Model.var_obs_predict` (np.ndarray):
                    The predictive variance of the observation at `t+1`.
                - :attr:`~canari.model.Model.mu_states_prior` (np.ndarray):
                    The prior mean of the hidden state at `t+1`.
                - :attr:`~canari.model.Model.var_states_prior` (np.ndarray):
                    The prior variance of the hidden state at `t+1`.

        """

        # LSTM prediction:
        lstm_states_index = self.get_states_index("lstm")
        if self.lstm_net and mu_lstm_pred is None and var_lstm_pred is None:
            mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                self.lstm_output_history, input_covariates
            )
            mu_lstm_pred, var_lstm_pred = self.lstm_net.forward(
                mu_x=np.float32(mu_lstm_input), var_x=np.float32(var_lstm_input)
            )

        # State-space model prediction:
        mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = common.forward(
            self.mu_states,
            self.var_states,
            self.transition_matrix,
            self.process_noise_matrix,
            self.observation_matrix,
            mu_lstm_pred,
            var_lstm_pred,
            lstm_states_index,
        )

        # Modification after SSM's prediction:
        if "autoregression" in self.states_name:
            mu_states_prior, var_states_prior = self._online_AR_forward_modification(
                mu_states_prior, var_states_prior
            )

        self.mu_states_prior = mu_states_prior
        self.var_states_prior = var_states_prior
        self.mu_obs_predict = mu_obs_pred
        self.var_obs_predict = var_obs_pred

        return mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior

    def backward(
        self,
        obs: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Update step in Kalman filter (for one time step).

        This function is used at one-time-step level. Recall :meth:`~canari.common.backward`
        from :class:`~canari.common`.

        Args:
            obs (float): Observation value.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing:

                - **delta_mu** (np.ndarray):
                    The delta for updating :attr:`~canari.model.Model.mu_states_prior`.
                - **delta_var** (np.ndarray):
                    The delta for updating :attr:`~canari.model.Model.var_states_prior`.
                - :attr:`~canari.model.Model.mu_states_posterior` (np.ndarray):
                    The posterior mean of the hidden states.
                - :attr:`~canari.model.Model.var_states_posterior` (np.ndarray):
                    The posterior variance of the hidden states.
        """

        delta_mu_states, delta_var_states = common.backward(
            obs,
            self.mu_obs_predict,
            self.var_obs_predict,
            self.var_states_prior,
            self.observation_matrix,
        )
        delta_mu_states = np.nan_to_num(delta_mu_states, nan=0.0)
        delta_var_states = np.nan_to_num(delta_var_states, nan=0.0)
        mu_states_posterior = self.mu_states_prior + delta_mu_states
        var_states_posterior = self.var_states_prior + delta_var_states

        if "autoregression" in self.states_name:
            mu_states_posterior, var_states_posterior = (
                self._online_AR_backward_modification(
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
        Apply RTS smoothing equation for a specity timestep. As a result of this function,
        the smoothed estimates for hidden states at the specific time step will be updated in
        :attr:`~canari.model.Model.states`.

        This function is used at one-time-step level. Recall :meth:`~canari.common.rts_smoother`
        from :class:`~canari.common`.

        Args:
            time_step (int): Target smoothing index.
            matrix_inversion_tol (float): Numerical stability threshold.
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

    def forecast(
        self, data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """
        Perform multi-step-ahead forecast over an entire dataset by recursively making
        one-step-ahead predictions, i.e., reapeatly apply the
        Kalman prediction step over multiple time steps.

        This function is used at entire-dataset-level. Recall repeatedly the function
        :meth:`~canari.model.Model.forward` at one-time-step level
        from :class:`~canari.model.Model`.

        Args:
            data (Dict[str, np.ndarray]): A dictionary containing key 'x' as input covariates,
                if exists 'y' (real observations) will not be used.

        Returns:
            Tuple[np.ndarray, np.ndarray, StatesHistory]:
                A tuple containing:

                - **mu_obs_preds** (np.ndarray):
                    The means for forecasts.
                - **std_obs_preds** (np.ndarray):
                    The standard deviations for forecasts.
                - :attr:`~canari.model.Model.states`:
                    The history of hidden states over time.
        """

        mu_obs_preds = []
        std_obs_preds = []

        for x in data["x"]:
            mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = self.forward(
                x
            )

            if self.lstm_net:
                lstm_index = self.get_states_index("lstm")
                self.lstm_output_history.update(
                    mu_states_prior[lstm_index],
                    var_states_prior[lstm_index, lstm_index],
                )

            self._set_posterior_states(mu_states_prior, var_states_prior)
            self._save_states_history()
            self.set_states(mu_states_prior, var_states_prior)
            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)
        return (
            np.array(mu_obs_preds).flatten(),
            np.array(std_obs_preds).flatten(),
            self.states,
        )

    def filter(
        self,
        data: Dict[str, np.ndarray],
        train_lstm: Optional[bool] = True,
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """
        Perform Kalman filter over an entire dataset, i.e., repeatly apply the Kalman prediction and
        update steps over multiple time steps.

        This function is used at entire-dataset-level. Recall repeatedly the function
        :meth:`~canari.model.Model.forward` and :meth:`~canari.model.Model.backward` at
        one-time-step level from :class:`~canari.model.Model`.

        Args:
            data (Dict[str, np.ndarray]): Includes 'x' and 'y'.
            train_lstm (bool): Whether to update LSTM's parameter weights and biases.
                Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray, StatesHistory]:
                A tuple containing:

                - **mu_obs_preds** (np.ndarray):
                    The means for forecasts.
                - **std_obs_preds** (np.ndarray):
                    The standard deviations for forecasts.
                - :attr:`~canari.model.Model.states`:
                    The history of hidden states over time.
        """

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
                lstm_index = self.get_states_index("lstm")
                delta_mu_lstm = np.array(
                    delta_mu_states[lstm_index]
                    / var_states_prior[lstm_index, lstm_index]
                )
                delta_var_lstm = np.array(
                    delta_var_states[lstm_index, lstm_index]
                    / var_states_prior[lstm_index, lstm_index] ** 2
                )
                if train_lstm:
                    self.lstm_net.update_param(
                        np.float32(delta_mu_lstm), np.float32(delta_var_lstm)
                    )
                self.lstm_output_history.update(
                    mu_states_posterior[lstm_index],
                    var_states_posterior[lstm_index, lstm_index],
                )

            self._save_states_history()
            self.set_states(mu_states_posterior, var_states_posterior)
            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)

        return (
            np.array(mu_obs_preds).flatten(),
            np.array(std_obs_preds).flatten(),
            self.states,
        )

    def smoother(self) -> StatesHistory:
        """
        Perform Kalman smoother over an entire time series data, i.e. repeatly apply the
        RTS smoothing equation over multiple time steps.

        This function is used at entire-dataset-level. Recall repeatedly the function
        :meth:`~canari.model.Model.rts_smoother` at
        one-time-step level from :class:`~canari.model.Model`.

        Returns:
            StatesHistory:
                :attr:`~canari.model.Model.states`: The history of hidden states over time.
        """

        num_time_steps = len(self.states.mu_smooth)
        for time_step in reversed(range(0, num_time_steps - 1)):
            self.rts_smoother(time_step)

        return self.states

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
        current_epoch: int,
        white_noise_decay: Optional[bool] = True,
        white_noise_max_std: Optional[float] = 5,
        white_noise_decay_factor: Optional[float] = 0.9,
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """
        Train the :class:`~canari.component.lstm_component.LstmNetwork` component on the provided
        training set, then evaluate on the validation set.
        Optionally apply exponential decay on the white noise standard deviation over epochs.

        At the end of this function, use :class:`~canari.model.Model.set_memory` to set the memory
        to `t=0`.

        Args:
            train_data (Dict[str, np.ndarray]):
                Dictionary with keys `'x'` and `'y'` for training inputs and targets.
            validation_data (Dict[str, np.ndarray]):
                Dictionary with keys `'x'` and `'y'` for validation inputs and targets.
            current_epoch (int): current epoch
            white_noise_decay (bool, optional):
                If True, apply an exponential decay on the white noise standard deviation
                over epochs, if a white noise component exists. Defaults to True.
            white_noise_max_std (float, optional):
                Upper bound on the white-noise standard deviation when decaying.
                Defaults to 5.
            white_noise_decay_factor (float, optional):
                Multiplicative decay factor applied to the white‐noise standard
                deviation each epoch. Defaults to 0.9.

        Returns:
            Tuple[np.ndarray, np.ndarray, StatesHistory]:
                A tuple containing:

                - **mu_obs_preds** (np.ndarray):
                    The means for multi-step-ahead predictions for the validation set.
                - **std_obs_preds** (np.ndarray):
                    The standard deviations for multi-step-ahead predictions for the validation set.
                - :attr:`~canari.model.Model.states`:
                    The history of hidden states over time.
        """

        # Decaying observation's variance
        if white_noise_decay and self.get_states_index("white noise") is not None:
            self._white_noise_decay(
                current_epoch, white_noise_max_std, white_noise_decay_factor
            )
        self.filter(train_data)
        self.smoother()
        mu_validation_preds, std_validation_preds, _ = self.forecast(validation_data)
        # self.set_memory(states=self.states, time_step=0)

        return (
            np.array(mu_validation_preds).flatten(),
            np.array(std_validation_preds).flatten(),
            self.states,
        )

    def early_stopping(
        self,
        evaluate_metric: float,
        current_epoch: int,
        max_epoch: int,
        mode: Optional[str] = "min",
        patience: Optional[int] = 20,
        skip_epoch: Optional[int] = 5,
    ) -> Tuple[bool, int, float, list]:
        """
        Check and apply early stopping based on a monitored metric.

        This method records `evaluate_metric` at each epoch,
        if there is an improvement on it,
        update :attr:`.early_stop_metric`, :attr:`.early_stop_init_mu_states`,
        :attr:`.early_stop_init_var_states`, :attr:`.early_stop_lstm_param`,
        and :attr:`.optimal_epoch`.

        Sets the `stop_training` to **True** if :attr:`.optimal_epoch` = `max_epoch`, or
        (`current_epoch` - :attr:`.optimal_epoch`)>= `patience`.

        When `stop_training` is **True**, set :attr:`.mu_states` = :attr:`.early_stop_init_mu_states`,
        :attr:`.var_states` = :attr:`.early_stop_init_var_states`, and set LSTM parameters to
        :attr:`.early_stop_lstm_param`.

        Args:
            current_epoch (int):
                Current epoch
            max_epoch (int):
                Maximum number of epoch
            evaluate_metric (float):
                Current metric value for this epoch.
            mode (Optional[str]):
                Direction for early stopping: 'min' (default).
            patience (Optional[int]):
                Number of epochs without improvement before stopping. Defaults to 20.
            skip_epoch (Optional[int]):
                Number of initial epochs to ignore when looking for improvements. Defaults to 5.

        Returns:
            Tuple[bool, int, float, List[float]]:
                - stop_training: True if training stops.
                - optimal_epoch: Epoch index of when having best metric.
                - early_stop_metric: Best evaluate_metric. .
                - early_stop_metric_history: History of `evaluate_metric` at all epochs.
        """

        if current_epoch == 0:
            self.early_stop_metric = -np.inf if mode == "max" else np.inf

        self.early_stop_metric_history.append(evaluate_metric)

        # Check for improvement
        improved = False
        improved = current_epoch > skip_epoch and (
            (mode == "max" and evaluate_metric > self.early_stop_metric)
            or (mode == "min" and evaluate_metric < self.early_stop_metric)
        )

        # Update metric and parameters if improved
        if improved:
            self.early_stop_metric = evaluate_metric
            self.early_stop_lstm_param = copy.copy(self.lstm_net.state_dict())
            self.early_stop_init_mu_states = copy.copy(self.mu_states)
            self.early_stop_init_var_states = copy.copy(self.var_states)
            self.optimal_epoch = current_epoch

        # Check stop condition
        if current_epoch == max_epoch - 1 or (
            current_epoch > skip_epoch - 1
            and current_epoch - self.optimal_epoch >= patience
        ):
            self.stop_training = True

        # Load best parameters
        if self.stop_training:
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

    def generate_time_series(
        self,
        num_time_series: int,
        num_time_steps: int,
        sample_from_lstm_pred=True,
        time_covariates=None,
        time_covariate_info=None,
        add_anomaly=False,
        anomaly_mag_range=None,
        anomaly_begin_range=None,
        anomaly_type="trend",
    ) -> np.ndarray:
        """
        Generate synthetic time series data based on the model components,
        with optional injected synthetic anomalies.

        Args:
            num_time_series (int):
                Number of independent series to generate.
            num_time_steps (int):
                Number of timesteps per generated series.
            sample_from_lstm_pred (bool, optional):
                If False, zeroes out LSTM‐derived variance so that generation
                ignores LSTM uncertainty. Defaults to True.
            time_covariates (np.ndarray of shape (num_time_steps, cov_dim), optional):
                Time‐varying covariates to include in generation. If provided,
                these will be standardized using `time_covariate_info` and
                passed through the model each step. Defaults to None.
            time_covariate_info (dict, optional):
                Required if `time_covariates` is not None. Must contain:
                - "initial_time_covariate" (np.ndarray): the starting covariate vector
                - "mu" (np.ndarray): means for standardization
                - "std" (np.ndarray): standard deviations for standardization
            add_anomaly (bool, optional):
                Whether to inject a synthetic anomaly into each series.
                Defaults to False.
            anomaly_mag_range (tuple of float, optional):
                (min, max) range for random anomaly magnitudes.
                Required if `add_anomaly=True`. Defaults to None.
            anomaly_begin_range (tuple of int, optional):
                (min, max) range of timestep indices at which anomaly may start.
                Required if `add_anomaly=True`. Defaults to None.
            anomaly_type (str, optional):
                Type of injected anomaly: "trend": a growing linear drift after anomaly
                starts, "level": a constant shift after anomaly starts. Defaults to "trend".

        Returns:
            Tuple[np.ndarray, np.ndarray, List[float], List[int]]:
                - **generated series** (np.ndarray):
                    Generated series with the shape (num_time_series, num_time_steps).
                - **input_covariates** (np.ndarray):
                    The input covariates used.
                - **anomaly magnitudes** (List[float]):
                    Anomaly magnitudes per series.
                - **anomaly start timesteps** (List[float]):
                    Anomaly start timesteps per series.

        """

        time_series_all = []
        anm_mag_all = []
        anm_begin_all = []
        mu_states_temp = copy.deepcopy(self.mu_states)
        var_states_temp = copy.deepcopy(self.var_states)

        # Prepare time covariates
        if time_covariates is not None:
            initial_time_covariate = time_covariate_info["initial_time_covariate"]
            input_covariates = self._prepare_covariates_generation(
                initial_time_covariate, num_time_steps, time_covariates
            )
            input_covariates = normalizer.standardize(
                input_covariates, time_covariate_info["mu"], time_covariate_info["std"]
            )
        else:
            input_covariates = np.empty((num_time_steps, 0))

        # Get LSTM initializations
        if "lstm" in self.states_name:
            if (
                self.lstm_output_history.mu is not None
                and self.lstm_output_history.var is not None
            ):
                lstm_output_history_mu_temp = copy.deepcopy(self.lstm_output_history.mu)
                lstm_output_history_var_temp = copy.deepcopy(
                    self.lstm_output_history.var
                )
                lstm_output_history_exist = True
            else:
                lstm_output_history_exist = False

            lstm_cell_states = self.lstm_net.get_lstm_states()

        for _ in range(num_time_series):
            one_time_series = []

            if "lstm" in self.states_name:
                # Reset lstm cell states
                self.lstm_net.set_lstm_states(lstm_cell_states)
                # Reset lstm output history
                if lstm_output_history_exist:
                    self.lstm_output_history.mu = copy.deepcopy(
                        lstm_output_history_mu_temp
                    )
                    self.lstm_output_history.var = copy.deepcopy(
                        lstm_output_history_var_temp
                    )
                else:
                    self.lstm_output_history.initialize(
                        self.lstm_net.lstm_look_back_len
                    )

            # Get the anomaly features
            if add_anomaly:
                anomaly_mag = np.random.uniform(
                    anomaly_mag_range[0], anomaly_mag_range[1]
                )
                anomaly_time = np.random.randint(
                    anomaly_begin_range[0], anomaly_begin_range[1]
                )
                anm_mag_all.append(anomaly_mag)
                anm_begin_all.append(anomaly_time)

            for i, x in enumerate(input_covariates):
                _, _, mu_states_prior, var_states_prior = self.forward(x)

                if "lstm" in self.states_name:
                    lstm_index = self.states_name.index("lstm")
                    if not sample_from_lstm_pred:
                        var_states_prior[lstm_index, :] = 0
                        var_states_prior[:, lstm_index] = 0

                state_sample = np.random.multivariate_normal(
                    mu_states_prior.flatten(), var_states_prior
                ).reshape(-1, 1)

                if "lstm" in self.states_name:
                    self.lstm_output_history.update(
                        state_sample[lstm_index],
                        np.zeros_like(var_states_prior[lstm_index, lstm_index]),
                    )

                obs_gen = self.observation_matrix @ state_sample
                obs_gen = obs_gen.item()

                if add_anomaly:
                    if i > anomaly_time:
                        if anomaly_type == "trend":
                            obs_gen += anomaly_mag * (i - anomaly_time)  # LT anomaly
                        elif anomaly_type == "level":
                            obs_gen += anomaly_mag  # LL anomaly
                self.set_states(state_sample, np.zeros_like(var_states_prior))
                one_time_series.append(obs_gen)

            self.set_states(mu_states_temp, var_states_temp)
            time_series_all.append(one_time_series)

        # Change lstm output history back to the original
        if "lstm" in self.states_name:
            if lstm_output_history_exist:
                self.lstm_output_history.mu = copy.deepcopy(lstm_output_history_mu_temp)
                self.lstm_output_history.var = copy.deepcopy(
                    lstm_output_history_var_temp
                )
            else:
                self.lstm_output_history.initialize(self.lstm_net.lstm_look_back_len)
            self.lstm_net.set_lstm_states(lstm_cell_states)

        return np.array(time_series_all), input_covariates, anm_mag_all, anm_begin_all
