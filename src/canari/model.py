"""
Hybrid LSTM-SSM model that combines Bayesian Long-short Term Memory (LSTM) Neural Networks
and State-Space Models (SSM).

This model supports a flexible architecture where multiple `component`
are assembled to define a structured state-space model.

On time series data, this model can:

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
import pandas as pd
from pytagi import Normalizer as normalizer
from pytagi.nn import OutputUpdater
from canari.component.base_component import BaseComponent
from canari import common
from canari.data_struct import LstmOutputHistory, StatesHistory, OutputHistory
from canari.common import GMA
from canari.data_process import DataProcess


class Model:
    """
    `Model` class for the Hybrid LSTM/SSM model.

    Args:
        *components (BaseComponent): One or more instances of classes derived from
                            :class:`~canari.component.base_component.BaseComponent`.

    Examples:
        >>> from canari.component import LocalTrend, Periodic, WhiteNoise
        >>> from canari import Model
        >>> # Components
        >>> local_trend = LocalTrend(mu_states=[1,0.5], var_states=[1,0.5])
        >>> periodic = Periodic(mu_states=[1,1],var_states=[2,2],period=52)
        >>> residual = WhiteNoise(std_error=0.04168)
        >>> # Define model
        >>> model = Model(local_trend, periodic, residual)

    Attributes:
        components (Dict[str, BaseComponent]):
            Dictionary to save model components' configurations.
        num_states (int):
            Number of hidden states.
        states_names (list[str]):
            Names of hidden states.
        mu_states (np.ndarray):
            Mean vector for the hidden states :math:`X_{t|t}` at the time step `t`.
        var_states (np.ndarray):
            Covariance matrix for the hidden states :math:`X_{t|t}` at the time step `t`.
        mu_states_prior (np.ndarray):
            Prior mean vector for the hidden states :math:`X_{t+1|t}` at the time step `t+1`.
        var_states_prior (np.ndarray):
            Prior covariance matrix for the hidden states :math:`X_{t+1|t}` at the time step `t+1`.
        mu_states_posterior (np.ndarray):
            Posteriror mean vector for the hidden states :math:`X_{t+1|t+1}` at the time step `t+1`.
            In case of missing data (NaN observation), it will have the same values
            as :attr:`mu_states_prior`.
        var_states_posterior (np.ndarray):
            Posteriror covariance matrix for the hidden states :math:`X_{t+1|t+1}` at the time
            step `t+1`. In case of missing data (NaN observation), it will have the same values
            as :attr:`var_states_prior`.
        states (StatesHistory):
            Container for storing prior, posterior, and smoothed values of hidden states over time.
        mu_obs_predict (np.ndarray):
            Means for observation predictions at a time step `t+1`.
        var_obs_predict (np.ndarray):
            Variances for observation predictions at a time step `t+1`.
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
        lstm_states_history (list):
            Container for saving the history for LSTM's hidden and cell states for all time steps.

        # Early stopping attributes: only being used when training a :class:`~canari.component.lstm_component.LstmNetwork` component.

        early_stop_metric (float):
            Best value associated with the metric being monitored.
        early_stop_metric_history (List[float]):
            Logged history of metric values across epochs.
        early_stop_lstm_param (Dict):
            LSTM's weight and bias parameters at the optimal epoch for :class:`pytagi.Sequential`.
        early_stop_states (np.ndarray):
            :attr:`states` at the optimal epoch.
        early_stop_lstm_states (np.ndarray):
           :attr:`lstm_states_history` at the optimal epoch
        optimal_epoch (int):
            Epoch at which the metric being monitored was best.
        stop_training (bool):
            Flag indicating whether training has been stopped due to
            early stopping or by reaching maximum number of epoch.

        # Optimization attribute
        metric_optim (float): metric used for optimization in :class:`~canari.model_optimizer`

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
            f"{component.component_name} {i}": component
            for i, component in enumerate(components)
        }
        self._initialize_model()
        self.states = StatesHistory()
        self.output_history = OutputHistory()

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
        self.output_col = []
        self.input_col = []
        self.output_lag_col = []

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

        # LSTM-related attributes TODO: put lstm_output_history and lstm_states_history into lstm_net
        self.lstm_net = None
        self.lstm_output_history = LstmOutputHistory()
        self.lstm_states_history = []

        # Autoregression-related attributes
        self.mu_W2bar = None
        self.var_W2bar = None
        self.mu_W2_prior = None
        self.var_W2_prior = None

        # Noise related attribute
        self.sched_sigma_v = None
        self._var_v2bar_prior = None
        self._mu_v2bar_tilde = None
        self._var_v2bar_tilde = None
        self._cov_v2bar_tilde = None

        # Early stopping attributes
        self.early_stop_metric = None
        self.early_stop_metric_history = []
        self.early_stop_lstm_param = None
        self.early_stop_states = None
        self.early_stop_lstm_states = None
        self.early_stop_lstm_output_mu = None
        self.early_stop_lstm_output_var = None
        self.optimal_epoch = 0
        self._current_epoch = 0
        self.stop_training = False

        # Metric for optimization
        self.metric_optim = None

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
        Initialize and configure an LSTM network if there is a LstmNetwork component is used.
        """

        lstm_component = next(
            (
                component
                for component in self.components.values()
                if "lstm" in component.component_name
            ),
            None,
        )
        if lstm_component:
            self.lstm_net = lstm_component.initialize_lstm_network()
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
                if "autoregression" in component.component_name
            ),
            None,
        )

        if "AR_error" in self.states_name:
            self.mu_W2bar = autoregression_component.mu_states[-1]
            self.var_W2bar = autoregression_component.var_states[-1]

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

    def _exponential_cov_states(
        self,
        cov_states,
        mu_states_prior,
        var_states_prior,
        mu_states_posterior,
        var_states_posterior,
    ) -> float:
        """
        The cross covariance matrix between `exp` and `scaled exp` with other states are non linear.
        This function computes the correct cross-covariance for `exp`
        and `scaled exp` with other states.

        Args:
            cov_states (np.ndarray): cross-covariances between two hidden states
                                        at two consecutive time steps.
            mu_states_prior (np.ndarray): Prior mean of the states.
            var_states_prior (np.ndarray): Prior variance-covariance matrix.
            mu_states_posterior (np.ndarray): Posterior mean of the states.
            var_states_posterior (np.ndarray): Posterior variance-covariance matrix.

        Returns:
            Tuple[np.ndarray]: Updated (cov_states).
        """
        latent_level_index = self.get_states_index("latent level")
        exp_index = self.get_states_index("exp")
        exp_scale_factor_index = self.get_states_index("exp scale factor")
        scaled_exp_index = self.get_states_index("scaled exp")

        mag_norm_exp_prior = (
            var_states_posterior[exp_index, latent_level_index]
            / var_states_posterior[latent_level_index, latent_level_index]
        )
        mag_norm_exp_trans = (
            var_states_prior[exp_index, latent_level_index]
            / var_states_prior[latent_level_index, latent_level_index]
        )
        skip_index1 = {exp_index, scaled_exp_index}
        for other_component_index in range(len(mu_states_posterior)):
            if other_component_index in skip_index1:
                continue
            cov_states[exp_index, other_component_index] = (
                mag_norm_exp_prior
                * cov_states[latent_level_index, other_component_index]
            )
            cov_states[other_component_index, exp_index] = (
                mag_norm_exp_trans
                * cov_states[other_component_index, latent_level_index]
            )

        cov_states[exp_index, exp_index] = (
            mag_norm_exp_prior
            * mag_norm_exp_trans
            * cov_states[latent_level_index, latent_level_index]
        )

        skip_index2 = {scaled_exp_index}
        for other_component_index in range(len(mu_states_posterior)):
            if other_component_index in skip_index2:
                continue
            cov_states[scaled_exp_index, other_component_index] = (
                cov_states[exp_scale_factor_index, other_component_index]
                * mu_states_posterior[exp_index].item()
                + cov_states[exp_index, other_component_index]
                * mu_states_posterior[exp_scale_factor_index].item()
            )
            cov_states[other_component_index, scaled_exp_index] = (
                cov_states[other_component_index, exp_scale_factor_index]
                * mu_states_prior[exp_index].item()
                + cov_states[other_component_index, exp_index]
                * mu_states_prior[exp_scale_factor_index].item()
            )

        cov_states[scaled_exp_index, scaled_exp_index] = (
            cov_states[exp_scale_factor_index, exp_scale_factor_index]
            * cov_states[exp_index, exp_index].item()
            + cov_states[exp_scale_factor_index, exp_index]
            * cov_states[exp_index, exp_scale_factor_index]
            + cov_states[exp_scale_factor_index, exp_scale_factor_index]
            * mu_states_posterior[exp_index].item()
            * mu_states_prior[exp_index].item()
            + cov_states[exp_scale_factor_index, exp_index]
            * mu_states_posterior[exp_index].item()
            * mu_states_prior[exp_scale_factor_index].item()
            + cov_states[exp_index, exp_scale_factor_index]
            * mu_states_posterior[exp_scale_factor_index].item()
            * mu_states_prior[exp_index].item()
            + cov_states[exp_index, exp_index]
            * mu_states_posterior[exp_scale_factor_index].item()
            * mu_states_prior[exp_scale_factor_index].item()
        )
        return cov_states

    def _update_exp_and_scaled_exp(
        self, mu_states, var_states, var_states_behind, method
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply forward path exponential moment transformations.

        Updates prior state means and variances based on the exponential model.
        The modification is applied after that `latent level`, `latent trend` and `exp scale factor`
        are updated by the transition matrix.
        After that,the closed form solutions to compute the prior distribution of `exp`
        from `latent level` and `latent trend`.
        GMA is also applied to `exp scale factor` and `exp` to get the prior distribution
        of `scaled exp`.
        These are used during the forward pass when exponential components are present.

        Args:
            mu_states_prior (np.ndarray): Prior mean vector of the states.
            var_states_prior (np.ndarray): Prior variance-covariance matrix of the states.
            var_states (np.ndarray): Variance-covariance matrix before the linear update
                                        of the states

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Updated (mu_states_prior, var_states_prior, mu_obs_predict, var_obs_predict).
        """
        latent_level_index = self.get_states_index("latent level")
        latent_trend_index = self.get_states_index("latent trend")
        exp_scale_factor_index = self.get_states_index("exp scale factor")
        exp_index = self.get_states_index("exp")
        scaled_exp_index = self.get_states_index("scaled exp")
        mu_obs_predict = []
        var_obs_predict = []

        mu_ll = np.asarray(mu_states[latent_level_index]).item()
        var_ll = np.asarray(var_states[latent_level_index, latent_level_index]).item()

        mu_states[exp_index] = np.exp(-mu_ll + 0.5 * var_ll) - 1

        var_states[exp_index, exp_index] = np.exp(-2 * mu_ll + var_ll) * (
            np.exp(var_ll) - 1
        )

        var_states[latent_level_index, exp_index] = -var_ll * np.exp(
            -mu_ll + 0.5 * var_ll
        )

        var_states[exp_index, latent_level_index] = var_states[
            latent_level_index, exp_index
        ]

        if method == "forward":
            skip_index = {latent_level_index, latent_trend_index, exp_index}
            var_states[latent_trend_index, exp_index] = -np.exp(
                -mu_ll + 0.5 * var_ll
            ) * (
                var_states_behind[latent_trend_index, latent_trend_index]
                + var_states_behind[latent_level_index, latent_trend_index]
            )
            var_states[exp_index, latent_trend_index] = var_states[
                latent_trend_index, exp_index
            ]
        elif method in {"backward", "smoother"}:
            skip_index = {latent_level_index, exp_index}

        magnitud_normal_space_exponential_space = (
            var_states[exp_index, latent_level_index]
            / var_states[latent_level_index, latent_level_index]
        )
        for other_component_index in range(len(mu_states)):
            if other_component_index in skip_index:
                continue
            cov_other_component_index = (
                magnitud_normal_space_exponential_space
                * var_states[latent_level_index, other_component_index]
            )
            var_states[exp_index, other_component_index] = cov_other_component_index
            var_states[other_component_index, exp_index] = cov_other_component_index

        mu_states, var_states = GMA(
            mu_states,
            var_states,
            index1=exp_scale_factor_index,
            index2=exp_index,
            replace_index=scaled_exp_index,
        ).get_results()

        if method == "forward":
            mu_obs_predict, var_obs_predict = common.calc_observation(
                mu_states, var_states, self.observation_matrix
            )

        return (
            mu_states,
            var_states,
            mu_obs_predict,
            var_obs_predict,
        )

    def _online_AR_forward_modification(self, mu_states_prior, var_states_prior):
        """
        Apply forward path autoregressive (AR) moment transformations.

        Updates prior state means and variances based on the autoregressive error model,
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
            var_states_prior[W2bar_index, W2bar_index] = self.var_W2bar.item()

            # # From W2bar to W2
            self.mu_W2_prior = self.mu_W2bar
            self.var_W2_prior = 3 * self.var_W2bar + 2 * self.mu_W2bar**2
            mu_states_prior[W2_index] = self.mu_W2_prior
            var_states_prior[W2_index, W2_index] = self.var_W2_prior.item()

            # # From W2 to W
            mu_states_prior[ar_error_index] = 0
            var_states_prior[ar_error_index, :] = np.zeros_like(
                var_states_prior[ar_error_index, :]
            )
            var_states_prior[:, ar_error_index] = np.zeros_like(
                var_states_prior[:, ar_error_index]
            )
            var_states_prior[ar_error_index, ar_error_index] = self.mu_W2bar.item()
            var_states_prior[ar_error_index, ar_index] = self.mu_W2bar.item()
            var_states_prior[ar_index, ar_error_index] = self.mu_W2bar.item()
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
            var_states_posterior[W2_index, W2_index] = var_W2_posterior.item()

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
            var_states_posterior[W2bar_index, W2bar_index] = self.var_W2bar.item()

            self.process_noise_matrix[ar_index, ar_index] = self.mu_W2bar.item()

        return mu_states_posterior, var_states_posterior

    def _BAR_backward_modification(
        self, mu_states_posterior, var_states_posterior
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        BAR backward modification.

        Apply backward BAR moment updates during state-space filtering.

        Computes the constrained posterior distribution of AR state according to the bounding
        coefficient gamma when it is provided.

        Args:
            mu_states_posterior (np.ndarray): Posterior mean vector of the states.
            var_states_posterior (np.ndarray): Posterior variance-covariance matrix of the states.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated (mu_states_posterior, var_states_posterior).
        """

        ar_index = self.get_states_index("autoregression")
        bar_index = self.get_states_index("bounded autoregression")

        mu_AR = mu_states_posterior[ar_index].item()
        var_AR = var_states_posterior[ar_index, ar_index].item()
        cov_AR = var_states_posterior[ar_index, :]

        bar_component = next(
            (
                component
                for component in self.components.values()
                if "bounded autoregression" in component.component_name
            ),
            None,
        )

        bound = bar_component.gamma * np.sqrt(
            bar_component.std_error**2 / (1 - bar_component.phi**2)
        )

        l_bar = mu_AR + bound

        mu_L = (
            l_bar * common.norm_cdf(l_bar / np.sqrt(var_AR))
            + np.sqrt(var_AR) * common.norm_pdf(l_bar / np.sqrt(var_AR))
            - bound
        )
        var_L = (
            (l_bar**2 + var_AR) * common.norm_cdf(l_bar / np.sqrt(var_AR))
            + l_bar * np.sqrt(var_AR) * common.norm_pdf(l_bar / np.sqrt(var_AR))
            - (mu_L + bound) ** 2
        )

        u_bar = -mu_AR + bound
        mu_U = (
            -u_bar * common.norm_cdf(u_bar / np.sqrt(var_AR))
            - np.sqrt(var_AR) * common.norm_pdf(u_bar / np.sqrt(var_AR))
            + bound
        )
        var_U = (
            (u_bar**2 + var_AR) * common.norm_cdf(u_bar / np.sqrt(var_AR))
            + u_bar * np.sqrt(var_AR) * common.norm_pdf(u_bar / np.sqrt(var_AR))
            - (-mu_U + bound) ** 2
        )

        mu_states_posterior[bar_index] = mu_L + mu_U - mu_AR
        cov_bar = cov_AR * (
            common.norm_cdf(l_bar / np.sqrt(var_AR))
            + common.norm_cdf(u_bar / np.sqrt(var_AR))
            - 1
        )
        var_bar = (
            var_L
            + (mu_L - mu_AR) ** 2
            + var_U
            + (mu_U - mu_AR) ** 2
            - (mu_states_posterior[bar_index] - mu_AR) ** 2
            - var_AR
        )
        var_states_posterior[bar_index, :] = cov_bar
        var_states_posterior[:, bar_index] = cov_bar
        var_states_posterior[bar_index, bar_index] = np.maximum(
            var_bar, 1e-8
        ).item()  # For numerical stability

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

        if num_generated_samples >= 0:
            # Forward generation: offsets are [0, 1, 2, ..., n-1]
            covariates_generation = np.arange(0, num_generated_samples).reshape(-1, 1)
        else:
            # Reverse generation: offsets are [-n, ..., -2, -1]
            n_abs = abs(num_generated_samples)
            covariates_generation = np.arange(-n_abs, 0).reshape(-1, 1)

        def safe_mod(x, base):
            return (x % base + base) % base

        for time_cov in time_covariates:
            if time_cov == "hour_of_day":
                covariates_generation = safe_mod(
                    initial_covariate + covariates_generation, 24
                )
            elif time_cov == "day_of_week":
                covariates_generation = safe_mod(
                    initial_covariate + covariates_generation, 7
                )
            elif time_cov == "day_of_year":
                covariates_generation = safe_mod(
                    initial_covariate + covariates_generation, 365
                )
            elif time_cov == "week_of_year":
                covariates_generation = safe_mod(
                    initial_covariate + covariates_generation, 52
                )
            elif time_cov == "month_of_year":
                covariates_generation = safe_mod(
                    initial_covariate + covariates_generation, 12
                )
            elif time_cov == "quarter_of_year":
                covariates_generation = safe_mod(
                    initial_covariate + covariates_generation, 4
                )
        return covariates_generation

    def _estim_hete_noise(
        self,
        mu_v2bar_prior: np.ndarray,
        var_v2bar_prior: np.ndarray,
    ):
        """
        Estimate variance for the white noise hidden states using LSTM with AGVI
        """

        if self.get_states_index("heteroscedastic noise") is not None:
            noise_index = self.get_states_index("heteroscedastic noise")
            self._mu_v2bar_tilde = np.exp(mu_v2bar_prior + 0.5 * var_v2bar_prior)
            self._var_v2bar_tilde = np.exp(2 * mu_v2bar_prior + var_v2bar_prior) * (
                np.exp(var_v2bar_prior) - 1
            )
            self._cov_v2bar_tilde = var_v2bar_prior * self._mu_v2bar_tilde
            self._var_v2bar_prior = var_v2bar_prior
            self.process_noise_matrix[noise_index, noise_index] = (
                self._mu_v2bar_tilde.item()
            )
            self.sched_sigma_v = self._mu_v2bar_tilde**0.5
        else:
            raise ValueError("In the LSTM component, model_noise should be True. ")

    def _delta_hete_noise(self):
        """
        Estimate delta for v2bar which is the second output of the LSTM network
        """

        noise_index = self.get_states_index("heteroscedastic noise")
        mu_noise_posterior = self.mu_states_posterior[noise_index]
        var_noise_posterior = self.var_states_posterior[noise_index, noise_index]

        mu_v2_posterior = mu_noise_posterior**2 + var_noise_posterior
        var_v2_posterior = (
            2 * var_noise_posterior**2 + 4 * var_noise_posterior * mu_noise_posterior**2
        )

        mu_v2_prior = self._mu_v2bar_tilde
        var_v2_prior = 3 * self._var_v2bar_tilde + 2 * self._mu_v2bar_tilde**2

        k = self._var_v2bar_tilde / var_v2_prior

        delta_mu_v2bar_tilde = k * (mu_v2_posterior - mu_v2_prior)
        delta_var_v2bar_tilde = k**2 * (var_v2_posterior - var_v2_prior)

        jcb = self._cov_v2bar_tilde / self._var_v2bar_tilde
        delta_mu_v2bar = jcb * delta_mu_v2bar_tilde / self._var_v2bar_prior
        delta_var_v2bar = jcb * delta_var_v2bar_tilde * jcb / self._var_v2bar_prior**2

        return delta_mu_v2bar, delta_var_v2bar

    def update_lstm_states_history(self, index: int, last_step: int):
        """
        Store LSTM states at specific time steps. Currently only save states the first and last time steps of
        the train, validation, and test sets. For other time steps, save None.

        Args:
            index (int): time step to save the lstm states at
        """

        if index == 0 or index == last_step:
            _lstm_states = self.lstm_net.get_lstm_states()
            self.lstm_states_history.append(_lstm_states)
        else:
            self.lstm_states_history.append(None)

    def white_noise_decay(
        self, epoch: int, white_noise_max_std: float, white_noise_decay_factor: float
    ):
        """
        Apply exponential decay to white noise standard deviation over epochs, and modify
        the variance for the white noise component in
        :attr:`~canari.model.Model.process_noise_matrix`.
        This decaying noise structure is intended to improve the training performance
        of TAGI-LSTM.

        Args:
            epoch (int): Current training epoch.
            white_noise_max_std (float): Maximum allowed noise std.
            white_noise_decay_factor (float): Factor controlling decay rate.
        """

        min_noise_std = 0
        scheduled_sigma_v = white_noise_max_std * np.exp(
            -white_noise_decay_factor * epoch
        )

        if self.get_states_index("white noise") is not None:
            noise_index = self.get_states_index("white noise")
            white_noise_component = next(
                (
                    component
                    for component in self.components.values()
                    if "white noise" in component.component_name
                ),
                None,
            )
            min_noise_std = white_noise_component.std_error
        elif self.get_states_index("heteroscedastic noise") is not None:
            noise_index = self.get_states_index("heteroscedastic noise")
        else:
            noise_index = None

        if noise_index is not None:
            if scheduled_sigma_v < min_noise_std:
                scheduled_sigma_v = min_noise_std
            self.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

        self.sched_sigma_v = scheduled_sigma_v

    def save_states_history(self):
        """
        Save current prior, posterior hidden states, and cross-covariaces between hidden states
        at two consecutive time steps for later use in Kalman's smoother.
        """

        self.states.mu_prior.append(self.mu_states_prior)
        self.states.var_prior.append(self.var_states_prior)
        self.states.mu_posterior.append(self.mu_states_posterior)
        self.states.var_posterior.append(self.var_states_posterior)
        cov_states = self.var_states @ self.transition_matrix.T
        if "exp" in self.states_name:
            cov_states = self._exponential_cov_states(
                cov_states,
                self.mu_states_prior,
                self.var_states_prior,
                self.mu_states_posterior,
                self.var_states_posterior,
            )

        self.states.cov_states.append(cov_states)
        self.states.mu_smooth.append(self.mu_states_posterior)
        self.states.var_smooth.append(self.var_states_posterior)

    def pretraining_filter(self, train_data: dict):
        """
        Run exactly `lstm_look_back_len` dummy steps through the LSTM so that the
        `lstm_output_history` gets filled with `lstm_look_back_len` predictions.
        Assumes `self.lstm_net` and `self.lstm_output_history` already exist.
        """

        # set lstm to training mode
        self.lstm_net.train()

        # initialize the smoothing buffer for SLTM
        if self.lstm_net.smooth:
            self.lstm_net.num_samples = self.lstm_net.lstm_infer_len + len(
                train_data["y"]
            )

        # prepare lookback covariates
        lookback_covariates = self._generate_look_back_covariates(train_data)

        self.lstm_output_history.initialize(self.lstm_net.lstm_look_back_len)
        # reset LSTM states to zeros
        lstm_states = self.lstm_net.get_lstm_states()
        for key in lstm_states:
            old_tuple = lstm_states[key]
            new_tuple = tuple(np.zeros_like(np.array(v)).tolist() for v in old_tuple)
            lstm_states[key] = new_tuple
        self.lstm_net.set_lstm_states(lstm_states)
        device = self.lstm_net.device

        dummy_mu_obs = np.array([np.nan], dtype=np.float32)
        dummy_var_obs = np.array([0.0], dtype=np.float32)

        out_updater = OutputUpdater(device)

        for i in range(self.lstm_net.lstm_infer_len):
            dummy_covariates = lookback_covariates[i]
            mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                self.lstm_output_history, dummy_covariates
            )

            mu_lstm_pred, var_lstm_pred = self.lstm_net.forward(
                mu_x=np.float32(mu_lstm_input),
                var_x=np.float32(var_lstm_input),
            )

            # Heteroscedastic noise
            if self.lstm_net.model_noise:
                mu_lstm_pred = mu_lstm_pred[0::2]
                var_lstm_pred = var_lstm_pred[0::2]
                dummy_mu_obs = np.array([np.nan, np.nan], dtype=np.float32)
                dummy_var_obs = np.array([0, 0], dtype=np.float32)

            out_updater.update(
                output_states=self.lstm_net.output_z_buffer,
                mu_obs=dummy_mu_obs,
                var_obs=dummy_var_obs,
                delta_states=self.lstm_net.input_delta_z_buffer,
            )

            self.lstm_net.backward()
            self.lstm_net.step()

            self.lstm_output_history.update(mu_lstm_pred, var_lstm_pred)

    def _generate_look_back_covariates(self, train_data):
        """
        Generate standardized look-back covariates for the LSTM by re-using
        Model._prepare_covariates_generation.
        Supports only time covariates, otherwise Nan is used for other covariates.
        """
        # Get indices and look-back length
        train_idx = 0
        inferred_len = self.lstm_net.lstm_infer_len

        # Gather covariate column names and count
        cov_names = train_data["cov_names"]
        n_cov = len(cov_names)

        # Gather scaling constants
        scale_const_mean = train_data["scale_const_mean"]
        scale_const_std = train_data["scale_const_std"]

        # Initialize dummy array with NaNs
        dummy = np.full((inferred_len, n_cov), np.nan)

        # Handle time covariates
        time_covs = train_data["time_covariates"] or []
        for tc in time_covs:
            if tc not in cov_names:
                continue
            col_idx = cov_names.index(tc)
            init_val = train_data["x"][train_idx][col_idx]
            mu = scale_const_mean[col_idx + 1]
            std = scale_const_std[col_idx + 1]
            init_val = init_val * std + mu
            raw = self._prepare_covariates_generation(
                np.rint(init_val),
                num_generated_samples=-inferred_len,
                time_covariates=[tc],
            )
            normed = (raw - mu) / (std + 1e-10)
            dummy[:, col_idx] = normed[:, 0]

        return dummy

    def update_lstm_output_history(self, mu_states: np.ndarray, var_states: np.ndarray):
        """
        Update the rolling history of LSTM output means and variances with the mu_states and var_states

        Args:
            mu_states (np.ndarray): mean value to be updated to LSTM output history.
            var_states (np.ndarray): variance value to be updated to LSTM output history.
        """

        lstm_index = self.get_states_index("lstm")
        self.lstm_output_history.update(
            mu_states[lstm_index],
            var_states[lstm_index, lstm_index],
        )

    def get_dict(self, time_step: Optional[int] = None) -> dict:
        """
        Export model attributes into a serializable dictionary.

        Args:
            time_step (Optional[int]): the time step to get the model and memory at.
                                        If None export the model with the current memory.
                                        Defaults to None.
        Returns:
            dict: Serializable model dictionary containing neccessary attributes.

        Examples:
            >>> saved_dict = model.get_dict()
        """

        save_dict = {}
        save_dict["components"] = self.components
        save_dict["states_name"] = self.states_name
        memory = self.get_memory(time_step=time_step)
        save_dict["memory"] = memory
        if self.lstm_net:
            save_dict["lstm_network_params"] = self.lstm_net.state_dict()

        return save_dict

    @staticmethod
    def load_dict(save_dict: dict):
        """
        Reconstruct a model instance from a saved dictionary.

        Args:
            save_dict (dict): Dictionary containing saved model structure and parameters.

        Returns:
            Model: An instance of :class:`~canari.model.Model` generated from the input dictionary.

        Examples:
            >>> saved_dict = model.get_dict()
            >>> loaded_model = Model.load_dict(saved_dict)
        """

        components = list(save_dict["components"].values())
        model = Model(*components)

        # TODO: trick: run skf.filter to initialize skf.model["norm_norm"] states
        if "cov_names" in save_dict:
            dummy_input = np.full((len(save_dict["cov_names"]),), np.nan)
            model.forward(input_covariates=dummy_input)

        model.set_memory(memory=save_dict["memory"])
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

    def auto_initialize_baseline_states(self, data: np.ndarray):
        """
        Automatically assign initial means and variances for baseline hidden states (level,
        trend, and acceleration) from input data using time series decomposition
        defined in :meth:`~canari.data_process.DataProcess.decompose_data`.

        Args:
            data (np.ndarray): Time series data.

        Examples:
            >>> train_set, val_set, test_set, all_data = dp.get_splits()
            >>> model.auto_initialize_baseline_states(train_data["y"][0 : 52])
        """

        trend, slope, _, _ = DataProcess.decompose_data(data.flatten())

        for i, _state_name in enumerate(self.states_name):
            if _state_name == "level":
                self.mu_states[i] = trend[0]
                if self.var_states[i, i] == 0:
                    self.var_states[i, i] = 1e-2
            elif _state_name == "trend":
                self.mu_states[i] = slope
                if self.var_states[i, i] == 0:
                    self.var_states[i, i] = 1e-2
            elif _state_name == "acceleration":
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
        if "level" in self.states_name and hasattr(self, "_mu_local_level"):
            local_level_index = self.get_states_index("level")
            self.mu_states[local_level_index] = self._mu_local_level

    def initialize_states_history(self):
        """
        Reinitialize prior, posterior, and smoothed values for hidden states in
        :attr:`~canari.model.Model.states` as well as :attr:`lstm_states_history` with empty lists.
        """

        self.states.initialize(self.states_name)
        self.lstm_states_history = []

    def get_memory(self, time_step: Optional[int] = None) -> dict:
        """
        Get memory which includes :attr:`mu_states`, :attr:`var_states`, :attr:`lstm_output_history`,
        and **lstm_states** of :attr:`lstm_net`. If `time_step` is provided, obtain the memory at that time step.
        Otherwise, obtain the memory at the current time step.

        Args:
            time_step (Optional[int]): time step to obtain the memory

        Returns:
            Dict
        """

        memory = {}
        lstm_states = None
        lstm_output_history_mu = None
        lstm_output_history_var = None

        if time_step is None:  # save the current memory of model
            mu_states = self.mu_states.copy()
            var_states = self.var_states.copy()
            if self.lstm_net:
                lstm_states = self.lstm_net.get_lstm_states()
                lstm_output_history_mu = self.lstm_output_history.mu.copy()
                lstm_output_history_var = self.lstm_output_history.var.copy()

        elif time_step == 0:  # save model's memory at t=0
            # mu, var states
            mu_states = self.states.mu_smooth[time_step].copy()
            var_states = np.diag(np.diag(self.states.var_smooth[time_step])).copy()
            if "level" in self.states_name and hasattr(self, "_mu_local_level"):
                local_level_index = self.get_states_index("level")
                mu_states[local_level_index] = self._mu_local_level

            if self.lstm_net:
                if self.lstm_net.smooth:
                    lstm_states = self.lstm_states_history[time_step].copy()
                    lstm_output_history_mu = self.lstm_output_history.mu.copy()
                    lstm_output_history_var = self.lstm_output_history.var.copy()
                else:
                    # lstm states
                    lstm_states = self.lstm_net.get_lstm_states()
                    for key in lstm_states:
                        old_tuple = lstm_states[key]
                        new_tuple = tuple(
                            np.zeros_like(np.array(v)).tolist() for v in old_tuple
                        )
                        lstm_states[key] = new_tuple

                    # lstm output history
                    lstm_output_history = LstmOutputHistory()
                    lstm_output_history.initialize(self.lstm_net.lstm_look_back_len)
                    lstm_output_history_mu = lstm_output_history.mu.copy()
                    lstm_output_history_var = lstm_output_history.var.copy()

        else:  # save model's memory at t = time_step
            # mu, var states
            mu_states = self.states.mu_smooth[time_step].copy()
            var_states = self.states.var_smooth[time_step].copy()
            if self.lstm_net:
                # lstm states
                lstm_states = self.lstm_states_history[time_step].copy()

                # lstm output history
                mu_lstm_to_set = self.states.get_mean(
                    states_name="lstm", states_type="smooth"
                )
                lstm_output_history_mu = mu_lstm_to_set[
                    time_step - self.lstm_net.lstm_look_back_len + 1 : time_step + 1
                ]
                std_lstm_to_set = self.states.get_std(
                    states_name="lstm", states_type="smooth"
                )
                lstm_output_history_var = (
                    std_lstm_to_set[
                        time_step - self.lstm_net.lstm_look_back_len + 1 : time_step + 1
                    ]
                    ** 2
                )

        memory["mu_states"] = mu_states
        memory["var_states"] = var_states
        memory["lstm_states"] = lstm_states
        memory["lstm_output_history_mu"] = lstm_output_history_mu
        memory["lstm_output_history_var"] = lstm_output_history_var

        return memory

    def set_memory(
        self,
        time_step: Optional[int] = None,
        memory: Optional[dict] = None,
    ):
        """
        Set memory which includes :attr:`~canari.model.Model.mu_states`,
        :attr:`~canari.model.Model.var_states`, :attr:`~canari.model.Model.lstm_output_history`
        and **lstm_states** in :attr:`lstm_net` with smoothed estimates to a specific time step.
        This is to prepare for the next analysis by ensuring the continuity of these variables,
        e.g., if the next analysis starts from time step `t`, should set the memory to the
        time step `t-1`.

        Args:
            time_step (Optional[int]): Time step to set the memory.
            memory (Optional[dict]): memory to be set.

        Examples:
            >>> # If the next analysis starts from the beginning of the time series
            >>> model.set_memory(time_step=0)
            >>> # If the next analysis starts from t = 200
            >>> model.set_memory(time_step=199)
        """

        if time_step is not None:
            memory = self.get_memory(time_step=time_step)

        self.set_states(memory["mu_states"], memory["var_states"])
        if self.lstm_net:
            self.lstm_output_history.set(
                memory["lstm_output_history_mu"], memory["lstm_output_history_var"]
            )
            self.lstm_net.set_lstm_states(memory["lstm_states"])

    def forward(
        self,
        input_covariates: Optional[np.ndarray] = None,
        var_input_covariates: Optional[np.ndarray] = None,
        mu_lstm_pred: Optional[np.ndarray] = None,
        var_lstm_pred: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Make a one-step-ahead prediction using the prediction step of the Kalman filter.
        If no `input_covariates` for LSTM, use an empty `np.ndarray`.
        Recall :meth:`~canari.common.forward` from :class:`~canari.common`.

        This function is used at the one-time-step level.

        Args:
            input_covariates (Optional[np.ndarray]): Input covariates for LSTM at time `t`.
            mu_lstm_pred (Optional[np.ndarray]): Predicted mean from LSTM at time `t+1`, used when
                we dont want LSTM to make predictions, but use LSTM predictions already have.
            var_lstm_pred (Optional[np.ndarray]): Predicted variance from LSTM at time `t+1`, used
                when we dont want LSTM to make predictions, but use LSTM predictions already have.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing:

                - :attr:`mu_obs_predict` (np.ndarray):
                    The predictive mean of the observation at `t+1`.
                - :attr:`var_obs_predict` (np.ndarray):
                    The predictive variance of the observation at `t+1`.
                - :attr:`mu_states_prior` (np.ndarray):
                    The prior mean of the hidden state at `t+1`.
                - :attr:`var_states_prior` (np.ndarray):
                    The prior variance of the hidden state at `t+1`.

        """

        # LSTM prediction:
        lstm_states_index = self.get_states_index("lstm")
        if self.lstm_net and mu_lstm_pred is None and var_lstm_pred is None:
            if var_input_covariates is not None:
                mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                    self.lstm_output_history, input_covariates, var_input_covariates
                )
            else:
                mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                    self.lstm_output_history, input_covariates
                )
            mu_lstm_pred, var_lstm_pred = self.lstm_net.forward(
                mu_x=np.float32(mu_lstm_input), var_x=np.float32(var_lstm_input)
            )

            # Heteroscedastic noise
            if self.lstm_net.model_noise:
                mu_v2bar_prior = mu_lstm_pred[1::2]
                var_v2bar_prior = var_lstm_pred[1::2]
                mu_lstm_pred = mu_lstm_pred[0::2]
                var_lstm_pred = var_lstm_pred[0::2]
                self._estim_hete_noise(mu_v2bar_prior, var_v2bar_prior)

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

        if "exp" in self.states_name:
            mu_states_prior, var_states_prior, mu_obs_pred, var_obs_pred = (
                self._update_exp_and_scaled_exp(
                    mu_states_prior, var_states_prior, self.var_states, "forward"
                )
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
        Update step in the Kalman filter for one time step.

        This function is used at the one-time-step level. Recall :meth:`~canari.common.backward`
        from :class:`~canari.common`.

        Args:
            obs (float): Observation value.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                A tuple containing:

                - **delta_mu** (np.ndarray):
                    The delta for updating :attr:`mu_states_prior`.
                - **delta_var** (np.ndarray):
                    The delta for updating :attr:`var_states_prior`.
                - :attr:`mu_states_posterior` (np.ndarray):
                    The posterior mean of the hidden states.
                - :attr:`var_states_posterior` (np.ndarray):
                    The posterior variance of the hidden states.
        """

        delta_mu_states, delta_var_states = common.backward(
            obs,
            self.mu_obs_predict,
            self.var_obs_predict,
            self.var_states_prior,
            self.observation_matrix,
        )

        # TODO: check replacing Nan could create problems
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

        if "bounded autoregression" in self.states_name:
            mu_states_posterior, var_states_posterior = self._BAR_backward_modification(
                mu_states_posterior, var_states_posterior
            )

        if "exp" in self.states_name:
            mu_states_posterior, var_states_posterior, *_ = (
                self._update_exp_and_scaled_exp(
                    mu_states_posterior, var_states_posterior, 0, "backward"
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

    def update_lstm_param(
        self,
        delta_mu_states: np.ndarray,
        delta_var_states: np.ndarray,
    ):
        """
        Obtain the posteriors for the LSTM neural network's parameters in :attr:`lstm_net` by adding `delta`
        to their priors.

        Args:
            delta_mu_states (np.ndarray): Delta mean for states.
            delta_var_states (np.ndarray): Delta variance states.
        """

        lstm_index = self.get_states_index("lstm")
        delta_mu_lstm = np.array(
            delta_mu_states[lstm_index] / self.var_states_prior[lstm_index, lstm_index]
        )
        delta_var_lstm = np.array(
            delta_var_states[lstm_index, lstm_index]
            / self.var_states_prior[lstm_index, lstm_index] ** 2
        )

        if self.lstm_net.model_noise:
            delta_mu_v2bar, delta_var_v2bar = self._delta_hete_noise()
            delta_mu_lstm = np.append(delta_mu_lstm, delta_mu_v2bar)
            delta_var_lstm = np.append(delta_var_lstm, delta_var_v2bar)

        self.lstm_net.set_delta_z(
            np.array(delta_mu_lstm, dtype=np.float32),
            np.array(delta_var_lstm, dtype=np.float32),
        )
        self.lstm_net.backward()
        self.lstm_net.step()

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
            matrix_inversion_tol (Optional[float]): Numerical stability threshold for matrix
                                            pseudoinversion (pinv). Defaults to 1E-12.
            tol_type (Optional[str]): Tolerance type, "relative" or "absolute".
                                        Defaults to "relative".
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

        if "exp" in self.states_name:

            (
                self.states.mu_smooth[time_step],
                self.states.var_smooth[time_step],
                *_,
            ) = self._update_exp_and_scaled_exp(
                self.states.mu_smooth[time_step],
                self.states.var_smooth[time_step],
                0,
                "smoother",
            )

    def forecast(
        self, data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """
        Perform multi-step-ahead forecast over an entire dataset by recursively making
        one-step-ahead predictions, i.e., reapeatly apply the
        Kalman prediction step over multiple time steps.

        This function is used at the entire-dataset-level. Recall repeatedly the function
        :meth:`forward` at one-time-step level from :class:`~canari.model.Model`.

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
                - :attr:`states`:
                    The history of hidden states over time.

        Examples:
            >>> mu_preds_val, std_preds_val, states = model.forecast(val_set)
        """

        mu_obs_preds = []
        std_obs_preds = []

        # set lstm to eval mode
        if self.lstm_net:
            self.lstm_net.eval()

        for index, x in enumerate(data["x"]):
            mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = self.forward(
                x
            )

            if self.lstm_net:
                self.update_lstm_states_history(index, last_step=len(data["y"]) - 1)
                self.update_lstm_output_history(mu_states_prior, var_states_prior)

            # Store variables
            self._set_posterior_states(mu_states_prior, var_states_prior)
            self.save_states_history()
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
        Run the Kalman filter over an entire dataset, i.e., repeatly apply the Kalman prediction and
        update steps over multiple time steps.

        This function is used at the entire-dataset-level. Recall repeatedly the function
        :meth:`forward` and :meth:`backward` at
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
                - :attr:`states`:
                    The history of hidden states over time.

        Examples:
            >>> mu_preds_train, std_preds_train, states = model.filter(train_set)
        """

        mu_obs_preds = []
        std_obs_preds = []
        self.initialize_states_history()

        # set lstm to train mode
        if self.lstm_net and train_lstm:
            self.lstm_net.train()

        for index, (x, y) in enumerate(zip(data["x"], data["y"])):

            mu_obs_pred, var_obs_pred, *_ = self.forward(x)
            (
                delta_mu_states,
                delta_var_states,
                mu_states_posterior,
                var_states_posterior,
            ) = self.backward(y)

            # Update LSTM parameters
            if self.lstm_net:
                self.update_lstm_states_history(index, last_step=len(data["y"]) - 1)

                if train_lstm:
                    self.update_lstm_param(delta_mu_states, delta_var_states)
                self.update_lstm_output_history(
                    mu_states_posterior, var_states_posterior
                )

            # Store variables
            self.save_states_history()
            self.set_states(mu_states_posterior, var_states_posterior)
            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)

        return (
            np.array(mu_obs_preds).flatten(),
            np.array(std_obs_preds).flatten(),
            self.states,
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
            tol_type (Optional[str]): Tolerance type, "relative" or "absolute".
                                        Defaults to "relative".

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

        if self.lstm_net and self.lstm_net.smooth and self.lstm_net.num_samples > 1:
            mu_zo_smooth, var_zo_smooth = self.lstm_net.smoother()
            mu_sequence = mu_zo_smooth[: self.lstm_net.lstm_infer_len]
            var_sequence = var_zo_smooth[: self.lstm_net.lstm_infer_len]
            mu_sequence = mu_sequence[-self.lstm_net.lstm_look_back_len :]
            var_sequence = var_sequence[-self.lstm_net.lstm_look_back_len :]
            self.lstm_output_history.mu = mu_sequence
            self.lstm_output_history.var = var_sequence

            # set the smoothed lstm_states at the first time step
            self.lstm_states_history[0] = self.lstm_net.get_lstm_states(
                self.lstm_net.lstm_infer_len - 1
            )

        return self.states

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
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

                - **mu_validation_preds** (np.ndarray):
                    The means for multi-step-ahead predictions for the validation set.
                - **std_validation_preds** (np.ndarray):
                    The standard deviations for multi-step-ahead predictions for the validation set.
                - :attr:`~canari.model.Model.states`:
                    The history of hidden states over time.

        Examples:
            >>> mu_preds_val, std_preds_val, states = model.lstm_train(train_data=train_set,validation_data=val_set)
        """

        # Decaying observation's variance
        if white_noise_decay:
            for noise_type in ("white noise", "heteroscedastic noise"):
                if self.get_states_index(noise_type) is not None:
                    self.white_noise_decay(
                        self._current_epoch,
                        white_noise_max_std,
                        white_noise_decay_factor,
                    )
                    break

        if self.lstm_net.smooth:
            self.pretraining_filter(train_data)

        self.filter(train_data)
        mu_validation_preds, std_validation_preds, _ = self.forecast(validation_data)
        self.smoother()
        self.set_memory(time_step=0)
        self._current_epoch += 1

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
        Apply early stopping based on the `evaluate_metric` when training a LSTM neural network.

        This method records `evaluate_metric` at each epoch,
        if there is an improvement on it,
        update :attr:`.early_stop_metric`, :attr:`.early_stop_lstm_param`,
        :attr:`.early_stop_states`, :attr:`.early_stop_lstm_states`,
        and :attr:`.optimal_epoch`.

        Sets the `stop_training` to **True** if :attr:`.optimal_epoch` = `max_epoch`, or
        (`current_epoch` - :attr:`.optimal_epoch`)>= `patience`.

        When `stop_training` is **True**, set :attr:`.states` = :attr:`.early_stop_states`,
        :attr:`.lstm_states_history` = :attr:`.early_stop_lstm_states`, set LSTM parameters to
        :attr:`.early_stop_lstm_param`, and set `memory` to `time_step=0`

        Args:
            current_epoch (int):
                Current epoch
            max_epoch (int):
                Maximum number of epochs
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

        Examples:
            >>> model.early_stopping(evaluate_metric=mse, current_epoch=1, max_epoch=50)
        """

        if current_epoch == 0:
            self.early_stop_metric = -np.inf if mode == "max" else np.inf

        self.early_stop_metric_history.append(evaluate_metric)

        # Check for improvement
        improved = False
        improved = current_epoch >= skip_epoch and (
            (mode == "max" and evaluate_metric > self.early_stop_metric)
            or (mode == "min" and evaluate_metric < self.early_stop_metric)
        )

        # Update metric and parameters if improved
        if improved:
            self.early_stop_metric = evaluate_metric
            self.early_stop_lstm_param = copy.copy(self.lstm_net.state_dict())
            self.early_stop_states = copy.copy(self.states)
            self.early_stop_lstm_states = copy.copy(self.lstm_states_history)
            self.early_stop_lstm_output_mu = self.lstm_output_history.mu.copy()
            self.early_stop_lstm_output_var = self.lstm_output_history.var.copy()
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
            self.states = copy.copy(self.early_stop_states)
            self.lstm_states_history = copy.copy(self.early_stop_lstm_states)
            self.lstm_output_history.mu = copy.copy(self.early_stop_lstm_output_mu)
            self.lstm_output_history.var = copy.copy(self.early_stop_lstm_output_var)
            self.set_memory(time_step=0)
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
        with optional synthetic anomaly injection.

        Args:
            num_time_series (int):
                Number of independent series to generate.
            num_time_steps (int):
                Number of timesteps per generated series.
            sample_from_lstm_pred (bool, optional):
                If False, zeroes out LSTM-derived variance so that the generation
                ignores the LSTM uncertainty. Defaults to True.
            time_covariates (np.ndarray of shape (num_time_steps, cov_dim), optional):
                Time-varying covariates to include in generation. If provided,
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
