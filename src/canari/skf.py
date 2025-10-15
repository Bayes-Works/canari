"""
Switching Kalman Filter (SKF) for detecting regime changes in time series data. It takes as inputs
two instances of :class:`~canari.model`, one model is used to model a normal regime, the other is
used to model an abnormal one. At each time step, SKF estimates the probability of each model.

On time series data, this model can:

    - Train its Bayesian LSTM network component from the normal model.
    - Detect regime changes (anomalies) and provide probabilities of regime switch.
    - Decompose orginal time serires data into unobserved hidden states. Provide mean values and the associate uncertainties for these hidden states.

"""

from typing import Tuple, Dict, Optional
import copy
import numpy as np
from pytagi import metric
from canari.model import Model
from canari import common
from canari.data_struct import StatesHistory
from canari.data_process import DataProcess


class SKF:
    """
    `SKF` class for Switching Kalman Filter.

    Args:
        norm_model (Model): Model representing normal behavior.
        abnorm_model (Model): Model representing abnormal behavior.
        std_transition_error (float): Std deviation of transition error.
        norm_to_abnorm_prob (float): Transition probability from normal to abnormal.
        abnorm_to_norm_prob (float): Transition probability from abnormal to normal.
        norm_model_prior_prob (float): Prior probability of the normal model.
        conditional_likelihood (bool): Whether to use conditional log-likelihood. Defaults to False.

    Examples:
        >>> from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise
        >>> from canari import Model, SKF
        >>> # Components
        >>> local_trend = LocalTrend()
        >>> local_acceleration = LocalAcceleration()
        >>> lstm_network = LstmNetwork(
                look_back_len=10,
                num_features=2,
                num_layer=1,
                num_hidden_unit=50,
                device="cpu",
                manual_seed=1,
            )
        >>> residual = WhiteNoise(std_error=0.05)
        >>> # Define model
        >>> normal_model = Model(local_trend, lstm_network, residual)
        >>> abnormal_model = Model(local_acceleration, lstm_network, residual)
        >>> skf = SKF(
                norm_model=normal_model,
                abnorm_model=abnormal_model,
                std_transition_error=1e-4,
                norm_to_abnorm_prob=1e-4,
            )

    Attributes:
        model (dict): Dictionary containing 4 instances of :class:`~canari.model`, i.e., 4
            transition models including
            'norm_norm': transition from normal to normal;
            'norm_abnorm': transition from normal to abnormal;
            'abnorm_norm': transition from abnormal to normal;
            'abnorm_abnorm': transition from abnormal to abnormal;
        num_states (int):
            Number of hidden states.
        states_names (list[str]):
            Names of hidden states.
        mu_states_prior (np.ndarray):
            Prior mean vector for the marginal hidden states :math:`X_{t+1|t}` at the time step `t+1`.
        var_states_prior (np.ndarray):
            Prior covariance matrix for the marginal hidden states :math:`X_{t+1|t}`
            at the time step `t+1`.
        mu_states_posterior (np.ndarray):
            Posteriror mean vector for the marginal hidden states :math:`X_{t+1|t+1}`
            at the time step `t+1`.
        var_states_posterior (np.ndarray):
            Posteriror covariance matrix for the marginal hidden states :math:`X_{t+1|t+1}` at the time
            step `t+1`.
        states (StatesHistory):
            Container for storing prior, posterior, and smoothed values of marginal hidden states
            over time.
        transition_prob (dict): Transition probability matrix: 'norm_norm', 'norm_abnorm',
                                'abnorm_norm', 'abnorm_abnorm'.
        marginal_prob (dict): Current marginal probability for 'normal' and 'abnormal' at time `t`.
        filter_marginal_prob_history (dict): Filter marginal probability history for 'normal'
                                            and 'abnormal' over time.
        smooth_marginal_prob_history (dict): Smoother marginal probability history for 'normal'
                                             and 'abnormal' over time.
        norm_to_abnorm_prob (float): Transition probability from normal to abnormal.
        abnorm_to_norm_prob (float): Transition probability from abnormal to normal.
        norm_model_prior_prob (float): Prior probability of the normal model.
        conditional_likelihood (bool): Whether to use conditional log-likelihood. Defaults to False.

        # LSTM-related attributes: only being used when a :class:`~canari.component.lstm_component.LstmNetwork` component is found.

        lstm_net (:class:`pytagi.Sequential`):
            It is a :class:`pytagi.Sequential` instance. LSTM neural network linked from
            'norm_norm' of :attr:`~canari.skf.SKF.model`, if LSTM component presents.

        lstm_output_history (LstmOutputHistory):
            Container for saving a rolling history of LSTM output over a fixed look-back window. Linked
            from 'norm_norm' of :attr:`~canari.skf.SKF.model`, if LSTM component presents.

        # Early stopping attributes: only being used when training a :class:`~canari.component.lstm_component.LstmNetwork` component.

        early_stop_metric (float):
            Best value of the metric being monitored.
        early_stop_metric_history (List[float]):
            Logged history of metric values across epochs.
        optimal_epoch (int):
            Epoch at which the monitored metric was best.
        stop_training (bool):
            Flag indicating whether training has been stopped due to
            early stopping or reaching the maximum number of epochs.
    """

    def __init__(
        self,
        norm_model: Model,
        abnorm_model: Model,
        std_transition_error: Optional[float] = 0.0,
        norm_to_abnorm_prob: Optional[float] = 1e-4,
        abnorm_to_norm_prob: Optional[float] = 0.1,
        norm_model_prior_prob: Optional[float] = 0.99,
        conditional_likelihood: Optional[bool] = False,
    ):
        """
        Initialization
        """

        self.std_transition_error = std_transition_error
        self.norm_to_abnorm_prob = norm_to_abnorm_prob
        self.abnorm_to_norm_prob = abnorm_to_norm_prob
        self.norm_model_prior_prob = norm_model_prior_prob
        self.conditional_likelihood = conditional_likelihood
        self.model = self._transition()
        self.states = StatesHistory()
        self._initialize_attributes()
        self._initialize_model(norm_model, abnorm_model)

    @staticmethod
    def _prob_history():
        """
        Create a dictionary to save marginal probability (normal and abnormal) history over time.
        """
        return {
            "norm": [],
            "abnorm": [],
        }

    @staticmethod
    def _transition():
        """
        Create a dictionary for transitions:
        'norm_norm': transition model from normal to normal;
        'norm_abnorm': from normal to abnormal;
        'abnorm_norm': from abnormal to normal;
        'abnorm_abnorm': from abnormal to abnormal;
        """
        return {
            "norm_norm": None,
            "abnorm_abnorm": None,
            "norm_abnorm": None,
            "abnorm_norm": None,
        }

    @staticmethod
    def _marginal():
        """
        Create a dictionary for mariginal: normal and abnormal
        """
        return {
            "norm": None,
            "abnorm": None,
        }

    def _initialize_attributes(self):
        """
        Initialize all attributes.
        """

        # General attributes
        self.num_states = 0
        self.states_name = []

        # SKF-related attributes
        self._memory_init = None
        self.mu_states_prior = None
        self.var_states_prior = None
        self.mu_states_posterior = None
        self.var_states_posterior = None

        self.transition_coef = self._transition()
        self.filter_marginal_prob_history = self._prob_history()
        self.smooth_marginal_prob_history = self._prob_history()
        self.marginal_list = {"norm", "abnorm"}

        self.transition_prob = self._transition()
        self.transition_prob["norm_norm"] = 1 - self.norm_to_abnorm_prob
        self.transition_prob["norm_abnorm"] = self.norm_to_abnorm_prob
        self.transition_prob["abnorm_norm"] = self.abnorm_to_norm_prob
        self.transition_prob["abnorm_abnorm"] = 1 - self.abnorm_to_norm_prob

        self.marginal_prob = self._marginal()
        self.marginal_prob["norm"] = self.norm_model_prior_prob
        self.marginal_prob["abnorm"] = 1 - self.norm_model_prior_prob

        # LSTM-related attributes
        self.lstm_net = None

        # Early stopping attributes
        self.stop_training = False
        self.optimal_epoch = 0
        self.early_stop_metric_history = []
        self.early_stop_metric = None

        # Optimization
        self.metric_optim = {}

    def _initialize_model(self, norm_model: Model, abnorm_model: Model):
        """Initialize transition models and link SKF to these new models.

        This method creates four transition-specific models (normal→normal, abnormal→abnormal,
        normal→abnormal, abnormal→normal).

        Args:
            norm_model (Model): Model for the normal regime.
            abnorm_model (Model): Model for the abnormal regime.

        Returns:
            None
        """

        self._create_transition_model(
            norm_model,
            abnorm_model,
        )
        self._link_skf_to_model()

    @staticmethod
    def _create_compatible_models(soure_model, target_model) -> None:
        """Pad and align two models so they share the same hidden state dimensions and names.

        When source_model has fewer states than target_model, new rows/columns of zeros
        are inserted into its state vectors and matrices at the appropriate indices.
        Also returns the list of newly added state names.

        Args:
            source_model (Model): Model to be padded.
            target_model (Model): Model whose state space defines the target dimension.

        Returns:
            source_model (Model): The padded source model.
            target_model (Model): The (potentially updated) target model.
            states_diff (list[str]): Names of states added to source_model.
        """

        pad_row = np.zeros((soure_model.num_states)).flatten()
        pad_col = np.zeros((target_model.num_states)).flatten()
        states_diff = []
        for i, state in enumerate(target_model.states_name):
            if state not in soure_model.states_name:
                soure_model.mu_states = common.pad_matrix(
                    soure_model.mu_states, i, pad_row=np.zeros(1)
                )
                soure_model.var_states = common.pad_matrix(
                    soure_model.var_states, i, pad_row, pad_col
                )
                soure_model.transition_matrix = common.pad_matrix(
                    soure_model.transition_matrix, i, pad_row, pad_col
                )
                soure_model.process_noise_matrix = common.pad_matrix(
                    soure_model.process_noise_matrix, i, pad_row, pad_col
                )
                soure_model.observation_matrix = common.pad_matrix(
                    soure_model.observation_matrix, i, pad_col=np.zeros(1)
                )
                soure_model.num_states += 1
                soure_model.states_name.insert(i, state)
                states_diff.append(state)

        if "white noise" in soure_model.states_name:
            index_noise = soure_model.states_name.index("white noise")
            target_model.process_noise_matrix[index_noise, index_noise] = (
                soure_model.process_noise_matrix[index_noise, index_noise]
            )
        return soure_model, target_model, states_diff

    def _create_transition_model(
        self,
        norm_model: Model,
        abnorm_model: Model,
    ):
        """Build 4 transition models and store them in self.model.

        Copies of the input normal and abnormal models are made to represent:
        - norm_norm: stay in normal regime
        - abnorm_abnorm: stay in abnormal regime
        - norm_abnorm: transition from normal to abnormal (with added transition noise)
        - abnorm_norm: transition from abnormal to normal

        Args:
            norm_model (Model): Model for the normal regime.
            abnorm_model (Model): Model for the abnormal regime.

        Returns:
            None
        """

        # Create transitional model
        norm_norm = copy.deepcopy(norm_model)
        norm_norm.lstm_net = norm_model.lstm_net
        abnorm_abnorm = copy.deepcopy(abnorm_model)
        norm_norm, abnorm_abnorm, states_diff = self._create_compatible_models(
            norm_norm, abnorm_abnorm
        )
        abnorm_norm = copy.deepcopy(norm_norm)
        norm_abnorm = copy.deepcopy(abnorm_abnorm)

        # Add transition noise to norm_abnorm.process_noise_matrix
        index_states_diff = norm_norm.get_states_index(states_diff[-1])
        norm_abnorm.process_noise_matrix[index_states_diff, index_states_diff] = (
            self.std_transition_error**2
        )

        # Store transitional models in a dictionary
        self.model["norm_norm"] = norm_norm
        self.model["abnorm_abnorm"] = abnorm_abnorm
        self.model["norm_abnorm"] = norm_abnorm
        self.model["abnorm_norm"] = abnorm_norm

    def _link_skf_to_model(self):
        """Attach SKF attributes (state count, names, LSTM) from 'norm_norm' model.

        Copies the number of states, state names, and LSTM network/reference to
        self.num_states, self.states_name, self.lstm_net, and self.lstm_output_history.

        Returns:
            None
        """

        self.num_states = self.model["norm_norm"].num_states
        self.states_name = self.model["norm_norm"].states_name
        if self.model["norm_norm"].lstm_net is not None:
            self.lstm_net = self.model["norm_norm"].lstm_net

    def _set_same_states_transition_models(self):
        """
        Synchronize all transition models to the same hidden-state initialization as
        in 'norm_norm' from :attr:`.model`.

        Returns:
            None
        """

        for transition_model in self.model.values():
            transition_model.set_states(
                self.model["norm_norm"].mu_states, self.model["norm_norm"].var_states
            )

    def _initialize_smoother(self):
        """
        Initialize smoothed hidden states at the final time step.

        Sets the mu_smooth and var_smooth in the last time step to the mixture of posterior
        hidden states from self.model['norm_norm'] and self.model['abnorm_abnorm'].

        Returns:
            None
        """

        self.model["norm_abnorm"].states.mu_smooth[-1] = (
            self.model["abnorm_abnorm"].states.mu_posterior[-1].copy()
        )
        self.model["norm_abnorm"].states.var_smooth[-1] = (
            self.model["abnorm_abnorm"].states.var_posterior[-1].copy()
        )
        self.model["abnorm_norm"].states.mu_smooth[-1] = (
            self.model["norm_norm"].states.mu_posterior[-1].copy()
        )
        self.model["abnorm_norm"].states.var_smooth[-1] = (
            self.model["norm_norm"].states.var_posterior[-1].copy()
        )

        self.states.mu_smooth[-1], self.states.var_smooth[-1] = common.gaussian_mixture(
            self.model["norm_norm"].states.mu_posterior[-1],
            self.model["norm_norm"].states.var_posterior[-1],
            self.filter_marginal_prob_history["norm"][-1],
            self.model["abnorm_abnorm"].states.mu_posterior[-1],
            self.model["abnorm_abnorm"].states.var_posterior[-1],
            self.filter_marginal_prob_history["abnorm"][-1],
        )

    def _save_states_history(self):
        """
        Save current prior, posterior hidden states for 4 transition models as well as
        the marginal model.
        """

        for transition_model in self.model.values():
            transition_model.save_states_history()

        self.states.mu_prior.append(self.mu_states_prior)
        self.states.var_prior.append(self.var_states_prior)
        self.states.mu_posterior.append(self.mu_states_posterior)
        self.states.var_posterior.append(self.var_states_posterior)
        self.states.mu_smooth.append(self.mu_states_posterior)
        self.states.var_smooth.append(self.var_states_posterior)

    def _compute_transition_likelihood(
        self,
        obs: float,
        mu_pred_transit: Dict[str, np.ndarray],
        var_pred_transit: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute the likelihoods under each transition hypothesis given the observation.

        If obs is NaN, returns likelihood=1 for all transitions. Otherwise:
        - When conditional_likelihood=True, adds white-noise realizations to the mean and
          averages the exponentiated log-likelihoods.

        Args:
            obs (float): Observation.
            mu_pred_transit (dict): Predicted means for each transition model.
            var_pred_transit (dict): Predicted variances for each transition model.

        Returns:
            dict[str, float]: Likelihood for each transition key.
        """

        epsilon = 1e-30
        transition_likelihood = self._transition()
        if np.isnan(obs):
            for transit in transition_likelihood:
                transition_likelihood[transit] = 1
        else:
            if self.conditional_likelihood:
                num_noise_realization = 10
                white_noise_index = self.states_name.index("white noise")
                var_obs_error = self.model["norm_norm"].process_noise_matrix[
                    white_noise_index, white_noise_index
                ]
                noise = np.random.normal(
                    0, var_obs_error**0.5, (num_noise_realization, 1)
                )

                for transit in transition_likelihood:
                    transition_likelihood[transit] = np.mean(
                        np.exp(
                            metric.log_likelihood(
                                mu_pred_transit[transit] + noise,
                                obs,
                                (var_pred_transit[transit] - var_obs_error) ** 0.5,
                            )
                        )
                    )
            else:
                for transit in transition_likelihood:
                    transition_likelihood[transit] = np.maximum(
                        np.exp(
                            metric.log_likelihood(
                                mu_pred_transit[transit],
                                obs,
                                var_pred_transit[transit] ** 0.5,
                            )
                        ),
                        epsilon,
                    )
        return transition_likelihood

    def _get_smooth_states_transition(
        self,
        time_step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve smoothed state vectors for all transition models at a given time.

        Args:
            time_step (int): Index of time step for which to fetch smoothed states.

        Returns:
            mu_states_transit (dict[str, np.ndarray]): Smoothed means per transition.
            var_states_transit (dict[str, np.ndarray]): Smoothed variances per transition.
        """

        mu_states_transit = self._transition()
        var_states_transit = self._transition()

        for transit, transition_model in self.model.items():
            mu_states_transit[transit] = transition_model.states.mu_smooth[time_step]
            var_states_transit[transit] = transition_model.states.var_smooth[time_step]

        return (
            mu_states_transit,
            var_states_transit,
        )

    def _collapse_states(
        self,
        mu_states_transit: np.ndarray,
        var_states_transit: np.ndarray,
        state_type: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Collapse state distributions across transition models into marginals and combined.

        Performs two levels of Gaussian mixtures:
        1) Mix origin→arrival pairs for each regime to get marginal means/vars.
        2) Mix the regime marginals with their probabilities.

        Args:
            mu_states_transit (dict[str, np.ndarray]): Predicted means per transition.
            var_states_transit (dict[str, np.ndarray]): Predicted variances per transition.
            state_type (str): 'prior', 'posterior', or 'smooth' to select mixture keys.

        Returns:
            Tuple containing:
            - mu_combined (np.ndarray): Combined state means.
            - var_combined (np.ndarray): Combined state variances.
            - mu_marginal (dict[str, np.ndarray]): Regime-specific marginal means.
            - var_marginal (dict[str, np.ndarray]): Regime-specific marginal variances.
        """

        mu_states_marginal = self._marginal()
        var_states_marginal = self._marginal()

        if state_type == "smooth":
            norm_keys = ("norm_norm", "norm_abnorm")
            abnorm_keys = ("abnorm_norm", "abnorm_abnorm")
        else:
            norm_keys = ("norm_norm", "abnorm_norm")
            abnorm_keys = ("norm_abnorm", "abnorm_abnorm")

        # Retrieve states for norm mixture
        mu_norm_1 = mu_states_transit[norm_keys[0]]
        var_norm_1 = var_states_transit[norm_keys[0]]
        mu_norm_2 = mu_states_transit[norm_keys[1]]
        var_norm_2 = var_states_transit[norm_keys[1]]

        mu_states_marginal["norm"], var_states_marginal["norm"] = (
            common.gaussian_mixture(
                mu_norm_1,
                var_norm_1,
                self.transition_coef[norm_keys[0]],
                mu_norm_2,
                var_norm_2,
                self.transition_coef[norm_keys[1]],
            )
        )

        # Retrieve states for abnorm mixture
        mu_abnorm_1 = mu_states_transit[abnorm_keys[0]]
        var_abnorm_1 = var_states_transit[abnorm_keys[0]]
        mu_abnorm_2 = mu_states_transit[abnorm_keys[1]]
        var_abnorm_2 = var_states_transit[abnorm_keys[1]]

        mu_states_marginal["abnorm"], var_states_marginal["abnorm"] = (
            common.gaussian_mixture(
                mu_abnorm_1,
                var_abnorm_1,
                self.transition_coef[abnorm_keys[0]],
                mu_abnorm_2,
                var_abnorm_2,
                self.transition_coef[abnorm_keys[1]],
            )
        )

        # Combine the two final distributions
        mu_states_combined, var_states_combined = common.gaussian_mixture(
            mu_states_marginal["norm"],
            var_states_marginal["norm"],
            self.marginal_prob["norm"],
            mu_states_marginal["abnorm"],
            var_states_marginal["abnorm"],
            self.marginal_prob["abnorm"],
        )

        return (
            mu_states_combined,
            var_states_combined,
            mu_states_marginal,
            var_states_marginal,
        )

    def _estimate_transition_coef(
        self,
        obs,
        mu_pred_transit,
        var_pred_transit,
    ) -> Dict[str, float]:
        """Estimate transition coefficients given observation and predicted transitions.

        Computes the joint transition probabilities combining prior probabilities,
        transition probabilities and likelihoods, then normalizes to yield
        conditional transition coefficients.

        Args:
            obs (float): Observation at current step.
            mu_pred_transit (dict[str, np.ndarray]): Predicted means per transition.
            var_pred_transit (dict[str, np.ndarray]): Predicted variances per transition.

        Returns:
            Dict[str, float]: Updated transition coefficients for each origin→arrival key.
        """

        epsilon = 0 * 1e-20
        transition_coef = self._transition()
        transition_likelihood = self._compute_transition_likelihood(
            obs, mu_pred_transit, var_pred_transit
        )

        #
        trans_prob = self._transition()
        sum_trans_prob = 0
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                trans_prob[transit] = (
                    transition_likelihood[transit]
                    * self.transition_prob[transit]
                    * self.marginal_prob[origin_state]
                )
                sum_trans_prob += trans_prob[transit]
        for transit in trans_prob:
            trans_prob[transit] = trans_prob[transit] / np.maximum(
                sum_trans_prob, epsilon
            )

        #
        self.marginal_prob["norm"] = trans_prob["norm_norm"] + trans_prob["abnorm_norm"]
        self.marginal_prob["abnorm"] = (
            trans_prob["abnorm_abnorm"] + trans_prob["norm_abnorm"]
        )
        #
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                transition_coef[transit] = trans_prob[transit] / np.maximum(
                    self.marginal_prob[arrival_state], epsilon
                )

        return transition_coef

    def auto_initialize_baseline_states(self, y: np.ndarray):
        """
        Automatically assign initial means and variances for baseline hidden states (level,
        trend, and acceleration) from input data using time series decomposition
        defined in :meth:`~canari.data_process.DataProcess.decompose_data`.

        Args:
            data (np.ndarray): Time series data.

        Examples:
            >>> skf.auto_initialize_baseline_states(train_set["y"][0:23])
        """

        self.model["norm_norm"].auto_initialize_baseline_states(y)

    def save_initial_states(self):
        """
        Save current memory from the transition model 'norm_norm' in :attr:`.model`
        for reuse in subsequent runs.
        """

        self._memory_init = self.model["norm_norm"].get_memory()

    def load_initial_states(self):
        """
        Restore saved memory into the transition model 'norm_norm' in :attr:`.model`.
        """

        self.model["norm_norm"].set_memory(memory=self._memory_init)

    def initialize_states_history(self):
        """
        Reinitialize prior, posterior, and smoothed values for marginal hidden states in
        :attr:`.states` with empty lists, as well as for all transition models in :attr:`.model`.
        """

        for transition_model in self.model.values():
            transition_model.initialize_states_history()
        self.states.initialize(self.states_name)
        self.model["norm_norm"].lstm_states_history = []

    def set_states(self):
        """
        Set 'mu_states' and 'var_states' for each transition models in :attr:`.model` using their posterior.
        """

        for transition_model in self.model.values():
            transition_model.set_states(
                transition_model.mu_states_posterior,
                transition_model.var_states_posterior,
            )

    def set_memory(self, time_step: int):
        """
        Apply :meth:`~canari.model.Model.set_memory` for the transition model 'norm_norm' stored in :attr:`.model`.
        If `time_step=0`, reset :attr:`.marginal_prob` using :attr:`.norm_model_prior_prob`.

        Args:
            states (StatesHistory): Full history of hidden states over time.
            time_step (int): Index of timestep to restore.

        Examples:
            >>> # If the next analysis starts from the beginning of the time series
            >>> skf.set_memory(states=skf.states, time_step=0))
            >>> # If the next analysis starts from t = 200
            >>> skf.set_memory(states=skf.states, time_step=200))
        """

        self.model["norm_norm"].set_memory(time_step=time_step)
        if time_step == 0:
            self.marginal_prob["norm"] = copy.copy(self.norm_model_prior_prob)
            self.marginal_prob["abnorm"] = 1 - self.norm_model_prior_prob

    def get_dict(self, time_step: Optional[int] = None) -> dict:
        """
        Export an SKF object into a dictionary.

        Args:
            time_step (Optional[int]): the time step to get the model and memory at.
                                        If None export the model with the current memory.
                                        Defaults to None.

        Returns:
            dict: Serializable model dictionary containing neccessary attributes.

        Examples:
            >>> saved_dict = skf.get_dict()
        """

        save_dict = {}
        save_dict["norm_model"] = self.model["norm_norm"].get_dict(time_step=time_step)
        save_dict["abnorm_model"] = self.model["abnorm_abnorm"].get_dict()

        save_dict["std_transition_error"] = self.std_transition_error
        save_dict["norm_to_abnorm_prob"] = self.norm_to_abnorm_prob
        save_dict["abnorm_to_norm_prob"] = self.abnorm_to_norm_prob
        save_dict["norm_model_prior_prob"] = self.norm_model_prior_prob

        return save_dict

    @staticmethod
    def load_dict(save_dict: dict):
        """
        Reconstruct an SKF instance from a saved serialized dictionary.

        Args:
            save_dict (dict): Dictionary produced by get_dict().

        Returns:
            SKF: A new SKF object with loaded parameters and states.

        Examples:
            >>> saved_dict = skf.get_dict()
            >>> loaded_skf = SKF.load_dict(saved_dict)
        """

        # Create normal model
        norm_components = list(save_dict["norm_model"]["components"].values())
        norm_model = Model(*norm_components)

        # TODO: trick: run skf.filter to initialize skf.model["norm_norm"] states
        if "cov_names" in save_dict:
            dummy_input = np.full((len(save_dict["cov_names"]),), np.nan)
            norm_model.forward(input_covariates=dummy_input)

        if norm_model.lstm_net:
            norm_model.lstm_net.load_state_dict(
                save_dict["norm_model"]["lstm_network_params"]
            )
            norm_model.lstm_net.set_lstm_states(
                save_dict["norm_model"]["memory"]["lstm_states"]
            )
            norm_model.lstm_output_history.set(
                save_dict["norm_model"]["memory"]["lstm_output_history_mu"],
                save_dict["norm_model"]["memory"]["lstm_output_history_var"],
            )

        # Create abnormal model
        ab_components = list(save_dict["abnorm_model"]["components"].values())
        ab_model = Model(*ab_components)

        skf = SKF(
            norm_model=norm_model,
            abnorm_model=ab_model,
            std_transition_error=save_dict["std_transition_error"],
            norm_to_abnorm_prob=save_dict["norm_to_abnorm_prob"],
            abnorm_to_norm_prob=save_dict["abnorm_to_norm_prob"],
            norm_model_prior_prob=save_dict["norm_model_prior_prob"],
        )
        skf.model["norm_norm"].set_states(
            save_dict["norm_model"]["memory"]["mu_states"],
            save_dict["norm_model"]["memory"]["var_states"],
        )

        return skf

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
        white_noise_decay: Optional[bool] = True,
        white_noise_max_std: Optional[float] = 5,
        white_noise_decay_factor: Optional[float] = 0.9,
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """
        Train the :class:`~canari.component.lstm_component.LstmNetwork` component
        on the provided training set, then evaluate on the validation set.

        Recalling :meth:`~canari.model.Model.lstm_train` for 'norm_norm' in :attr:`.model`

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

                - **mu_obs_preds** (np.ndarray):
                    The means for multi-step-ahead predictions for the validation set.
                - **std_obs_preds** (np.ndarray):
                    The standard deviations for multi-step-ahead predictions for the validation set.
                - :attr:`~canari.model.Model.states`:
                    The history of hidden states over time.

        Examples:
            >>> mu_preds_val, std_preds_val, states = skf.lstm_train(train_data=train_set,validation_data=val_set)
        """

        return self.model["norm_norm"].lstm_train(
            train_data,
            validation_data,
            white_noise_decay,
            white_noise_max_std,
            white_noise_decay_factor,
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
        Apply early stopping based on a monitored metric when training a LSTM neural network.

        Recalling :meth:`~canari.model.Model.early_stopping` for 'norm_norm' in :attr:`.model`.

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

        Examples:
            >>> skf.early_stopping(evaluate_metric=mse, current_epoch=1, max_epoch=50)
        """

        (
            self.stop_training,
            self.optimal_epoch,
            self.early_stop_metric,
            self.early_stop_metric_history,
        ) = self.model["norm_norm"].early_stopping(
            current_epoch=current_epoch,
            max_epoch=max_epoch,
            mode=mode,
            patience=patience,
            evaluate_metric=evaluate_metric,
            skip_epoch=skip_epoch,
        )
        return (
            self.stop_training,
            self.optimal_epoch,
            self.early_stop_metric,
            self.early_stop_metric_history,
        )

    def forward(
        self,
        obs: float,
        input_covariates: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step in the Switching Kalman filter. This makes a one-step-ahead prediction.
        It is a mixture prediction from all transition models in :attr:`.model`.

        Recall :meth:`~canari.common.forward` for all transition models.

        This function is used at the one-time-step level.

        Args:
            obs (float): Current observation.
            input_covariates (Optional[np.ndarray]): Input covariates for LSTM at time `t`.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                A tuple containing:

                - :attr:`~canari.model.Model.mu_obs_predict` (np.ndarray):
                    The predictive mean of the observation at `t+1`.
                - :attr:`~canari.model.Model.var_obs_predict` (np.ndarray):
                    The predictive variance of the observation at `t+1`.
        """

        mu_pred_transit = self._transition()
        var_pred_transit = self._transition()
        mu_states_transit = self._transition()
        var_states_transit = self._transition()

        if self.lstm_net:
            mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                self.model["norm_norm"].lstm_output_history, input_covariates
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
                self.model["norm_norm"]._estim_hete_noise(
                    mu_v2bar_prior, var_v2bar_prior
                )

        else:
            mu_lstm_pred = None
            var_lstm_pred = None

        for transit, transition_model in self.model.items():
            (
                mu_pred_transit[transit],
                var_pred_transit[transit],
                mu_states_transit[transit],
                var_states_transit[transit],
            ) = transition_model.forward(
                mu_lstm_pred=mu_lstm_pred, var_lstm_pred=var_lstm_pred
            )

        self.transition_coef = self._estimate_transition_coef(
            obs,
            mu_pred_transit,
            var_pred_transit,
        )

        mu_states_prior, var_states_prior, _, _ = self._collapse_states(
            mu_states_transit, var_states_transit, state_type="prior"
        )

        mu_obs_pred, var_obs_pred = common.calc_observation(
            mu_states_prior,
            var_states_prior,
            self.model["norm_norm"].observation_matrix,
        )

        self.mu_states_prior = mu_states_prior
        self.var_states_prior = var_states_prior

        return mu_obs_pred, var_obs_pred

    def backward(
        self,
        obs: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step for Swithching Kalman filter.
        Recall :meth:`~canari.common.backward` for all transition models in :attr:`.model`.

        This function is used at the one-time-step level.

        Args:
            obs (float): Observation at the current time step.

        Returns:
            Tuple(mu_states_posterior, var_states_posterior): Posterior state estimates.
        """

        mu_states_transit = self._transition()
        var_states_transit = self._transition()

        for transit, transition_model in self.model.items():
            _, _, mu_states_transit[transit], var_states_transit[transit] = (
                transition_model.backward(obs)
            )

        (
            mu_states_posterior,
            var_states_posterior,
            mu_states_marginal,
            var_states_marginal,
        ) = self._collapse_states(
            mu_states_transit, var_states_transit, state_type="posterior"
        )

        # Reassign posterior states
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                self.model[transit]._set_posterior_states(
                    mu_states_marginal[origin_state], var_states_marginal[origin_state]
                )

        self.mu_states_posterior = mu_states_posterior
        self.var_states_posterior = var_states_posterior

        return mu_states_posterior, var_states_posterior

    def rts_smoother(
        self,
        time_step: int,
        matrix_inversion_tol: Optional[float] = 1e-2,
        tol_type: Optional[str] = "absolute",  # relative of absolute
    ):
        """
        Smoother for the Switching Kalman filter at a given time step.

        Recall :meth:`~canari.common.rts_smoother` for all transition models in :attr:`.model`.

        This function is used at the one-time-step level.

        Args:
            time_step (int): Index at which to perform smoothing.
            matrix_inversion_tol (float): Numerical stability threshold for matrix
                                            pseudoinversion (pinv). Defaults to 1E-2.
            tol_type (Optional[str]): Tolerance type, "relative" or "absolute".
                                        Defaults to "relative".
        """

        epsilon = 0 * 1e-20
        for transition_model in self.model.values():
            transition_model.rts_smoother(
                time_step, matrix_inversion_tol=matrix_inversion_tol, tol_type=tol_type
            )

        joint_transition_prob = self._transition()
        arrival_state_marginal = self._marginal()

        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                joint_transition_prob[transit] = (
                    self.filter_marginal_prob_history[origin_state][time_step]
                    * self.transition_prob[transit]
                )

        arrival_state_marginal["norm"] = (
            joint_transition_prob["norm_norm"] + joint_transition_prob["abnorm_norm"]
        )
        arrival_state_marginal["abnorm"] = (
            joint_transition_prob["norm_abnorm"]
            + joint_transition_prob["abnorm_abnorm"]
        )

        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                joint_transition_prob[transit] = joint_transition_prob[
                    transit
                ] / np.maximum(arrival_state_marginal[arrival_state], epsilon)

        joint_future_prob = self._transition()
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                joint_future_prob[transit] = (
                    joint_transition_prob[transit]
                    * self.smooth_marginal_prob_history[arrival_state][time_step + 1]
                )

        self.marginal_prob["norm"] = (
            joint_future_prob["norm_norm"] + joint_future_prob["norm_abnorm"]
        )
        self.marginal_prob["abnorm"] = (
            joint_future_prob["abnorm_norm"] + joint_future_prob["abnorm_abnorm"]
        )

        self.smooth_marginal_prob_history["norm"][time_step] = self.marginal_prob[
            "norm"
        ].copy()
        self.smooth_marginal_prob_history["abnorm"][time_step] = self.marginal_prob[
            "abnorm"
        ].copy()

        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                self.transition_coef[transit] = joint_future_prob[transit] / np.maximum(
                    self.marginal_prob[origin_state], epsilon
                )

        mu_states_transit, var_states_transit = self._get_smooth_states_transition(
            time_step
        )
        (
            self.states.mu_smooth[time_step],
            self.states.var_smooth[time_step],
            mu_states_marginal,
            var_states_marginal,
        ) = self._collapse_states(
            mu_states_transit,
            var_states_transit,
            state_type="smooth",
        )

        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                self.model[transit].states.mu_smooth[time_step] = mu_states_marginal[
                    arrival_state
                ].copy()
                self.model[transit].states.var_smooth[time_step] = var_states_marginal[
                    arrival_state
                ].copy()

    def filter(
        self,
        data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, StatesHistory]:
        """
        Run the Kalman filter over an entire dataset.

        This function is used at the entire-dataset-level. Recall repeatedly the function
        :meth:`.forward` and :meth:`.backward` at
        one-time-step level from :class:`~canari.skf.SKF`.

        Args:
            data (Dict[str, np.ndarray]): Includes 'x' and 'y'.

        Returns:
            Tuple[np.ndarray, StatesHistory]:
                A tuple containing:

                - **filter_marginal_prob_abnorm** (np.ndarray):
                    A history of filtering marginal probability for the abnorm model.
                - :attr:`~canari.model.Model.states`:
                    The history of marginal hidden states over time.

        Examples:
            >>> anomaly_prob, states = skf.filter(data=train_set)
        """

        mu_obs_preds = []
        var_obs_preds = []
        self.filter_marginal_prob_history = self._prob_history()

        # Initialize hidden states
        self._set_same_states_transition_models()
        self.initialize_states_history()

        # set lstm to train mode
        if self.lstm_net:
            self.lstm_net.train()
            if self.lstm_net.smooth:
                self.lstm_net.num_samples = len(data["y"])

        for index, (x, y) in enumerate(zip(data["x"], data["y"])):
            mu_obs_pred, var_obs_pred = self.forward(input_covariates=x, obs=y)
            mu_states_posterior, var_states_posterior = self.backward(y)

            if self.lstm_net:
                self.model["norm_norm"].update_lstm_output_history(
                    mu_states_posterior, var_states_posterior
                )
                self.model["norm_norm"].update_lstm_states_history(
                    index, last_step=len(data["y"]) - 1
                )

            self._save_states_history()
            self.set_states()
            mu_obs_preds.append(mu_obs_pred)
            var_obs_preds.append(var_obs_pred)
            self.filter_marginal_prob_history["norm"].append(self.marginal_prob["norm"])
            self.filter_marginal_prob_history["abnorm"].append(
                self.marginal_prob["abnorm"]
            )

        self.set_memory(time_step=0)
        return (
            np.array(self.filter_marginal_prob_history["abnorm"]),
            self.states,
        )

    def smoother(
        self,
        matrix_inversion_tol: Optional[float] = 1e-2,
        tol_type: Optional[str] = "absolute",  # relative of absolute
    ) -> Tuple[np.ndarray, StatesHistory]:
        """
        Run the Kalman smoother over an entire time series data.

        This function is used at the entire-dataset-level. Recall repeatedly the function
        :meth:`.rts_smoother` at one-time-step level from :class:`~canari.skf.SKF`.

        Args:
            matrix_inversion_tol (float): Numerical stability threshold for matrix
                                            pseudoinversion (pinv). Defaults to 1E-2.
            tol_type (Optional[str]): Tolerance type, "relative" or "absolute".
                                        Defaults to "relative".

        Returns:
            Tuple[np.ndarray, StatesHistory]:
                A tuple containing:

                - **smooth_marginal_prob_abnorm** (np.ndarray):
                    A history of smoother marginal probability for the abnorm model.
                - :attr:`~canari.model.Model.states`:
                    The history of marginal hidden states over time.
        """

        num_time_steps = len(self.states.mu_smooth)
        self.smooth_marginal_prob_history = copy.copy(self.filter_marginal_prob_history)
        self._initialize_smoother()
        for time_step in reversed(range(0, num_time_steps - 1)):
            self.rts_smoother(time_step, matrix_inversion_tol, tol_type)

        return (
            np.array(self.smooth_marginal_prob_history["abnorm"]),
            self.states,
        )

    def detect_synthetic_anomaly(
        self,
        data: Dict[str, np.ndarray],
        threshold: Optional[float] = 0.5,
        max_timestep_to_detect: Optional[int] = None,
        num_anomaly: Optional[int] = None,
        slope_anomaly: Optional[float] = None,
        anomaly_start: Optional[float] = 0.33,
        anomaly_end: Optional[float] = 0.66,
    ) -> Tuple[float, float]:
        """
        Add synthetic anomalies to orginal data, use Switching Kalman filter to detect those
        synthetic anomalies, and compute the detection/false-alarm rates.

        Args:
            data (Dict[str, np.ndarray]): Original time series data.
            threshold (Optional[float] ): Threshold for the maximal target anomaly detection rate.
                                Defauls to 0.5.
            max_timestep_to_detect (Optional[int]): Maximum number of timesteps to allow detection.
                                        Defauls to None (to the end of time series).
            num_anomaly (Optional[int]): Number of synthetic anomalies to add. This will create as
                                many time series, because one time series contains only one
                                anomaly.
            slope_anomaly (Optional[float]): Magnitude of the anomaly slope.
            anomaly_start (Optional[float]): Fractional start position of anomaly. Defaults to 0.33.
            anomaly_end (Optional[float]): Fractional end position of anomaly. Defaults to 0.66.

        Returns:
            Tuple(detection_rate, false_rate, false_alarm_train):
                detection_rate (float): # time series where anomalies detected / # total synthetic time series with anomalies added.
                false_rate (float): # time series where anomalies NOT detected / # total synthetic time series with anomalies added.
                false_alarm_train (str): 'Yes' if any alarm during training data.
        """

        num_timesteps = len(data["y"])
        num_anomaly_detected = 0
        num_false_alarm = 0
        false_alarm_train = "No"

        synthetic_data = DataProcess.add_synthetic_anomaly(
            data,
            num_samples=num_anomaly,
            slope=[slope_anomaly],
            anomaly_start=anomaly_start,
            anomaly_end=anomaly_end,
        )

        filter_marginal_abnorm_prob, _ = self.filter(data=data)
        self.load_initial_states()

        # Check false alarm in the training set
        if any(filter_marginal_abnorm_prob > threshold):
            false_alarm_train = "Yes"

        # Iterate over data with synthetic anomalies
        for i in range(0, num_anomaly):
            filter_marginal_abnorm_prob, _ = self.filter(data=synthetic_data[i])
            window_start = synthetic_data[i]["anomaly_timestep"]

            if max_timestep_to_detect is None:
                window_end = num_timesteps
            else:
                window_end = window_start + max_timestep_to_detect
            if any(filter_marginal_abnorm_prob[window_start:window_end] > threshold):
                num_anomaly_detected += 1
            if any(filter_marginal_abnorm_prob[:window_start] > threshold):
                num_false_alarm += 1

            self.load_initial_states()

        detection_rate = num_anomaly_detected / num_anomaly
        false_rate = num_false_alarm / num_anomaly

        return detection_rate, false_rate, false_alarm_train
