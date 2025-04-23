from typing import Tuple, Dict, Optional
import copy
import numpy as np
from pytagi import metric
from canari.model import Model
from canari import common
from canari.data_struct import StatesHistory
from canari.data_process import DataProcess


class SKF:
    """Switching Kalman Filter"""

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
        self.std_transition_error = std_transition_error
        self.norm_to_abnorm_prob = norm_to_abnorm_prob
        self.abnorm_to_norm_prob = abnorm_to_norm_prob
        self.norm_model_prior_prob = norm_model_prior_prob
        self.conditional_likelihood = conditional_likelihood
        self.model = self.transition()
        self.states = StatesHistory()
        self._initialize_attributes()
        self._initialize_model(norm_model, abnorm_model)

    @staticmethod
    def prob_history():
        """Create a dictionary to save marginal probability history"""
        return {
            "norm": [],
            "abnorm": [],
        }

    @staticmethod
    def transition():
        """Create a dictionary for transition"""
        return {
            "norm_norm": None,
            "abnorm_abnorm": None,
            "norm_abnorm": None,
            "abnorm_norm": None,
        }

    @staticmethod
    def marginal():
        """Create a dictionary for mariginal"""
        return {
            "norm": None,
            "abnorm": None,
        }

    def _initialize_attributes(self):
        """Initialize attribute"""

        # General attributes
        self.num_states = 0
        self.states_name = []

        # SKF-related attributes
        self.mu_states_init = None
        self.var_states_init = None
        self.mu_states_prior = None
        self.var_states_prior = None
        self.mu_states_posterior = None
        self.var_states_posterior = None

        self.transition_coef = self.transition()
        self.filter_marginal_prob_history = self.prob_history()
        self.smooth_marginal_prob_history = self.prob_history()
        self.marginal_list = {"norm", "abnorm"}

        self.transition_prob = self.transition()
        self.transition_prob["norm_norm"] = 1 - self.norm_to_abnorm_prob
        self.transition_prob["norm_abnorm"] = self.norm_to_abnorm_prob
        self.transition_prob["abnorm_norm"] = self.abnorm_to_norm_prob
        self.transition_prob["abnorm_abnorm"] = 1 - self.abnorm_to_norm_prob

        self.marginal_prob_current = self.marginal()
        self.marginal_prob_current["norm"] = self.norm_model_prior_prob
        self.marginal_prob_current["abnorm"] = 1 - self.norm_model_prior_prob

        # LSTM-related attributes
        self.lstm_net = None
        self.lstm_output_history = None

        # Early stopping attributes
        self.stop_training = False
        self.optimal_epoch = 0
        self.early_stop_metric_history = []
        self.early_stop_metric = None

    def _initialize_model(self, norm_model: Model, abnorm_model: Model):
        """Initialize model"""

        self._create_transition_model(
            norm_model,
            abnorm_model,
        )
        self._link_skf_to_model()
        self.save_initial_states()

    @staticmethod
    def _create_compatible_models(soure_model, target_model) -> None:
        """Create compatiable model by padding zero to states and matrices"""

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
        """Create transitional models from normal and abnormal models"""

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
        """Link SKF's attributes to norm_norm model's attributes"""

        self.num_states = self.model["norm_norm"].num_states
        self.states_name = self.model["norm_norm"].states_name
        if self.model["norm_norm"].lstm_net is not None:
            self.lstm_net = self.model["norm_norm"].lstm_net
            self.lstm_output_history = self.model["norm_norm"].lstm_output_history

    def set_same_states_transition_models(self):
        """Copy the states from self.model["norm_norm"] to all other transition models"""

        for transition_model in self.model.values():
            transition_model.set_states(
                self.model["norm_norm"].mu_states, self.model["norm_norm"].var_states
            )

    def _initialize_smoother(self):
        """Set the smoothed estimates at the last time step = posterior"""

        self.states.mu_smooth[-1], self.states.var_smooth[-1] = common.gaussian_mixture(
            self.model["norm_norm"].states.mu_posterior[-1],
            self.model["norm_norm"].states.var_posterior[-1],
            self.filter_marginal_prob_history["norm"][-1],
            self.model["abnorm_abnorm"].states.mu_posterior[-1],
            self.model["abnorm_abnorm"].states.var_posterior[-1],
            self.filter_marginal_prob_history["abnorm"][-1],
        )

    def _save_states_history(self):
        """Save states' priors, posteriors and cross-covariances at one time step"""

        for transition_model in self.model.values():
            transition_model._save_states_history()

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
        Compute the likelihood of observing 'obs' given predicted means and variances, for each transition model.
        """

        transition_likelihood = self.transition()
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
                    transition_likelihood[transit] = np.exp(
                        metric.log_likelihood(
                            mu_pred_transit[transit],
                            obs,
                            var_pred_transit[transit] ** 0.5,
                        )
                    )
        return transition_likelihood

    def _get_smooth_states_transition(
        self,
        time_step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get states values for collapse step
        """
        mu_states_transit = self.transition()
        var_states_transit = self.transition()

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
        """
        Collapse step
        """

        mu_states_marginal = self.marginal()
        var_states_marginal = self.marginal()

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
            self.marginal_prob_current["norm"],
            mu_states_marginal["abnorm"],
            var_states_marginal["abnorm"],
            self.marginal_prob_current["abnorm"],
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
        """
        Estimate coefficients for each transition model
        """

        epsilon = 0 * 1e-20
        transition_coef = self.transition()
        transition_likelihood = self._compute_transition_likelihood(
            obs, mu_pred_transit, var_pred_transit
        )

        #
        trans_prob = self.transition()
        sum_trans_prob = 0
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                trans_prob[transit] = (
                    transition_likelihood[transit]
                    * self.transition_prob[transit]
                    * self.marginal_prob_current[origin_state]
                )
                sum_trans_prob += trans_prob[transit]
        for transit in trans_prob:
            trans_prob[transit] = trans_prob[transit] / np.maximum(
                sum_trans_prob, epsilon
            )

        #
        self.marginal_prob_current["norm"] = (
            trans_prob["norm_norm"] + trans_prob["abnorm_norm"]
        )
        self.marginal_prob_current["abnorm"] = (
            trans_prob["abnorm_abnorm"] + trans_prob["norm_abnorm"]
        )
        #
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                transition_coef[transit] = trans_prob[transit] / np.maximum(
                    self.marginal_prob_current[arrival_state], epsilon
                )

        return transition_coef

    def auto_initialize_baseline_states(self, y: np.ndarray):
        """Automatically initialize baseline states from data for normal model"""
        self.model["norm_norm"].auto_initialize_baseline_states(y)
        self.save_initial_states()

    def save_initial_states(self):
        """
        save initial states at time = 0 for reuse SKF
        """

        self.mu_states_init = self.model["norm_norm"].mu_states.copy()
        self.var_states_init = self.model["norm_norm"].var_states.copy()

    def load_initial_states(self):
        """ """

        self.model["norm_norm"].mu_states = self.mu_states_init.copy()
        self.model["norm_norm"].var_states = self.var_states_init.copy()

    def initialize_states_history(self):
        """Initialize history for all time steps for the combined states and each transition model"""

        for transition_model in self.model.values():
            transition_model.initialize_states_history()
        self.states.initialize(self.states_name)

    def set_states(self):
        """Assign new values for the states of each transition model"""

        for transition_model in self.model.values():
            transition_model.set_states(
                transition_model.mu_states_posterior,
                transition_model.var_states_posterior,
            )

    def set_memory(self, states: StatesHistory, time_step: int):
        """
        Set memory at a specific time step
        """

        self.model["norm_norm"].set_memory(
            states=self.model["norm_norm"].states, time_step=0
        )
        if time_step == 0:
            self.load_initial_states()
            self.marginal_prob_current["norm"] = self.norm_model_prior_prob
            self.marginal_prob_current["abnorm"] = 1 - self.norm_model_prior_prob

    def get_dict(self) -> dict:
        """
        Save the SKF model as a dict.
        """

        save_dict = {}
        save_dict["norm_model"] = self.model["norm_norm"].get_dict()
        save_dict["abnorm_model"] = self.model["abnorm_abnorm"].get_dict()
        save_dict["std_transition_error"] = self.std_transition_error
        save_dict["norm_to_abnorm_prob"] = self.norm_to_abnorm_prob
        save_dict["abnorm_to_norm_prob"] = self.abnorm_to_norm_prob
        save_dict["norm_model_prior_prob"] = self.norm_model_prior_prob
        if self.lstm_net:
            save_dict["lstm_network_params"] = self.lstm_net.state_dict()

        return save_dict

    @staticmethod
    def load_dict(save_dict: dict):
        """
        Create a model from a saved dict
        """

        # Create normal model
        norm_components = list(save_dict["norm_model"]["components"].values())
        norm_model = Model(*norm_components)
        if norm_model.lstm_net:
            norm_model.lstm_net.load_state_dict(save_dict["lstm_network_params"])

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
            save_dict["norm_model"]["mu_states"], save_dict["norm_model"]["var_states"]
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
        Train the LstmNetwork of the normal model
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
        mode: Optional[str] = "max",
        patience: Optional[int] = 20,
        evaluate_metric: Optional[float] = None,
        skip_epoch: Optional[int] = 5,
    ) -> Tuple[bool, int, float, list]:
        (
            self.stop_training,
            self.optimal_epoch,
            self.early_stop_metric,
            self.early_stop_metric_history,
        ) = self.model["norm_norm"].early_stopping(
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
        Forward pass for 4 transition models
        """

        mu_pred_transit = self.transition()
        var_pred_transit = self.transition()
        mu_states_transit = self.transition()
        var_states_transit = self.transition()

        if self.lstm_net:
            mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                self.lstm_output_history, input_covariates
            )
            mu_lstm_pred, var_lstm_pred = self.lstm_net.forward(
                mu_x=np.float32(mu_lstm_input), var_x=np.float32(var_lstm_input)
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
    ):
        """
        Update step in states-space model
        """
        mu_states_transit = self.transition()
        var_states_transit = self.transition()

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
                self.model[transit].set_posterior_states(
                    mu_states_marginal[origin_state], var_states_marginal[origin_state]
                )

        self.mu_states_posterior = mu_states_posterior
        self.var_states_posterior = var_states_posterior

        return mu_states_posterior, var_states_posterior

    def rts_smoother(self, time_step: int):
        """
        RTS smoother
        """

        epsilon = 0 * 1e-20
        for transition_model in self.model.values():
            transition_model.rts_smoother(time_step, matrix_inversion_tol=1e-3)

        joint_transition_prob = self.transition()
        arrival_state_marginal = self.marginal()

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

        joint_future_prob = self.transition()
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                joint_future_prob[transit] = (
                    joint_transition_prob[transit]
                    * self.smooth_marginal_prob_history[arrival_state][time_step + 1]
                )

        self.marginal_prob_current["norm"] = (
            joint_future_prob["norm_norm"] + joint_future_prob["norm_abnorm"]
        )
        self.marginal_prob_current["abnorm"] = (
            joint_future_prob["abnorm_norm"] + joint_future_prob["abnorm_abnorm"]
        )

        self.smooth_marginal_prob_history["norm"][time_step] = (
            self.marginal_prob_current["norm"]
        )
        self.smooth_marginal_prob_history["abnorm"][time_step] = (
            self.marginal_prob_current["abnorm"]
        )

        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                self.transition_coef[transit] = joint_future_prob[transit] / np.maximum(
                    self.marginal_prob_current[origin_state], epsilon
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
                ]
                self.model[transit].states.var_smooth[time_step] = var_states_marginal[
                    arrival_state
                ]

    def filter(
        self,
        data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, StatesHistory]:
        """
        Filtering
        """

        mu_obs_preds = []
        var_obs_preds = []
        self.filter_marginal_prob_history = self.prob_history()

        # Initialize hidden states
        self.set_same_states_transition_models()
        self.initialize_states_history()

        for x, y in zip(data["x"], data["y"]):
            mu_obs_pred, var_obs_pred = self.forward(input_covariates=x, obs=y)
            mu_states_posterior, var_states_posterior = self.backward(y)

            if self.lstm_net:
                lstm_index = self.model["norm_norm"].get_states_index("lstm")
                self.lstm_output_history.update(
                    mu_states_posterior[lstm_index],
                    var_states_posterior[
                        lstm_index,
                        lstm_index,
                    ],
                )

            self._save_states_history()
            self.set_states()
            mu_obs_preds.append(mu_obs_pred)
            var_obs_preds.append(var_obs_pred)
            self.filter_marginal_prob_history["norm"].append(
                self.marginal_prob_current["norm"]
            )
            self.filter_marginal_prob_history["abnorm"].append(
                self.marginal_prob_current["abnorm"]
            )

        self.set_memory(states=self.model["norm_norm"].states, time_step=0)
        return (
            np.array(self.filter_marginal_prob_history["abnorm"]),
            self.states,
        )

    def smoother(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, StatesHistory]:
        """
        Smoother for whole time series
        """

        num_time_steps = len(data["y"])
        self.smooth_marginal_prob_history = copy.copy(self.filter_marginal_prob_history)
        self._initialize_smoother()
        for time_step in reversed(range(0, num_time_steps - 1)):
            self.rts_smoother(time_step)

        return (
            np.array(self.smooth_marginal_prob_history["abnorm"]),
            self.states,
        )

    def detect_synthetic_anomaly(
        self,
        data: list[Dict[str, np.ndarray]],
        threshold: Optional[float] = 0.5,
        max_timestep_to_detect: Optional[int] = None,
        num_anomaly: Optional[int] = None,
        slope_anomaly: Optional[float] = None,
        anomaly_start: Optional[float] = 0.33,
        anomaly_end: Optional[float] = 0.66,
    ) -> Tuple[float, float]:
        """Detect anomaly"""

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

        detection_rate = num_anomaly_detected / num_anomaly
        false_rate = num_false_alarm / num_anomaly

        return detection_rate, false_rate, false_alarm_train
