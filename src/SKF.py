from typing import Tuple, Dict, Optional
import copy
import numpy as np
from pytagi import metric
from src.model import Model, load_model_dict
from src import common
import copy
from src.data_struct import (
    StatesHistory,
    initialize_transition,
    initialize_marginal,
    initialize_marginal_prob_history,
)
from examples import DataProcess
import matplotlib.pyplot as plt


class SKF:
    """ "
    Switching Kalman Filter
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
        self.std_transition_error = std_transition_error
        self.norm_to_abnorm_prob = norm_to_abnorm_prob
        self.abnorm_to_norm_prob = abnorm_to_norm_prob
        self.norm_model_prior_prob = norm_model_prior_prob
        self.conditional_likelihood = conditional_likelihood
        self.create_transition_model(
            norm_model,
            abnorm_model,
        )
        self.transition_coef = initialize_transition()
        self.define_lstm_network()
        self.states = StatesHistory()
        self.filter_marginal_prob_history = None
        self.smooth_marginal_prob_history = None
        self.marginal_list = {"norm", "abnorm"}
        self._marginal_prob = initialize_marginal()
        self._marginal_prob["norm"] = self.norm_model_prior_prob
        self._marginal_prob["abnorm"] = 1 - self.norm_model_prior_prob
        self.initialize_early_stop()

    def create_transition_model(
        self,
        norm_model: Model,
        abnorm_model: Model,
    ):
        """
        Create transitional models
        """

        # Create transitional model
        norm_norm = copy.deepcopy(norm_model)
        norm_norm.lstm_net = norm_model.lstm_net
        norm_norm.create_compatible_model(abnorm_model)
        if "white noise" in norm_norm.states_name:
            index_noise = norm_norm.states_name.index("white noise")
            abnorm_model.process_noise_matrix[index_noise, index_noise] = (
                norm_norm.process_noise_matrix[index_noise, index_noise]
            )
        abnorm_norm = copy.deepcopy(norm_norm)
        norm_abnorm = copy.deepcopy(abnorm_model)
        abnorm_abnorm = copy.deepcopy(abnorm_model)

        # Add transition noise to norm_abnorm.process_noise_matrix
        index_pad_state = norm_norm.index_pad_state
        norm_abnorm.process_noise_matrix[index_pad_state, index_pad_state] = (
            self.std_transition_error**2
        )

        # Store transitional models in a dictionary
        self.model = initialize_transition()
        self.model["norm_norm"] = norm_norm
        self.model["abnorm_abnorm"] = abnorm_abnorm
        self.model["norm_abnorm"] = norm_abnorm
        self.model["abnorm_norm"] = abnorm_norm

        # Transition probability
        self.transition_prob = initialize_transition()
        self.transition_prob["norm_norm"] = 1 - self.norm_to_abnorm_prob
        self.transition_prob["norm_abnorm"] = self.norm_to_abnorm_prob
        self.transition_prob["abnorm_norm"] = self.abnorm_to_norm_prob
        self.transition_prob["abnorm_abnorm"] = 1 - self.abnorm_to_norm_prob

        self.num_states = self.model["norm_norm"].num_states
        self.states_name = self.model["norm_norm"].states_name
        self.index_pad_state = self.model["norm_norm"].index_pad_state

    def define_lstm_network(self):
        """
        Assign self.lstm_net using self.model["norm_norm"].lstm_net
        """
        if self.model["norm_norm"].lstm_net is not None:
            self.lstm_net = self.model["norm_norm"].lstm_net
            self.lstm_states_index = self.model["norm_norm"].lstm_states_index
            self.lstm_output_history = self.model["norm_norm"].lstm_output_history
            self.update_lstm_output_history = self.model[
                "norm_norm"
            ].update_lstm_output_history
            self.initialize_lstm_output_history = self.model[
                "norm_norm"
            ].initialize_lstm_output_history
        else:
            self.lstm_net = None

    def initialize_early_stop(self):
        """
        Initialize for early stopping
        """

        self.stop_training = False
        self.optimal_epoch = 0
        self.early_stop_metric_history = []
        self.early_stop_metric = None

    def auto_initialize_baseline_states(self, y: np.ndarray):
        """
        Automatically initialize baseline states from data for normal model
        """

        self.model["norm_norm"].auto_initialize_baseline_states(y)

    def set_same_states_transition_model(self):
        """
        Copy the states from self.model["norm_norm"] to all other transition models
        """

        for transition_model in self.model.values():
            transition_model.set_states(
                self.model["norm_norm"].mu_states, self.model["norm_norm"].var_states
            )

    def save_states_history(self):
        """
        Save states' priors, posteriors and cross-covariances at one time step
        """

        for transition_model in self.model.values():
            transition_model.save_states_history()

        self.states.mu_prior.append(self.mu_states_prior.flatten())
        self.states.var_prior.append(self.var_states_prior)
        self.states.mu_posterior.append(self.mu_states_posterior.flatten())
        self.states.var_posterior.append(self.var_states_posterior)
        self.states.mu_smooth.append([])
        self.states.var_smooth.append([])

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
        """
        Initialize history for all time steps for the combined states and each transition model
        """

        for transition_model in self.model.values():
            transition_model.initialize_states_history()
        self.states.initialize(self.states_name)

    def initialize_smoother_buffers(self):
        """
        Set the smoothed estimates at the last time step = posterior
        """

        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                marginal_arrival = f"{arrival_state}_{arrival_state}"
                self.model[transit].states.mu_smooth[-1] = self.model[
                    marginal_arrival
                ].states.mu_posterior[-1]
                self.model[transit].states.var_smooth[-1] = self.model[
                    marginal_arrival
                ].states.var_posterior[-1]

        self.states.mu_smooth[-1], self.states.var_smooth[-1] = common.gaussian_mixture(
            self.model["norm_norm"].states.mu_posterior[-1],
            self.model["norm_norm"].states.var_posterior[-1],
            self.filter_marginal_prob_history["norm"][-1],
            self.model["abnorm_abnorm"].states.mu_posterior[-1],
            self.model["abnorm_abnorm"].states.var_posterior[-1],
            self.filter_marginal_prob_history["abnorm"][-1],
        )

        self.smooth_marginal_prob_history["norm"][-1] = (
            self.filter_marginal_prob_history["norm"][-1]
        )
        self.smooth_marginal_prob_history["abnorm"][-1] = (
            self.filter_marginal_prob_history["abnorm"][-1]
        )

    def set_states(self):
        """
        Assign new values for the states of each transition model
        """

        for transition_model in self.model.values():
            transition_model.set_states(
                transition_model.mu_states_posterior,
                transition_model.var_states_posterior,
            )

    def _compute_transition_likelihood(
        self,
        obs: float,
        mu_pred_transit: Dict[str, np.ndarray],
        var_pred_transit: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute the likelihood of observing 'obs' given predicted means and variances, for each transition model.
        """

        transition_likelihood = initialize_transition()
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
        mu_states_transit = initialize_transition()
        var_states_transit = initialize_transition()

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

        mu_states_marginal = initialize_marginal()
        var_states_marginal = initialize_marginal()

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
            self._marginal_prob["norm"],
            mu_states_marginal["abnorm"],
            var_states_marginal["abnorm"],
            self._marginal_prob["abnorm"],
        )

        return (
            mu_states_combined,
            var_states_combined,
            mu_states_marginal,
            var_states_marginal,
        )

    def estimate_transition_coef(
        self,
        obs,
        mu_pred_transit,
        var_pred_transit,
    ) -> Dict[str, float]:
        """
        Estimate coefficients for each transition model
        """

        epsilon = 0 * 1e-300
        transition_coef = initialize_transition()
        transition_likelihood = self._compute_transition_likelihood(
            obs, mu_pred_transit, var_pred_transit
        )

        #
        trans_prob = initialize_transition()
        sum_trans_prob = 0
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                trans_prob[transit] = (
                    transition_likelihood[transit]
                    * self.transition_prob[transit]
                    * self._marginal_prob[origin_state]
                )
                sum_trans_prob += trans_prob[transit]
        for transit in trans_prob:
            trans_prob[transit] = trans_prob[transit] / np.maximum(
                sum_trans_prob, epsilon
            )

        #
        self._marginal_prob["norm"] = (
            trans_prob["norm_norm"] + trans_prob["abnorm_norm"]
        )
        self._marginal_prob["abnorm"] = (
            trans_prob["abnorm_abnorm"] + trans_prob["norm_abnorm"]
        )
        #
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                transition_coef[transit] = trans_prob[transit] / np.maximum(
                    self._marginal_prob[arrival_state], epsilon
                )

        return transition_coef

    def initialize_states_with_smoother_estimates(self, epoch):
        self.model["norm_norm"].initialize_states_with_smoother_estimates_v1(epoch)

    def clear_history(self):
        """
        Clear history for next SKF run
        """

        if self.lstm_net:
            self.lstm_net.reset_lstm_states()
            self.initialize_lstm_output_history()
        self._marginal_prob["norm"] = self.norm_model_prior_prob
        self._marginal_prob["abnorm"] = 1 - self.norm_model_prior_prob

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """
        Train the LstmNetwork of the normal model
        """

        return self.model["norm_norm"].lstm_train(train_data, validation_data)

    def early_stopping(
        self,
        mode: Optional[str] = "max",
        patience: Optional[int] = 20,
        evaluate_metric: Optional[float] = None,
        skip_epoch: Optional[int] = 5,
    ) -> Tuple[bool, int, float, list]:
        self.model["norm_norm"].early_stopping(
            mode=mode,
            patience=patience,
            evaluate_metric=evaluate_metric,
            skip_epoch=skip_epoch,
        )
        self.stop_training = self.model["norm_norm"].stop_training
        self.optimal_epoch = self.model["norm_norm"].optimal_epoch
        self.early_stop_metric_history = self.model[
            "norm_norm"
        ].early_stop_metric_history
        self.early_stop_metric = self.model["norm_norm"].early_stop_metric

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

        mu_pred_transit = initialize_transition()
        var_pred_transit = initialize_transition()
        mu_states_transit = initialize_transition()
        var_states_transit = initialize_transition()

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

        self.transition_coef = self.estimate_transition_coef(
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
        mu_states_transit = initialize_transition()
        var_states_transit = initialize_transition()

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

        epsilon = 0 * 1e-300
        for transition_model in self.model.values():
            transition_model.rts_smoother(time_step, matrix_inversion_tol=1e-3)

        joint_transition_prob = initialize_transition()
        arrival_state_marginal = initialize_marginal()

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

        joint_future_prob = initialize_transition()
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                joint_future_prob[transit] = (
                    joint_transition_prob[transit]
                    * self.smooth_marginal_prob_history[arrival_state][time_step + 1]
                )

        self._marginal_prob["norm"] = (
            joint_future_prob["norm_norm"] + joint_future_prob["norm_abnorm"]
        )
        self._marginal_prob["abnorm"] = (
            joint_future_prob["abnorm_norm"] + joint_future_prob["abnorm_abnorm"]
        )

        self.smooth_marginal_prob_history["norm"][time_step] = self._marginal_prob[
            "norm"
        ].copy()
        self.smooth_marginal_prob_history["abnorm"][time_step] = self._marginal_prob[
            "abnorm"
        ].copy()

        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                self.transition_coef[transit] = joint_future_prob[transit] / np.maximum(
                    self._marginal_prob[origin_state], epsilon
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
        data = common.set_default_input_covariates(data)
        num_time_steps = len(data["y"])
        mu_obs_preds = []
        var_obs_preds = []
        self.filter_marginal_prob_history = initialize_marginal_prob_history(
            num_time_steps
        )
        if self.lstm_net:
            lstm_index = self.lstm_states_index

        # Initialize hidden states
        self.set_same_states_transition_model()
        self.initialize_states_history()

        for time_step, (x, y) in enumerate(zip(data["x"], data["y"])):
            mu_obs_pred, var_obs_pred = self.forward(input_covariates=x, obs=y)
            mu_states_posterior, var_states_posterior = self.backward(y)

            if self.lstm_net:
                self.update_lstm_output_history(
                    mu_states_posterior[lstm_index],
                    var_states_posterior[
                        lstm_index,
                        lstm_index,
                    ],
                )

            self.save_states_history()
            self.set_states()
            mu_obs_preds.append(mu_obs_pred)
            var_obs_preds.append(var_obs_pred)
            self.filter_marginal_prob_history["norm"][time_step] = self._marginal_prob[
                "norm"
            ].copy()
            self.filter_marginal_prob_history["abnorm"][time_step] = (
                self._marginal_prob["abnorm"].copy()
            )

        # clear SKF history for the next filter
        self.clear_history()
        return (
            self.filter_marginal_prob_history["abnorm"],
            self.states,
        )

    def smoother(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, StatesHistory]:
        """
        Smoother for whole time series
        """

        num_time_steps = len(data["y"])
        self.smooth_marginal_prob_history = initialize_marginal_prob_history(
            num_time_steps
        )

        # Smoother
        self.initialize_smoother_buffers()
        for time_step in reversed(range(0, num_time_steps - 1)):
            self.rts_smoother(time_step)

        return (
            self.smooth_marginal_prob_history["abnorm"],
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
        """ """

        synthetic_data = DataProcess.add_synthetic_anomaly(
            data,
            num_samples=num_anomaly,
            slope=[slope_anomaly],
            anomaly_start=anomaly_start,
            anomaly_end=anomaly_end,
        )
        num_timesteps = len(data["y"])
        num_anomaly_detected = 0
        num_false_alarm = 0

        for i in range(0, num_anomaly):
            self.load_initial_states()
            filter_marginal_abnorm_prob, states = self.filter(data=synthetic_data[i])
            window_start = synthetic_data[i]["anomaly_timestep"]

            if max_timestep_to_detect is None:
                window_end = num_timesteps
            else:
                window_end = window_start + max_timestep_to_detect
            if any(filter_marginal_abnorm_prob[window_start:window_end] > threshold):
                num_anomaly_detected += 1
            if any(filter_marginal_abnorm_prob[:window_start] > threshold):
                num_false_alarm += 1

            # states_plot = np.array(states.mu_posterior)
            # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 4))

            # axes[0].plot(states_plot[:, 0])
            # axes[0].plot(synthetic_data[i]["y"])
            # axes[0].axvline(x=window_start, color="b", linestyle="--")
            # axes[1].plot(filter_marginal_abnorm_prob, color="r")
            # axes[1].axvline(x=window_start, color="b", linestyle="--")
            # plt.show()

            check = 1

        detection_rate = num_anomaly_detected / num_anomaly
        false_rate = num_false_alarm / num_anomaly

        return detection_rate, false_rate

    def save_model_dict(self) -> dict:
        """
        Save the SKF model as a dict.
        """

        SKF_dict = {}
        SKF_dict["norm_model"] = self.model["norm_norm"].save_model_dict()
        SKF_dict["abnorm_model"] = self.model["abnorm_abnorm"].save_model_dict()
        SKF_dict["std_transition_error"] = self.std_transition_error
        SKF_dict["norm_to_abnorm_prob"] = self.norm_to_abnorm_prob
        SKF_dict["abnorm_to_norm_prob"] = self.abnorm_to_norm_prob
        SKF_dict["norm_model_prior_prob"] = self.norm_model_prior_prob
        if self.lstm_net:
            SKF_dict["lstm_network_params"] = self.lstm_net.get_state_dict()

        return SKF_dict


def load_SKF_dict(save_dict: dict) -> SKF:
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

    if skf.lstm_net:
        skf.lstm_net.load_state_dict(save_dict["lstm_network_params"])

    return skf
