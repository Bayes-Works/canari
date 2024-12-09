import numpy as np
import copy
from typing import Tuple, List, Dict, Optional
import pytagi.metric as metric
from src.model import Model
import src.common as common
from src.data_struct import (
    StatesHistory,
    initialize_transition,
    initialize_marginal,
    initialize_marginal_prob_history,
)


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
        conditional_likelihood: Optional[bool] = True,
    ):
        self.std_transition_error = std_transition_error
        self.conditional_likelihood = conditional_likelihood
        self.create_transition_model(
            norm_model,
            abnorm_model,
            norm_to_abnorm_prob,
            abnorm_to_norm_prob,
        )
        self.transition_coef = initialize_transition()
        self.define_lstm_network()
        self.states = StatesHistory()
        self.filter_marginal_prob_history = None
        self.smooth_marginal_prob_history = None
        self.marginal_list = {"norm", "abnorm"}
        self._marginal_prob = initialize_marginal()
        self._marginal_prob["norm"] = norm_model_prior_prob
        self._marginal_prob["abnorm"] = 1 - norm_model_prior_prob

    def create_transition_model(
        self,
        norm_model: Model,
        abnorm_model: Model,
        norm_to_abnorm_prob: float,
        abnorm_to_norm_prob: float,
    ):
        """
        Create transitional models
        """

        # Create transitional model
        norm_model.create_compatible_model(abnorm_model)
        abnorm_norm = norm_model.duplicate()
        norm_abnorm = abnorm_model.duplicate()

        # Add transition noise to norm_abnorm.process_noise_matrix
        index_pad_state = norm_model.index_pad_state
        norm_abnorm.process_noise_matrix[index_pad_state, index_pad_state] = (
            self.std_transition_error**2
        )

        # Store transitional models in a dictionary
        self.model = initialize_transition()
        self.model["norm_norm"] = norm_model
        self.model["abnorm_abnorm"] = abnorm_model
        self.model["norm_abnorm"] = norm_abnorm
        self.model["abnorm_norm"] = abnorm_norm

        # Transition probability
        self.transition_prob = initialize_transition()
        self.transition_prob["norm_norm"] = 1 - norm_to_abnorm_prob
        self.transition_prob["norm_abnorm"] = norm_to_abnorm_prob
        self.transition_prob["abnorm_norm"] = abnorm_to_norm_prob
        self.transition_prob["abnorm_abnorm"] = 1 - abnorm_to_norm_prob

        self.num_states = self.model["norm_norm"].num_states
        self.states_name = self.model["norm_norm"].states_name
        self.index_pad_state = self.model["norm_norm"].index_pad_state

    def define_lstm_network(self):
        """
        Assign self.lstm_net using self.model["norm_norm"].lstm_net
        """

        self.lstm_net = self.model["norm_norm"].lstm_net
        self.lstm_states_index = self.model["norm_norm"].lstm_states_index
        self.lstm_output_history = self.model["norm_norm"].lstm_output_history
        self.update_lstm_output_history = self.model[
            "norm_norm"
        ].update_lstm_output_history

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

    def save_states_history(self, time_step: int):
        """
        Save states' priors, posteriors and cross-covariances at one time step
        """

        for transition_model in self.model.values():
            transition_model.save_states_history(time_step)

        self.states.mu_prior[time_step] = self.mu_states_prior.copy().flatten()
        self.states.var_prior[time_step] = self.var_states_prior.copy()
        self.states.mu_posterior[time_step] = self._mu_states_posterior.copy().flatten()
        self.states.var_posterior[time_step] = self.var_states_posterior.copy()

    def initialize_states_history(self, num_time_steps: int):
        """
        Initialize history for all time steps for the combined states and each transition model
        """

        for transition_model in self.model.values():
            transition_model.initialize_states_history(num_time_steps)
        self.states.initialize(num_time_steps, self.num_states)

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

    def _get_states(
        self, category: str, state_type: str, time_step: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get states values for collapse step
        """

        if state_type == "prior":
            return (
                self.model[category].mu_states_prior,
                self.model[category].var_states_prior,
            )
        elif state_type == "posterior":
            return (
                self.model[category].mu_states_posterior,
                self.model[category].var_states_posterior,
            )
        elif state_type == "smooth":
            return (
                self.model[category].states.mu_smooth[time_step],
                self.model[category].states.var_smooth[time_step],
            )

    def _collapse_states(
        self, state_type: str, time_step: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Collapse step
        """

        mu_states_marginal = initialize_marginal()
        var_states_marginal = initialize_marginal()

        # Retrieve states for norm mixture
        mu_norm_norm, var_norm_norm = self._get_states(
            "norm_norm", state_type, time_step
        )
        mu_abnorm_norm, var_abnorm_norm = self._get_states(
            "abnorm_norm", state_type, time_step
        )

        mu_states_marginal["norm"], var_states_marginal["norm"] = (
            common.gaussian_mixture(
                mu_norm_norm,
                var_norm_norm,
                self.transition_coef["norm_norm"],
                mu_abnorm_norm,
                var_abnorm_norm,
                self.transition_coef["abnorm_norm"],
            )
        )

        # Retrieve states for abnorm mixture
        mu_norm_abnorm, var_norm_abnorm = self._get_states(
            "norm_abnorm", state_type, time_step
        )
        mu_abnorm_abnorm, var_abnorm_abnorm = self._get_states(
            "abnorm_abnorm", state_type, time_step
        )

        mu_states_marginal["abnorm"], var_states_marginal["abnorm"] = (
            common.gaussian_mixture(
                mu_norm_abnorm,
                var_norm_abnorm,
                self.transition_coef["norm_abnorm"],
                mu_abnorm_abnorm,
                var_abnorm_abnorm,
                self.transition_coef["abnorm_abnorm"],
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
            trans_prob[transit] = trans_prob[transit] / sum_trans_prob

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
                self.transition_coef[transit] = (
                    trans_prob[transit] / self._marginal_prob[arrival_state]
                )

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """
        Train the LstmNetwork of the normal model
        """

        return self.model["norm_norm"].lstm_train(train_data, validation_data)

    def forward(
        self,
        y,
        mu_lstm_pred: Optional[np.ndarray] = None,
        var_lstm_pred: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for 4 transition models
        """

        mu_pred_transit = initialize_transition()
        var_pred_transit = initialize_transition()

        for transit, transition_model in self.model.items():
            (
                mu_pred_transit[transit],
                var_pred_transit[transit],
            ) = transition_model.forward(mu_lstm_pred, var_lstm_pred)

        self.estimate_transition_coef(
            y,
            mu_pred_transit,
            var_pred_transit,
        )

        self.mu_states_prior, self.var_states_prior, _, _ = self._collapse_states(
            state_type="prior"
        )

        mu_obs_pred, var_obs_pred = common.calc_observation(
            self.mu_states_prior,
            self.var_states_prior,
            self.model["norm_norm"].observation_matrix,
        )

        return mu_obs_pred, var_obs_pred

    def backward(
        self,
        obs: float,
    ):
        """
        Update step in states-space model
        """

        for transition_model in self.model.values():
            mu_delta, var_delta = transition_model.backward(obs)
            transition_model.estimate_posterior_states(mu_delta, var_delta)

        (
            self._mu_states_posterior,
            self.var_states_posterior,
            mu_states_marginal,
            var_states_marginal,
        ) = self._collapse_states(state_type="posterior")

        # Reassign posterior states
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                self.model[transit].set_posterior_states(
                    mu_states_marginal[origin_state], var_states_marginal[origin_state]
                )

    def rts_smoother(self, time_step: int):
        """
        RTS smoother
        """

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
                joint_transition_prob[transit] = (
                    joint_transition_prob[transit]
                    / arrival_state_marginal[arrival_state]
                )

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
                self.transition_coef[transit] = (
                    joint_future_prob[transit] / self._marginal_prob[origin_state]
                )

        (
            self.states.mu_smooth[time_step],
            self.states.var_smooth[time_step],
            mu_states_marginal,
            var_states_marginal,
        ) = self._collapse_states(state_type="smooth", time_step=time_step)

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

        num_time_steps = len(data["y"])
        mu_obs_preds = []
        var_obs_preds = []
        mu_lstm_pred = None
        var_lstm_pred = None
        self.filter_marginal_prob_history = initialize_marginal_prob_history(
            num_time_steps
        )

        # Initialize hidden states
        self.set_same_states_transition_model()
        self.initialize_states_history(num_time_steps)

        for time_step, (x, y) in enumerate(zip(data["x"], data["y"])):
            if self.lstm_net:
                mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                    self.lstm_output_history, x
                )
                mu_lstm_pred, var_lstm_pred = self.lstm_net.forward(
                    mu_x=np.float32(mu_lstm_input), var_x=np.float32(var_lstm_input)
                )

            mu_obs_pred, var_obs_pred = self.forward(y, mu_lstm_pred, var_lstm_pred)
            self.backward(y)

            if self.lstm_states_index:
                self.update_lstm_output_history(
                    self._mu_states_posterior[self.lstm_states_index],
                    self.var_states_posterior[
                        self.lstm_states_index,
                        self.lstm_states_index,
                    ],
                )

            self.save_states_history(time_step)
            self.set_states()
            mu_obs_preds.append(mu_obs_pred)
            var_obs_preds.append(var_obs_pred)
            self.filter_marginal_prob_history["norm"][time_step] = self._marginal_prob[
                "norm"
            ].copy()
            self.filter_marginal_prob_history["abnorm"][time_step] = (
                self._marginal_prob["abnorm"].copy()
            )

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
