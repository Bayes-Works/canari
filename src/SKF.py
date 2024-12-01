import numpy as np
import copy
from typing import Tuple, List, Dict, Optional
import pytagi.metric as metric
from src.model import Model
import src.common as common
from src.data_struct import (
    StatesHistory,
    MarginalProbability,
    initialize_transition,
    initialize_marginal,
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
        self.transition_likelihood = initialize_transition()

        self.define_lstm_network()

        self.marginal_list = {"norm", "abnorm"}
        self.marginal_prob = initialize_marginal()
        self.marginal_prob["norm"] = norm_model_prior_prob
        self.marginal_prob["abnorm"] = 1 - norm_model_prior_prob
        self.marginal_prob_history = MarginalProbability()
        self.states = StatesHistory()

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
        for transition_model in self.model.values():
            transition_model.set_states(
                self.model["norm_norm"].mu_states, self.model["norm_norm"].var_states
            )

    def save_states_history(self, time_step: int):
        """
        Save states' priors, posteriors and cross-covariances for smoother
        """

        for transition_model in self.model.values():
            transition_model.save_states_history(time_step)

        self.states.mu_prior[time_step] = copy.copy(self.mu_states_prior.flatten())
        self.states.var_prior[time_step] = copy.copy(self.var_states_prior)
        self.states.mu_posterior[time_step] = copy.copy(
            self.mu_states_posterior.flatten()
        )
        self.states.var_posterior[time_step] = copy.copy(self.var_states_posterior)

    def initialize_states_history(self, num_time_steps: int):
        for transition_model in self.model.values():
            transition_model.initialize_states_history(num_time_steps)
        self.states.initialize(num_time_steps, self.num_states)

    def initialize_smoother_buffers(self):
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
        # TODO
        self.states.mu_smooth[-1], self.states.var_smooth[-1] = common.gaussian_mixture(
            self.model["norm_norm"].states.mu_posterior[-1],
            self.model["norm_norm"].states.var_posterior[-1],
            self.marginal_prob_history.norm[-1],
            self.model["abnorm_abnorm"].states.mu_posterior[-1],
            self.model["abnorm_abnorm"].states.var_posterior[-1],
            self.marginal_prob_history.abnorm[-1],
        )

    def set_states(self):
        for transition_model in self.model.values():
            transition_model.set_states(
                transition_model.mu_states_posterior,
                transition_model.var_states_posterior,
            )

    def estimate_transition_coef(
        self,
        obs,
        mu_pred_transit,
        var_pred_transit,
    ):
        # epsilon = 1e-10
        if np.isnan(obs):
            for transit in self.transition_likelihood:
                self.transition_likelihood[transit] = 1
        else:
            if self.conditional_likelihood:
                num_noise_realization = 10
                white_noise_index = self.states_name.index("white noise")
                var_obs_error = self.model["norm_norm"].process_noise_matrix[
                    white_noise_index, white_noise_index
                ]
                # var_obs_error = self.obs_noise**2
                noise = np.random.normal(
                    0, var_obs_error**0.5, (num_noise_realization, 1)
                )

                for transit in self.transition_likelihood:
                    self.transition_likelihood[transit] = np.mean(
                        np.exp(
                            metric.log_likelihood(
                                mu_pred_transit[transit] + noise,
                                obs,
                                (var_pred_transit[transit] - var_obs_error) ** 0.5,
                            )
                        )
                    )
            else:
                for transit in self.transition_likelihood:
                    self.transition_likelihood[transit] = np.exp(
                        metric.log_likelihood(
                            mu_pred_transit[transit],
                            obs,
                            var_pred_transit[transit] ** 0.5,
                        )
                    )

        #
        trans_prob = initialize_transition()
        sum_trans_prob = 0
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                trans_prob[transit] = (
                    self.transition_likelihood[transit]
                    * self.transition_prob[transit]
                    * self.marginal_prob[origin_state]
                )
                sum_trans_prob += trans_prob[transit]
        for transit in trans_prob:
            trans_prob[transit] = trans_prob[transit] / sum_trans_prob

        #
        self.marginal_prob["norm"] = trans_prob["norm_norm"] + trans_prob["abnorm_norm"]
        self.marginal_prob["abnorm"] = (
            trans_prob["abnorm_abnorm"] + trans_prob["norm_abnorm"]
        )
        #
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                self.transition_coef[transit] = (
                    trans_prob[transit] / self.marginal_prob[arrival_state]
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
    ):
        """
        Forward pass for 4 sub-models
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

        mu_states = initialize_marginal()
        var_states = initialize_marginal()

        # Collapse
        mu_states["norm"], var_states["norm"] = common.gaussian_mixture(
            self.model["norm_norm"].mu_states_prior,
            self.model["norm_norm"].var_states_prior,
            self.transition_coef["norm_norm"],
            self.model["abnorm_norm"].mu_states_prior,
            self.model["abnorm_norm"].var_states_prior,
            self.transition_coef["abnorm_norm"],
        )

        mu_states["abnorm"], var_states["abnorm"] = common.gaussian_mixture(
            self.model["norm_abnorm"].mu_states_prior,
            self.model["norm_abnorm"].var_states_prior,
            self.transition_coef["norm_abnorm"],
            self.model["abnorm_abnorm"].mu_states_prior,
            self.model["abnorm_abnorm"].var_states_prior,
            self.transition_coef["abnorm_abnorm"],
        )

        self.mu_states_prior, self.var_states_prior = common.gaussian_mixture(
            mu_states["norm"],
            var_states["norm"],
            self.marginal_prob["norm"],
            mu_states["abnorm"],
            var_states["abnorm"],
            self.marginal_prob["abnorm"],
        )

        mu_obs_pred, var_obs_pred = common.calc_observation(
            self.mu_states_prior,
            self.var_states_prior,
            self.model["norm_norm"].observation_matrix,
        )

        # TODO: better names for transition_pred, collapse_pred, obs_pred
        return mu_obs_pred, var_obs_pred

    def backward(
        self,
        obs: float,
    ) -> None:
        """
        Update step in states-space model
        """

        for transition_model in self.model.values():
            mu_delta, var_delta = transition_model.backward(obs)
            transition_model.estimate_posterior_states(mu_delta, var_delta)

        mu_states = initialize_marginal()
        var_states = initialize_marginal()

        # Collapse
        mu_states["norm"], var_states["norm"] = common.gaussian_mixture(
            self.model["norm_norm"].mu_states_posterior,
            self.model["norm_norm"].var_states_posterior,
            self.transition_coef["norm_norm"],
            self.model["abnorm_norm"].mu_states_posterior,
            self.model["abnorm_norm"].var_states_posterior,
            self.transition_coef["abnorm_norm"],
        )
        mu_states["abnorm"], var_states["abnorm"] = common.gaussian_mixture(
            self.model["norm_abnorm"].mu_states_posterior,
            self.model["norm_abnorm"].var_states_posterior,
            self.transition_coef["norm_abnorm"],
            self.model["abnorm_abnorm"].mu_states_posterior,
            self.model["abnorm_abnorm"].var_states_posterior,
            self.transition_coef["abnorm_abnorm"],
        )

        self.mu_states_posterior, self.var_states_posterior = common.gaussian_mixture(
            mu_states["norm"],
            var_states["norm"],
            self.marginal_prob["norm"],
            mu_states["abnorm"],
            var_states["abnorm"],
            self.marginal_prob["abnorm"],
        )

        # Reassign posterior states
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                self.model[transit].set_posterior_states(
                    mu_states[origin_state], var_states[origin_state]
                )

    def rts_smoother(self, time_step: int):
        """
        RTS smoother for each sub-model
        """

        for transition_model in self.model.values():
            transition_model.rts_smoother(time_step, rcond=1e-3)

        # TODO: if epsilon is needed
        epsilon = 1e-10

        joint_transition_prob = initialize_transition()
        arrival_state_marginal = initialize_marginal()

        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                prob_origin_model = getattr(self.marginal_prob_history, origin_state)
                joint_transition_prob[transit] = (
                    prob_origin_model[time_step] * self.transition_prob[transit]
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
                prob_arrival_model = getattr(self.marginal_prob_history, arrival_state)
                joint_future_prob[transit] = (
                    joint_transition_prob[transit] * prob_arrival_model[time_step + 1]
                )

        smoother_state_prob = initialize_marginal()
        smoother_state_prob["norm"] = (
            joint_future_prob["norm_norm"] + joint_future_prob["norm_abnorm"]
        )
        smoother_state_prob["abnorm"] = (
            joint_future_prob["abnorm_norm"] + joint_future_prob["abnorm_abnorm"]
        )

        self.marginal_prob_history.norm[time_step] = copy.copy(
            smoother_state_prob["norm"]
        )
        self.marginal_prob_history.abnorm[time_step] = copy.copy(
            smoother_state_prob["abnorm"]
        )

        transition_coef = initialize_transition()
        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                prob_arrival_model = getattr(self.marginal_prob_history, arrival_state)
                transition_coef[transit] = joint_future_prob[transit] / np.maximum(
                    smoother_state_prob[origin_state], epsilon
                )

        mu_states = initialize_marginal()
        var_states = initialize_marginal()
        # Collapse
        (mu_states["norm"], var_states["norm"]) = common.gaussian_mixture(
            self.model["norm_norm"].states.mu_smooth[time_step],
            self.model["norm_norm"].states.var_smooth[time_step],
            transition_coef["norm_norm"],
            self.model["norm_abnorm"].states.mu_smooth[time_step],
            self.model["norm_abnorm"].states.var_smooth[time_step],
            transition_coef["norm_abnorm"],
        )

        (mu_states["abnorm"], var_states["abnorm"]) = common.gaussian_mixture(
            self.model["abnorm_norm"].states.mu_smooth[time_step],
            self.model["abnorm_norm"].states.var_smooth[time_step],
            transition_coef["abnorm_norm"],
            self.model["abnorm_abnorm"].states.mu_smooth[time_step],
            self.model["abnorm_abnorm"].states.var_smooth[time_step],
            transition_coef["abnorm_abnorm"],
        )

        (
            self.states.mu_smooth[time_step],
            self.states.var_smooth[time_step],
        ) = common.gaussian_mixture(
            mu_states["norm"],
            var_states["norm"],
            smoother_state_prob["norm"],
            mu_states["abnorm"],
            var_states["abnorm"],
            smoother_state_prob["abnorm"],
        )

        for origin_state in self.marginal_list:
            for arrival_state in self.marginal_list:
                transit = f"{origin_state}_{arrival_state}"
                self.model[transit].states.mu_smooth[time_step] = mu_states[
                    arrival_state
                ]
                self.model[transit].states.var_smooth[time_step] = var_states[
                    arrival_state
                ]

    def filter(
        self,
        data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filtering
        """

        num_time_steps = len(data["y"])
        mu_obs_preds = []
        var_obs_preds = []
        mu_lstm_pred = None
        var_lstm_pred = None
        self.marginal_prob_history.initialize(num_time_steps)

        # Initialize hidden states
        self.set_same_states_transition_model()
        self.initialize_states_history(num_time_steps)

        for time_step, (x, y) in enumerate(zip(data["x"], data["y"])):
            if self.lstm_net:
                mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                    self.lstm_output_history, x
                )
                mu_lstm_pred, var_lstm_pred = self.lstm_net.forward(
                    mu_x=mu_lstm_input, var_x=var_lstm_input
                )

            mu_obs_pred, var_obs_pred = self.forward(y, mu_lstm_pred, var_lstm_pred)
            self.backward(y)

            if self.lstm_states_index:
                self.update_lstm_output_history(
                    self.mu_states_posterior[self.lstm_states_index],
                    self.var_states_posterior[
                        self.lstm_states_index,
                        self.lstm_states_index,
                    ],
                )

            self.save_states_history(time_step)
            self.set_states()
            mu_obs_preds.append(mu_obs_pred)
            var_obs_preds.append(var_obs_pred)
            self.marginal_prob_history.norm[time_step] = self.marginal_prob["norm"]
            self.marginal_prob_history.abnorm[time_step] = self.marginal_prob["abnorm"]

        return (
            mu_obs_preds,
            var_obs_preds,
            self.marginal_prob_history.abnorm,
            self.states,
        )

    def smoother(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smoother for whole time series
        """

        num_time_steps = len(data["y"])
        self.set_same_states_transition_model()
        self.initialize_states_history(num_time_steps)

        # Filter
        mu_obs_preds, var_obs_preds, _, _ = self.filter(data)

        # Smoother
        self.initialize_smoother_buffers()
        for time_step in reversed(range(0, num_time_steps - 1)):
            self.rts_smoother(time_step)

        return (
            mu_obs_preds,
            var_obs_preds,
            self.marginal_prob_history.abnorm,
            self.states,
        )
