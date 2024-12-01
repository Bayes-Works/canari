import numpy as np
import copy
from typing import Tuple, List, Dict, Optional
import pytagi.metric as metric
from src.model import Model
import src.common as common
from src.data_struct import (
    SmootherStates,
    MarginalProbability,
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
        self.initialize_transition_model(
            norm_model,
            abnorm_model,
            std_transition_error,
            norm_to_abnorm_prob,
            abnorm_to_norm_prob,
            norm_model_prior_prob,
        )
        self.smoother_states = SmootherStates()
        self.marginal_prob = MarginalProbability()
        self.marginal_states = {"norm", "abnorm"}

    @staticmethod
    def initialize_transitions():
        """
        Create a dictionary for model transition
        """
        return {
            "norm_norm": None,
            "abnorm_abnorm": None,
            "norm_abnorm": None,
            "abnorm_norm": None,
        }

    @staticmethod
    def initialize_marginal():
        """
        Create a dictionary for models
        """
        return {
            "norm": None,
            "abnorm": None,
        }

    def initialize_transition_model(
        self,
        norm_model: Model,
        abnorm_model: Model,
        std_transition_error,
        norm_to_abnorm_prob,
        abnorm_to_norm_prob,
        norm_model_prior_prob,
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
            std_transition_error**2
        )

        # Store transitional models in a dictionary
        self.model = SKF.initialize_transitions()
        self.model["norm_norm"] = norm_model
        self.model["abnorm_abnorm"] = abnorm_model
        self.model["norm_abnorm"] = norm_abnorm
        self.model["abnorm_norm"] = abnorm_norm

        self.transition_prob_matrix = SKF.initialize_transitions()
        self.transition_prob_matrix["norm_norm"] = 1 - norm_to_abnorm_prob
        self.transition_prob_matrix["norm_abnorm"] = norm_to_abnorm_prob
        self.transition_prob_matrix["abnorm_norm"] = abnorm_to_norm_prob
        self.transition_prob_matrix["abnorm_abnorm"] = 1 - abnorm_to_norm_prob

        self.prob_model = SKF.initialize_marginal()
        self.prob_model["norm"] = norm_model_prior_prob
        self.prob_model["abnorm"] = 1 - norm_model_prior_prob

        self.coef_transition_model = SKF.initialize_transitions()
        self.likelihood = SKF.initialize_transitions()

        self.index_pad_state = self.model["norm_norm"].index_pad_state
        self.lstm_states_index = self.model["norm_norm"].lstm_states_index
        self.num_states = self.model["norm_norm"].num_states
        self.states_name = self.model["norm_norm"].states_name

    def auto_initialize_baseline_states(self, y: np.ndarray):
        """
        Automatically initialize baseline states from data for normal model
        """

        self.model["norm_norm"].auto_initialize_baseline_states(y)

    def initialize_model_states(self):
        # TODO: clean: set mean var for acceleration
        mean = 1e-10
        var = 1e-10
        for transition_model in self.model.values():
            transition_model.set_states(
                self.model["norm_norm"].mu_states, self.model["norm_norm"].var_states
            )
            transition_model.mu_states[self.index_pad_state] = mean
            transition_model.var_states[self.index_pad_state, self.index_pad_state] = (
                var
            )

    def save_for_smoother(self, time_step: int):
        """
        Save states' priors, posteriors and cross-covariances for smoother
        """

        for transition_model in self.model.values():
            transition_model.save_for_smoother(time_step)

        self.smoother_states.mu_prior[time_step] = copy.copy(
            self.mu_states_prior.flatten()
        )
        self.smoother_states.var_prior[time_step] = copy.copy(self.var_states_prior)
        self.smoother_states.mu_posterior[time_step] = copy.copy(
            self.mu_states_posterior.flatten()
        )
        self.smoother_states.var_posterior[time_step] = copy.copy(
            self.var_states_posterior
        )

    def initialize_smoother_states(self, num_time_steps: int):
        for transition_model in self.model.values():
            transition_model.initialize_smoother_states(num_time_steps)
        self.smoother_states.initialize(num_time_steps, self.num_states)

    def initialize_smoother_buffers(self):
        for origin_state in self.marginal_states:
            for arrival_state in self.marginal_states:
                transit = f"{origin_state}_{arrival_state}"
                marginal_arrival = f"{arrival_state}_{arrival_state}"
                self.model[transit].smoother_states.mu_smooth[-1] = self.model[
                    marginal_arrival
                ].smoother_states.mu_posterior[-1]
                self.model[transit].smoother_states.var_smooth[-1] = self.model[
                    marginal_arrival
                ].smoother_states.var_posterior[-1]
        # TODO
        self.smoother_states.mu_smooth[-1], self.smoother_states.var_smooth[-1] = (
            common.gaussian_mixture(
                self.model["norm_norm"].smoother_states.mu_posterior[-1],
                self.model["norm_norm"].smoother_states.var_posterior[-1],
                self.marginal_prob.norm[-1],
                self.model["abnorm_abnorm"].smoother_states.mu_posterior[-1],
                self.model["abnorm_abnorm"].smoother_states.var_posterior[-1],
                self.marginal_prob.abnorm[-1],
            )
        )

    def set_states(self):
        for transition_model in self.model.values():
            transition_model.set_states(
                transition_model.mu_states_posterior,
                transition_model.var_states_posterior,
            )

    def estimate_model_coef(
        self,
        obs,
        mu_pred_transit,
        var_pred_transit,
    ):
        # epsilon = 1e-10
        if np.isnan(obs):
            for transit in self.likelihood:
                self.likelihood[transit] = 1
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

                for transit in self.likelihood:
                    self.likelihood[transit] = np.mean(
                        np.exp(
                            metric.log_likelihood(
                                mu_pred_transit[transit] + noise,
                                obs,
                                (var_pred_transit[transit] - var_obs_error) ** 0.5,
                            )
                        )
                    )
            else:
                for transit in self.likelihood:
                    self.likelihood[transit] = np.exp(
                        metric.log_likelihood(
                            mu_pred_transit[transit],
                            obs,
                            var_pred_transit[transit] ** 0.5,
                        )
                    )

        #
        trans_prob = SKF.initialize_transitions()
        sum_trans_prob = 0
        for origin_state in self.marginal_states:
            for arrival_state in self.marginal_states:
                transit = f"{origin_state}_{arrival_state}"
                trans_prob[transit] = (
                    self.likelihood[transit]
                    * self.transition_prob_matrix[transit]
                    * self.prob_model[origin_state]
                )
                sum_trans_prob += trans_prob[transit]
        for transit in trans_prob:
            trans_prob[transit] = trans_prob[transit] / sum_trans_prob

        #
        self.prob_model["norm"] = trans_prob["norm_norm"] + trans_prob["abnorm_norm"]
        self.prob_model["abnorm"] = (
            trans_prob["abnorm_abnorm"] + trans_prob["norm_abnorm"]
        )
        #
        for origin_state in self.marginal_states:
            for arrival_state in self.marginal_states:
                transit = f"{origin_state}_{arrival_state}"
                self.coef_transition_model[transit] = (
                    trans_prob[transit] / self.prob_model[arrival_state]
                )

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, SmootherStates]:
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

        mu_pred_transit = SKF.initialize_transitions()
        var_pred_transit = SKF.initialize_transitions()

        for transit, transition_model in self.model.items():
            (
                mu_pred_transit[transit],
                var_pred_transit[transit],
            ) = transition_model.forward(mu_lstm_pred, var_lstm_pred)

        self.estimate_model_coef(
            y,
            mu_pred_transit,
            var_pred_transit,
        )

        mu_states = SKF.initialize_marginal()
        var_states = SKF.initialize_marginal()

        # Collapse
        mu_states["norm"], var_states["norm"] = common.gaussian_mixture(
            self.model["norm_norm"].mu_states_prior,
            self.model["norm_norm"].var_states_prior,
            self.coef_transition_model["norm_norm"],
            self.model["abnorm_norm"].mu_states_prior,
            self.model["abnorm_norm"].var_states_prior,
            self.coef_transition_model["abnorm_norm"],
        )

        mu_states["abnorm"], var_states["abnorm"] = common.gaussian_mixture(
            self.model["norm_abnorm"].mu_states_prior,
            self.model["norm_abnorm"].var_states_prior,
            self.coef_transition_model["norm_abnorm"],
            self.model["abnorm_abnorm"].mu_states_prior,
            self.model["abnorm_abnorm"].var_states_prior,
            self.coef_transition_model["abnorm_abnorm"],
        )

        self.mu_states_prior, self.var_states_prior = common.gaussian_mixture(
            mu_states["norm"],
            var_states["norm"],
            self.prob_model["norm"],
            mu_states["abnorm"],
            var_states["abnorm"],
            self.prob_model["abnorm"],
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

        mu_states = SKF.initialize_marginal()
        var_states = SKF.initialize_marginal()

        # Collapse
        mu_states["norm"], var_states["norm"] = common.gaussian_mixture(
            self.model["norm_norm"].mu_states_posterior,
            self.model["norm_norm"].var_states_posterior,
            self.coef_transition_model["norm_norm"],
            self.model["abnorm_norm"].mu_states_posterior,
            self.model["abnorm_norm"].var_states_posterior,
            self.coef_transition_model["abnorm_norm"],
        )
        mu_states["abnorm"], var_states["abnorm"] = common.gaussian_mixture(
            self.model["norm_abnorm"].mu_states_posterior,
            self.model["norm_abnorm"].var_states_posterior,
            self.coef_transition_model["norm_abnorm"],
            self.model["abnorm_abnorm"].mu_states_posterior,
            self.model["abnorm_abnorm"].var_states_posterior,
            self.coef_transition_model["abnorm_abnorm"],
        )

        self.mu_states_posterior, self.var_states_posterior = common.gaussian_mixture(
            mu_states["norm"],
            var_states["norm"],
            self.prob_model["norm"],
            mu_states["abnorm"],
            var_states["abnorm"],
            self.prob_model["abnorm"],
        )

        # Reassign posterior states
        for origin_state in self.marginal_states:
            for arrival_state in self.marginal_states:
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

        joint_transition_prob = SKF.initialize_transitions()
        arrival_state_marginal = SKF.initialize_marginal()

        for origin_state in self.marginal_states:
            for arrival_state in self.marginal_states:
                transit = f"{origin_state}_{arrival_state}"
                prob_origin_model = getattr(self.marginal_prob, origin_state)
                joint_transition_prob[transit] = (
                    prob_origin_model[time_step] * self.transition_prob_matrix[transit]
                )

        arrival_state_marginal["norm"] = (
            joint_transition_prob["norm_norm"] + joint_transition_prob["abnorm_norm"]
        )
        arrival_state_marginal["abnorm"] = (
            joint_transition_prob["norm_abnorm"]
            + joint_transition_prob["abnorm_abnorm"]
        )

        for origin_state in self.marginal_states:
            for arrival_state in self.marginal_states:
                transit = f"{origin_state}_{arrival_state}"
                joint_transition_prob[transit] = joint_transition_prob[
                    transit
                ] / np.maximum(arrival_state_marginal[arrival_state], epsilon)

        joint_future_prob = SKF.initialize_transitions()
        for origin_state in self.marginal_states:
            for arrival_state in self.marginal_states:
                transit = f"{origin_state}_{arrival_state}"
                prob_arrival_model = getattr(self.marginal_prob, arrival_state)
                joint_future_prob[transit] = (
                    joint_transition_prob[transit] * prob_arrival_model[time_step + 1]
                )

        smoother_state_prob = SKF.initialize_marginal()
        smoother_state_prob["norm"] = (
            joint_future_prob["norm_norm"] + joint_future_prob["norm_abnorm"]
        )
        smoother_state_prob["abnorm"] = (
            joint_future_prob["abnorm_norm"] + joint_future_prob["abnorm_abnorm"]
        )

        self.marginal_prob.norm[time_step] = copy.copy(smoother_state_prob["norm"])
        self.marginal_prob.abnorm[time_step] = copy.copy(smoother_state_prob["abnorm"])

        coef_transition_model = SKF.initialize_transitions()
        for origin_state in self.marginal_states:
            for arrival_state in self.marginal_states:
                transit = f"{origin_state}_{arrival_state}"
                prob_arrival_model = getattr(self.marginal_prob, arrival_state)
                coef_transition_model[transit] = joint_future_prob[
                    transit
                ] / np.maximum(smoother_state_prob[origin_state], epsilon)

        mu_states = SKF.initialize_marginal()
        var_states = SKF.initialize_marginal()
        # Collapse
        (mu_states["norm"], var_states["norm"]) = common.gaussian_mixture(
            self.model["norm_norm"].smoother_states.mu_smooth[time_step],
            self.model["norm_norm"].smoother_states.var_smooth[time_step],
            coef_transition_model["norm_norm"],
            self.model["norm_abnorm"].smoother_states.mu_smooth[time_step],
            self.model["norm_abnorm"].smoother_states.var_smooth[time_step],
            coef_transition_model["norm_abnorm"],
        )

        (mu_states["abnorm"], var_states["abnorm"]) = common.gaussian_mixture(
            self.model["abnorm_norm"].smoother_states.mu_smooth[time_step],
            self.model["abnorm_norm"].smoother_states.var_smooth[time_step],
            coef_transition_model["abnorm_norm"],
            self.model["abnorm_abnorm"].smoother_states.mu_smooth[time_step],
            self.model["abnorm_abnorm"].smoother_states.var_smooth[time_step],
            coef_transition_model["abnorm_abnorm"],
        )

        (
            self.smoother_states.mu_smooth[time_step],
            self.smoother_states.var_smooth[time_step],
        ) = common.gaussian_mixture(
            mu_states["norm"],
            var_states["norm"],
            smoother_state_prob["norm"],
            mu_states["abnorm"],
            var_states["abnorm"],
            smoother_state_prob["abnorm"],
        )

        for origin_state in self.marginal_states:
            for arrival_state in self.marginal_states:
                transit = f"{origin_state}_{arrival_state}"
                self.model[transit].smoother_states.mu_smooth[time_step] = mu_states[
                    arrival_state
                ]
                self.model[transit].smoother_states.var_smooth[time_step] = var_states[
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
        self.marginal_prob.initialize(num_time_steps + 1)

        # Initialize hidden states
        self.initialize_model_states()
        self.initialize_smoother_states(num_time_steps + 1)

        for time_step, (x, y) in enumerate(zip(data["x"], data["y"])):
            if self.model["norm_norm"].lstm_net:
                mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                    self.model["norm_norm"].lstm_output_history, x
                )
                mu_lstm_pred, var_lstm_pred = self.model["norm_norm"].lstm_net.forward(
                    mu_x=mu_lstm_input, var_x=var_lstm_input
                )

                # mu_lstm_pred = self.lstm_pred[time_step, 0]
                # var_lstm_pred = self.lstm_pred[time_step, 1]
                # y = self.lstm_pred[time_step, 2]

            mu_obs_pred, var_obs_pred = self.forward(y, mu_lstm_pred, var_lstm_pred)
            self.backward(y)

            if self.lstm_states_index:
                self.model["norm_norm"].update_lstm_output_history(
                    self.mu_states_posterior[self.lstm_states_index],
                    self.var_states_posterior[
                        self.lstm_states_index,
                        self.lstm_states_index,
                    ],
                )

            self.save_for_smoother(time_step + 1)
            self.set_states()
            mu_obs_preds.append(mu_obs_pred)
            var_obs_preds.append(var_obs_pred)
            self.marginal_prob.norm[time_step + 1] = self.prob_model["norm"]
            self.marginal_prob.abnorm[time_step + 1] = self.prob_model["abnorm"]

        return (
            mu_obs_preds,
            var_obs_preds,
            self.marginal_prob.abnorm[1:],
            self.smoother_states,
        )

    def smoother(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smoother for whole time series
        """

        num_time_steps = len(data["y"])
        self.initialize_model_states()
        self.initialize_smoother_states(num_time_steps + 1)

        # Filter
        mu_obs_preds, var_obs_preds, _, _ = self.filter(data)

        # Smoother
        self.initialize_smoother_buffers()
        for time_step in reversed(range(0, num_time_steps)):
            self.rts_smoother(time_step)

        return (
            mu_obs_preds,
            var_obs_preds,
            self.marginal_prob.abnorm[1:],
            self.smoother_states,
        )
