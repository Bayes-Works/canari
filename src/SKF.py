import numpy as np
import copy
from typing import Tuple, List, Dict, Optional
import pytagi.metric as metric
from src.model import Model
import src.common as common
from src.data_struct import (
    SmootherStates,
    ModelProbability,
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
        self.initialize_SKF_models(
            norm_model,
            abnorm_model,
            std_transition_error,
            norm_to_abnorm_prob,
            abnorm_to_norm_prob,
            norm_model_prior_prob,
        )

        self.conditional_likelihood = conditional_likelihood
        self.smoother_states = SmootherStates()
        self.model_prob = None
        self.SKF_smoother = False

    @staticmethod
    def create_compatible_model(source: Model, target: Model) -> Model:
        """
        Create compatiable model by padding zero to states and matrices
        """

        pad_row = np.zeros((source.num_states)).flatten()
        pad_col = np.zeros((target.num_states)).flatten()
        for i, state in enumerate(target.states_name):
            if state not in source.states_name:
                source.mu_states = SKF.pad_matrix(
                    source.mu_states, i, pad_row=np.zeros(1)
                )
                source.var_states = SKF.pad_matrix(
                    source.var_states, i, pad_row, pad_col
                )
                source.transition_matrix = SKF.pad_matrix(
                    source.transition_matrix, i, pad_row, pad_col
                )
                source.process_noise_matrix = SKF.pad_matrix(
                    source.process_noise_matrix, i, pad_row, pad_col
                )
                source.observation_matrix = SKF.pad_matrix(
                    source.observation_matrix, i, pad_col=np.zeros(1)
                )
                source.num_states += 1
                source.states_name.insert(i, state)
                source.index_pad_state = i
        source.lstm_states_index = target.states_name.index("lstm")
        return source

    @staticmethod
    def pad_matrix(
        x: np.ndarray,
        idx,
        pad_row: Optional[np.ndarray] = None,
        pad_col: Optional[np.ndarray] = None,
    ):
        """
        Add padding for states
        """

        if pad_row is not None:
            x = np.insert(x, idx, pad_row, axis=0)
        if pad_col is not None:
            x = np.insert(x, idx, pad_col, axis=1)
        return x

    @staticmethod
    def duplicate_model(model):
        """
        Duplicate models
        """

        model_copy = Model()
        model_copy.transition_matrix = copy.copy(model.transition_matrix)
        model_copy.process_noise_matrix = copy.copy(model.process_noise_matrix)
        model_copy.observation_matrix = copy.copy(model.observation_matrix)
        model_copy.mu_states = copy.copy(model.mu_states)
        model_copy.var_states = copy.copy(model.var_states)
        model_copy.states_name = copy.copy(model.states_name)
        model_copy.num_states = copy.copy(model.num_states)
        model_copy.lstm_net = None
        model_copy.lstm_states_index = copy.copy(model.lstm_states_index)
        model_copy.states_name = copy.copy(model.states_name)
        return model_copy

    @staticmethod
    def gaussian_mixture(mu1, var1, coeff1, mu2, var2, coeff2):
        mu_mixture = mu1 * coeff1 + mu2 * coeff2
        m1 = mu1 - mu_mixture
        m2 = mu2 - mu_mixture
        var_mixture = coeff1 * (var1 + m1 @ m1.T) + coeff2 * (var2 + m2 @ m2.T)
        return mu_mixture, var_mixture

    @staticmethod
    # TODO: input for dimension
    def create_transition_dict():
        return {
            "norm_norm": 0,
            "abnorm_abnorm": 0,
            "norm_abnorm": 0,
            "abnorm_norm": 0,
        }

    @staticmethod
    # TODO: input for dimension
    def create_model_dict():
        return {
            "norm": 0,
            "abnorm": 0,
        }

    def initialize_SKF_models(
        self,
        norm_model: Model,
        abnorm_model: Model,
        std_transition_error,
        norm_to_abnorm_prob,
        abnorm_to_norm_prob,
        norm_model_prior_prob,
    ):
        """
        Create sub-models
        """

        # Normal to normal
        norm = SKF.create_compatible_model(
            source=norm_model,
            target=abnorm_model,
        )

        # Abnormal to normal
        abnorm_norm = SKF.duplicate_model(norm)

        #  Abnormal to abnormal
        abnorm = abnorm_model

        # Normal to abnormal
        norm_abnorm = SKF.duplicate_model(abnorm)
        index_pad_state = norm_model.index_pad_state
        norm_abnorm.process_noise_matrix[index_pad_state, index_pad_state] = (
            std_transition_error**2
        )

        # Store in dictionary
        self.model = {
            "norm_norm": norm,
            "abnorm_abnorm": abnorm,
            "norm_abnorm": norm_abnorm,
            "abnorm_norm": abnorm_norm,
        }

        # Others
        self.index_pad_state = self.model["norm_norm"].index_pad_state
        self.lstm_states_index = self.model["norm_norm"].lstm_states_index
        self.num_states = self.model["norm_norm"].num_states
        self.std_transition_error = std_transition_error

        self.transition_prob = SKF.create_transition_dict()
        self.transition_prob["norm_norm"] = 1 - norm_to_abnorm_prob
        self.transition_prob["norm_abnorm"] = norm_to_abnorm_prob
        self.transition_prob["abnorm_norm"] = abnorm_to_norm_prob
        self.transition_prob["abnorm_abnorm"] = 1 - abnorm_to_norm_prob

        self.prob_model = SKF.create_model_dict()
        self.prob_model["norm"] = norm_model_prior_prob
        self.prob_model["abnorm"] = 1 - norm_model_prior_prob

        self.coef_model = SKF.create_transition_dict()
        self.likelihood = SKF.create_transition_dict()
        self.states = {"norm", "abnorm"}

    def auto_initialize_baseline_states(self, y: np.ndarray):
        """
        Automatically initialize baseline states from data for normal model
        """

        self.model["norm_norm"].auto_initialize_baseline_states(y)

    def initialize_model_states(self):
        # TODO: clean: set mean var for acceleration
        mean = 0
        var = 0
        for model in self.model.values():
            model.set_states(
                self.model["norm_norm"].mu_states, self.model["norm_norm"].var_states
            )
            model.mu_states[self.index_pad_state] = mean
            model.var_states[self.index_pad_state, self.index_pad_state] = var

    def save_for_smoother(self, time_step: int):
        """
        Save states' priors, posteriors and cross-covariances for smoother
        """

        for model in self.model.values():
            model.save_for_smoother(time_step)

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
        for model in self.model.values():
            model.initialize_smoother_states(num_time_steps)

    def initialize_smoother_buffers(self):
        for model in self.model.values():
            model.initialize_smoother_buffers()
        # TODO
        self.smoother_states.mu_smooth[-1], self.smoother_states.var_smooth[-1] = (
            SKF.gaussian_mixture(
                self.model["norm_norm"].smoother_states.mu_posterior[-1],
                self.model["norm_norm"].smoother_states.var_posterior[-1],
                self.model_prob.norm[-1],
                self.model["abnorm_abnorm"].smoother_states.mu_posterior[-1],
                self.model["abnorm_abnorm"].smoother_states.var_posterior[-1],
                self.model_prob.abnorm[-1],
            )
        )

    def set_states(self):
        for model in self.model.values():
            model.set_states(model.mu_states_posterior, model.var_states_posterior)

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
                var_obs_error = self.model["norm_norm"].process_noise_matrix[-1, -1]
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
        trans_prob = SKF.create_transition_dict()
        sum_trans_prob = 0
        for origin_state in self.states:
            for arrival_state in self.states:
                transit = f"{origin_state}_{arrival_state}"
                trans_prob[transit] = (
                    self.likelihood[transit]
                    * self.transition_prob[transit]
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
        for origin_state in self.states:
            for arrival_state in self.states:
                transit = f"{origin_state}_{arrival_state}"
                self.coef_model[transit] = (
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

        mu_pred_transit = SKF.create_transition_dict()
        var_pred_transit = SKF.create_transition_dict()
        mu_states = SKF.create_transition_dict()
        var_states = SKF.create_transition_dict()

        for transit, model in self.model.items():
            (
                mu_pred_transit[transit],
                var_pred_transit[transit],
                mu_states[transit],
                var_states[transit],
            ) = model.forward(mu_lstm_pred, var_lstm_pred)

        self.estimate_model_coef(
            y,
            mu_pred_transit,
            var_pred_transit,
        )

        mu_pred = SKF.create_model_dict()
        var_pred = SKF.create_model_dict()

        # Collapse
        mu_pred["norm"], var_pred["norm"] = SKF.gaussian_mixture(
            self.model["norm_norm"].mu_states_prior,
            self.model["norm_norm"].var_states_prior,
            self.coef_model["norm_norm"],
            self.model["abnorm_norm"].mu_states_prior,
            self.model["abnorm_norm"].var_states_prior,
            self.coef_model["abnorm_norm"],
        )

        mu_pred["abnorm"], var_pred["abnorm"] = SKF.gaussian_mixture(
            self.model["norm_abnorm"].mu_states_prior,
            self.model["norm_abnorm"].var_states_prior,
            self.coef_model["norm_abnorm"],
            self.model["abnorm_abnorm"].mu_states_prior,
            self.model["abnorm_abnorm"].var_states_prior,
            self.coef_model["abnorm_abnorm"],
        )

        self.mu_states_prior, self.var_states_prior = SKF.gaussian_mixture(
            mu_pred["norm"],
            var_pred["norm"],
            self.prob_model["norm"],
            mu_pred["abnorm"],
            var_pred["abnorm"],
            self.prob_model["abnorm"],
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
    ) -> None:
        """
        Update step in states-space model
        """

        for model in self.model.values():
            mu_delta, var_delta = model.backward(obs)
            model.estimate_posterior_states(mu_delta, var_delta)

        mu_pred = SKF.create_model_dict()
        var_pred = SKF.create_model_dict()

        # Collapse 11, 21
        mu_pred["norm"], var_pred["norm"] = SKF.gaussian_mixture(
            self.model["norm_norm"].mu_states_posterior,
            self.model["norm_norm"].var_states_posterior,
            self.coef_model["norm_norm"],
            self.model["abnorm_norm"].mu_states_posterior,
            self.model["abnorm_norm"].var_states_posterior,
            self.coef_model["abnorm_norm"],
        )

        # Collapse 21, 22
        mu_pred["abnorm"], var_pred["abnorm"] = SKF.gaussian_mixture(
            self.model["norm_abnorm"].mu_states_posterior,
            self.model["norm_abnorm"].var_states_posterior,
            self.coef_model["norm_abnorm"],
            self.model["abnorm_abnorm"].mu_states_posterior,
            self.model["abnorm_abnorm"].var_states_posterior,
            self.coef_model["abnorm_abnorm"],
        )

        self.mu_states_posterior, self.var_states_posterior = SKF.gaussian_mixture(
            mu_pred["norm"],
            var_pred["norm"],
            self.prob_model["norm"],
            mu_pred["abnorm"],
            var_pred["abnorm"],
            self.prob_model["abnorm"],
        )

        for origin_state in self.states:
            for arrival_state in self.states:
                transit = f"{origin_state}_{arrival_state}"
                self.model[transit].set_posterior_states(
                    mu_pred[arrival_state], var_pred[arrival_state]
                )

    def rts_smoother(self, time_step: int):
        """
        RTS smoother for each sub-model
        """

        for model in self.model.values():
            model.rts_smoother(time_step)

        epsilon = 1e-10

        U = SKF.create_transition_dict()
        temp_U = SKF.create_model_dict()

        for origin_state in self.states:
            for arrival_state in self.states:
                transit = f"{origin_state}_{arrival_state}"
                prob_origin_model = getattr(self.model_prob, origin_state)
                U[transit] = (
                    prob_origin_model[time_step] * self.transition_prob[transit]
                )

        temp_U["norm"] = U["norm_norm"] + U["abnorm_norm"]
        temp_U["abnorm"] = U["norm_abnorm"] + U["abnorm_abnorm"]
        for origin_state in self.states:
            for arrival_state in self.states:
                transit = f"{origin_state}_{arrival_state}"
                U[transit] = U[transit] / temp_U[arrival_state]

        _M = SKF.create_transition_dict()
        for origin_state in self.states:
            for arrival_state in self.states:
                transit = f"{origin_state}_{arrival_state}"
                prob_arrival_model = getattr(self.model_prob, arrival_state)
                _M[transit] = U[transit] * prob_arrival_model[time_step + 1]

        M = SKF.create_model_dict()
        M["norm"] = _M["norm_norm"] + _M["norm_abnorm"]
        M["abnorm"] = _M["abnorm_norm"] + _M["abnorm_abnorm"]

        self.model_prob.norm[time_step] = copy.copy(M["norm"])
        self.model_prob.abnorm[time_step] = copy.copy(M["abnorm"])

        coef_model = SKF.create_transition_dict()
        for origin_state in self.states:
            for arrival_state in self.states:
                transit = f"{origin_state}_{arrival_state}"
                prob_arrival_model = getattr(self.model_prob, arrival_state)
                coef_model[transit] = _M[transit] / M[origin_state]

        mu_states = SKF.create_model_dict()
        var_states = SKF.create_model_dict()
        # Collapse 11, 12
        (mu_states["norm"], var_states["norm"]) = SKF.gaussian_mixture(
            self.model["norm_norm"].smoother_states.mu_smooth[time_step],
            self.model["norm_norm"].smoother_states.var_smooth[time_step],
            coef_model["norm_norm"],
            self.model["norm_abnorm"].smoother_states.mu_smooth[time_step],
            self.model["norm_abnorm"].smoother_states.var_smooth[time_step],
            coef_model["norm_abnorm"],
        )

        # Collapse 21, 22
        (mu_states["abnorm"], var_states["abnorm"]) = SKF.gaussian_mixture(
            self.model["abnorm_norm"].smoother_states.mu_smooth[time_step],
            self.model["abnorm_norm"].smoother_states.var_smooth[time_step],
            coef_model["abnorm_norm"],
            self.model["abnorm_abnorm"].smoother_states.mu_smooth[time_step],
            self.model["abnorm_abnorm"].smoother_states.var_smooth[time_step],
            coef_model["abnorm_norm"],
        )

        for origin_state in self.states:
            for arrival_state in self.states:
                transit = f"{origin_state}_{arrival_state}"
                self.model[transit].smoother_states.mu_smooth[time_step] = mu_states[
                    origin_state
                ]
                self.model[transit].smoother_states.var_smooth[time_step] = var_states[
                    origin_state
                ]

        (
            self.smoother_states.mu_smooth[time_step],
            self.smoother_states.var_smooth[time_step],
        ) = SKF.gaussian_mixture(
            mu_states["norm"],
            var_states["norm"],
            M["norm"],
            mu_states["abnorm"],
            var_states["abnorm"],
            M["abnorm"],
        )

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
        self.model_prob = ModelProbability()
        self.model_prob.initialize(num_time_steps + 1)

        # Initialize hidden states
        self.initialize_model_states()
        self.initialize_smoother_states(num_time_steps + 1)
        self.smoother_states.initialize(num_time_steps + 1, self.num_states)

        for time_step, (x, y) in enumerate(zip(data["x"], data["y"])):
            if self.model["norm_norm"].lstm_net:
                mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                    self.model["norm_norm"].lstm_output_history, x
                )
                mu_lstm_pred, var_lstm_pred = self.model["norm_norm"].lstm_net.forward(
                    mu_x=mu_lstm_input, var_x=var_lstm_input
                )

            mu_obs_pred, var_obs_pred = self.forward(y, mu_lstm_pred, var_lstm_pred)
            self.backward(y)

            if self.model["norm_norm"].lstm_net:
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
            self.model_prob.norm[time_step + 1] = copy.copy(self.prob_model["norm"])
            self.model_prob.abnorm[time_step + 1] = copy.copy(self.prob_model["abnorm"])

        return (
            mu_obs_preds,
            var_obs_preds,
            self.model_prob.abnorm[1:],
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
            self.model_prob.abnorm[1:],
            self.smoother_states,
        )
