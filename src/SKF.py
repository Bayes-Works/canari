import numpy as np
import copy
from typing import Tuple, List, Dict, Optional
import pytagi.metric as metric
from src.model import Model
import src.common as common
from src.data_struct import (
    SmootherStates,
    StatesInTime,
    ModelTransition,
    ProbabilityModel,
)


class SKF:
    """ "
    Switching Kalman Filter
    """

    def __init__(
        self,
        normal_model: Model,
        abnormal_model: Model,
        std_transition_error: Optional[float] = 0.0,
        normal_to_abnormal_prob: Optional[float] = 1e-4,
        abnormal_to_normal_prob: Optional[float] = 0.1,
        normal_model_prior_prob: Optional[float] = 0.99,
        conditional_likelihood: Optional[bool] = True,
    ):
        self.initialize_SKF_models(normal_model, abnormal_model, std_transition_error)

        self.transition_prob = ModelTransition()
        self.transition_prob.norm_to_norm = 1 - normal_to_abnormal_prob
        self.transition_prob.norm_to_abnorm = normal_to_abnormal_prob
        self.transition_prob.abnorm_to_norm = abnormal_to_normal_prob
        self.transition_prob.abnorm_to_abnorm = 1 - abnormal_to_normal_prob

        self._model_prob = ProbabilityModel()
        self._model_prob.normal = normal_model_prior_prob
        self._model_prob.abnormal = 1 - normal_model_prior_prob

        self.coef_model = ModelTransition()
        self.likelihood = ModelTransition()
        #
        self.SKF_smoother = False
        self.model_prob = None
        self.std_transition_error = std_transition_error
        self.conditional_likelihood = conditional_likelihood
        self.states = StatesInTime()

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
                source.index_pad_states = i
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
    def gaussian_mixture(mu1, var1, coeff1, mu2, var2, coeff2):
        mu_mixture = mu1 * coeff1 + mu2 * coeff2
        m1 = mu1 - mu_mixture
        m2 = mu2 - mu_mixture
        var_mixture = coeff1 * (var1 + m1 @ m1.T) + coeff2 * (var2 + m2 @ m2.T)
        return mu_mixture.flatten(), var_mixture

    def initialize_SKF_models(
        self,
        normal_model: Model,
        abnormal_model: Model,
        std_transition_error: Optional[float] = 0.0,
    ):
        """
        Initial sub-models
        """

        # Normal to normal
        self.norm_model = SKF.create_compatible_model(
            source=normal_model,
            target=abnormal_model,
        )

        # Abnormal to abnormal
        self.abnorm_model = abnormal_model
        self.abnorm_model.lstm_states_index = self.norm_model.lstm_states_index

        # Abnormal to abnormal
        self.abnorm_to_norm_model = self.norm_model.duplicate_model()

        # Normal to abnormal
        self.norm_to_abnorm_model = self.abnorm_model.duplicate_model()
        index_pad_states = self.norm_model.index_pad_states
        self.norm_to_abnorm_model.process_noise_matrix[
            index_pad_states, index_pad_states
        ] = (std_transition_error**2)

    def auto_initialize_baseline_states(self, y: np.ndarray):
        """
        Automatically initialize baseline states from data for normal model
        """

        self.norm_model.auto_initialize_baseline_states(y)

    def save_for_smoother(self, time_step: int):
        """
        Save states' priors, posteriors and cross-covariances for smoother
        """
        self.norm_model.save_for_smoother(time_step)
        self.abnorm_model.save_for_smoother(time_step)
        self.norm_to_abnorm_model.save_for_smoother(time_step)
        self.abnorm_to_norm_model.save_for_smoother(time_step)

    def initialize_smoother_states(self, num_time_steps: int):
        self.SKF_smoother = True
        self.norm_model.initialize_smoother_states(num_time_steps)
        self.abnorm_model.initialize_smoother_states(num_time_steps)
        self.norm_to_abnorm_model.initialize_smoother_states(num_time_steps)
        self.abnorm_to_norm_model.initialize_smoother_states(num_time_steps)
        self.states.initialize(num_time_steps + 1, self.norm_model.num_states)

    def initialize_smoother_buffers(self):
        self.norm_model.initialize_smoother_buffers()
        self.norm_to_abnorm_model.initialize_smoother_buffers()
        self.abnorm_model.initialize_smoother_buffers()
        self.abnorm_to_norm_model.initialize_smoother_buffers()
        self.states.mu[-1], self.states.var[-1] = SKF.gaussian_mixture(
            self.norm_model.smoother_states.mu_posterior[-1],
            self.norm_model.smoother_states.var_posterior[-1],
            self.model_prob[-1, 0],
            self.abnorm_model.smoother_states.mu_posterior[-1],
            self.abnorm_model.smoother_states.var_posterior[-1],
            self.model_prob[-1, 1],
        )

    def initialize_model_states(self):
        self.abnorm_model.set_states(
            self.norm_model.mu_states, self.norm_model.var_states
        )
        self.norm_to_abnorm_model.set_states(
            self.norm_model.mu_states, self.norm_model.var_states
        )
        self.abnorm_to_norm_model.set_states(
            self.norm_model.mu_states, self.norm_model.var_states
        )

        self.abnorm_model.mu_states[2] = 1e-10
        self.abnorm_model.var_states[2, 2] = 1e-10
        self.abnorm_to_norm_model.mu_states[2] = 1e-10
        self.abnorm_to_norm_model.var_states[2, 2] = 1e-10

    def set_states(self):
        self.norm_model.set_states(
            self.norm_model.mu_states_posterior, self.norm_model.var_states_posterior
        )
        self.abnorm_model.set_states(
            self.abnorm_model.mu_states_posterior,
            self.abnorm_model.var_states_posterior,
        )
        self.norm_to_abnorm_model.set_states(
            self.norm_to_abnorm_model.mu_states_posterior,
            self.norm_to_abnorm_model.var_states_posterior,
        )
        self.abnorm_to_norm_model.set_states(
            self.abnorm_to_norm_model.mu_states_posterior,
            self.abnorm_to_norm_model.var_states_posterior,
        )

    def estimate_model_coef(
        self,
        obs,
        mu_pred_norm,
        var_pred_norm,
        mu_pred_norm_to_ab,
        var_pred_norm_to_ab,
        mu_pred_abnorm,
        var_pred_abnorm,
        mu_pred_ab_to_norm,
        var_pred_ab_to_norm,
    ):
        epsilon = 1e-10
        if np.isnan(obs):
            self.likelihood = ModelTransition()
        else:
            if self.conditional_likelihood:
                num_noise_realization = 10
                noise = np.random.normal(
                    0, self.std_transition_error, (num_noise_realization, 1)
                )
                self.likelihood.norm_to_norm = np.mean(
                    np.exp(
                        metric.log_likelihood(
                            mu_pred_norm + noise,
                            obs,
                            (var_pred_norm - self.std_transition_error**2) ** 0.5,
                        )
                    )
                )
                self.likelihood.norm_to_abnorm = np.mean(
                    np.exp(
                        metric.log_likelihood(
                            mu_pred_norm_to_ab + noise,
                            obs,
                            (var_pred_norm_to_ab - self.std_transition_error**2) ** 0.5,
                        )
                    )
                )
                self.likelihood.abnorm_to_norm = np.mean(
                    np.exp(
                        metric.log_likelihood(
                            mu_pred_ab_to_norm + noise,
                            obs,
                            (var_pred_ab_to_norm - self.std_transition_error**2) ** 0.5,
                        )
                    )
                )
                self.likelihood.abnorm_to_abnorm = np.mean(
                    np.exp(
                        metric.log_likelihood(
                            mu_pred_abnorm + noise,
                            obs,
                            (var_pred_abnorm - self.std_transition_error**2) ** 0.5,
                        )
                    )
                )
            else:
                self.likelihood.norm_to_norm = np.exp(
                    metric.log_likelihood(mu_pred_norm, obs, var_pred_norm**0.5)
                )
                self.likelihood.norm_to_abnorm = np.exp(
                    metric.log_likelihood(
                        mu_pred_norm_to_ab, obs, var_pred_norm_to_ab**0.5
                    )
                )
                self.likelihood.abnorm_to_norm = np.exp(
                    metric.log_likelihood(
                        mu_pred_ab_to_norm, obs, var_pred_ab_to_norm**0.5
                    )
                )
                self.likelihood.abnorm_to_abnorm = np.exp(
                    metric.log_likelihood(mu_pred_abnorm, obs, var_pred_abnorm**0.5)
                )

        transition_prob = ModelTransition()
        transition_prob.norm_to_norm = (
            self.likelihood.norm_to_norm
            * self.transition_prob.norm_to_norm
            * self._model_prob.normal
        )
        transition_prob.norm_to_abnorm = (
            self.likelihood.norm_to_abnorm
            * self.transition_prob.norm_to_abnorm
            * self._model_prob.normal
        )
        transition_prob.abnorm_to_norm = (
            self.likelihood.abnorm_to_norm
            * self.transition_prob.abnorm_to_norm
            * self._model_prob.abnormal
        )
        transition_prob.abnorm_to_abnorm = (
            self.likelihood.abnorm_to_abnorm
            * self.transition_prob.abnorm_to_abnorm
            * self._model_prob.abnormal
        )

        total_sum = (
            transition_prob.norm_to_norm
            + transition_prob.norm_to_abnorm
            + transition_prob.abnorm_to_norm
            + transition_prob.abnorm_to_abnorm
        )
        transition_prob.norm_to_norm = transition_prob.norm_to_norm / total_sum
        transition_prob.norm_to_abnorm = transition_prob.norm_to_abnorm / total_sum
        transition_prob.abnorm_to_norm = transition_prob.abnorm_to_norm / total_sum
        transition_prob.abnorm_to_abnorm = transition_prob.abnorm_to_abnorm / total_sum

        self._model_prob.normal = (
            transition_prob.norm_to_norm + transition_prob.abnorm_to_norm
        )
        self._model_prob.abnormal = (
            transition_prob.norm_to_abnorm + transition_prob.abnorm_to_abnorm
        )

        self.coef_model.norm_to_norm = transition_prob.norm_to_norm / np.maximum(
            self._model_prob.normal, epsilon
        )
        self.coef_model.norm_to_abnorm = transition_prob.norm_to_abnorm / np.maximum(
            self._model_prob.abnormal, epsilon
        )
        self.coef_model.abnorm_to_norm = transition_prob.abnorm_to_norm / np.maximum(
            self._model_prob.normal, epsilon
        )
        self.coef_model.abnorm_to_abnorm = (
            transition_prob.abnorm_to_abnorm
            / np.maximum(self._model_prob.abnormal, epsilon)
        )

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, SmootherStates]:
        """
        Train the LstmNetwork of the normal model
        """
        return self.norm_model.lstm_train(train_data, validation_data)

    def forward(
        self,
        y,
        mu_lstm_pred: Optional[np.ndarray] = None,
        var_lstm_pred: Optional[np.ndarray] = None,
    ):
        """
        Forward pass for 4 sub-models
        """

        # Normal to normal
        mu_pred_norm, var_pred_norm = self.norm_model.forward(
            mu_lstm_pred, var_lstm_pred
        )

        # Normal to abnormal
        mu_pred_norm_to_ab, var_pred_norm_to_ab = self.norm_to_abnorm_model.forward(
            mu_lstm_pred, var_lstm_pred
        )

        # Abnormal to abnormal
        mu_pred_abnorm, var_pred_abnorm = self.abnorm_model.forward(
            mu_lstm_pred, var_lstm_pred
        )

        # Abnormal to normal
        mu_pred_ab_to_norm, var_pred_ab_to_norm = self.abnorm_to_norm_model.forward(
            mu_lstm_pred, var_lstm_pred
        )

        self.estimate_model_coef(
            y,
            mu_pred_norm,
            var_pred_norm,
            mu_pred_norm_to_ab,
            var_pred_norm_to_ab,
            mu_pred_abnorm,
            var_pred_abnorm,
            mu_pred_ab_to_norm,
            var_pred_ab_to_norm,
        )

        # Collapse
        mu_states_normal, var_states_normal = SKF.gaussian_mixture(
            self.norm_model.mu_states_prior,
            self.norm_model.var_states_prior,
            self.coef_model.norm_to_norm,
            self.abnorm_to_norm_model.mu_states_prior,
            self.abnorm_to_norm_model.var_states_prior,
            self.coef_model.abnorm_to_norm,
        )

        mu_states_abnormal, var_states_abnormal = SKF.gaussian_mixture(
            self.norm_to_abnorm_model.mu_states_prior,
            self.norm_to_abnorm_model.var_states_prior,
            self.coef_model.norm_to_abnorm,
            self.abnorm_model.mu_states_prior,
            self.abnorm_model.var_states_prior,
            self.coef_model.abnorm_to_abnorm,
        )

        mu_states_mixture, var_states_mixture = SKF.gaussian_mixture(
            mu_states_normal,
            var_states_normal,
            self._model_prob.normal,
            mu_states_abnormal,
            var_states_abnormal,
            self._model_prob.abnormal,
        )

        mu_obs_pred, var_obs_pred = common.calc_observation(
            mu_states_mixture, var_states_mixture, self.abnorm_model.observation_matrix
        )

        return mu_obs_pred, var_obs_pred

    def backward(
        self,
        obs: float,
        time_step: int,
    ) -> None:
        """
        Update step in states-space model
        """

        # Normal to normal
        mu_delta_norm, var_delta_norm = self.norm_model.backward(obs)
        self.norm_model.estimate_posterior_states(mu_delta_norm, var_delta_norm)

        # Normal to abnormal
        mu_delta_norm_to_ab, var_delta_norm_to_ab = self.norm_to_abnorm_model.backward(
            obs
        )
        self.norm_to_abnorm_model.estimate_posterior_states(
            mu_delta_norm_to_ab, var_delta_norm_to_ab
        )

        # Abnormal to normal
        mu_delta_abnorm, var_delta_abnorm = self.abnorm_model.backward(
            obs,
        )
        self.abnorm_model.estimate_posterior_states(mu_delta_abnorm, var_delta_abnorm)

        # Abnormal to abnormal
        mu_delta_ab_to_norm, var_delta_ab_to_norm = self.abnorm_to_norm_model.backward(
            obs,
        )
        self.abnorm_to_norm_model.estimate_posterior_states(
            mu_delta_ab_to_norm, var_delta_ab_to_norm
        )

        # Collapse 11, 21
        mu_states_normal, var_states_normal = SKF.gaussian_mixture(
            self.norm_model.mu_states_posterior,
            self.norm_model.var_states_posterior,
            self.coef_model.norm_to_norm,
            self.abnorm_to_norm_model.mu_states_posterior,
            self.abnorm_to_norm_model.var_states_posterior,
            self.coef_model.abnorm_to_norm,
        )
        self.norm_model.set_posterior_states(mu_states_normal, var_states_normal)
        self.abnorm_to_norm_model.set_posterior_states(
            mu_states_normal, var_states_normal
        )

        # Collapse 21, 22
        mu_states_abnormal, var_states_abnormal = SKF.gaussian_mixture(
            self.norm_to_abnorm_model.mu_states_posterior,
            self.norm_to_abnorm_model.var_states_posterior,
            self.coef_model.norm_to_abnorm,
            self.abnorm_model.mu_states_posterior,
            self.abnorm_model.var_states_posterior,
            self.coef_model.abnorm_to_abnorm,
        )
        self.abnorm_model.set_posterior_states(mu_states_abnormal, var_states_abnormal)
        self.norm_to_abnorm_model.set_posterior_states(
            mu_states_abnormal, var_states_abnormal
        )

        # Collapse 1 & 2
        (
            self.states.mu[time_step],
            self.states.var[time_step],
        ) = SKF.gaussian_mixture(
            mu_states_normal,
            var_states_normal,
            self._model_prob.normal,
            mu_states_abnormal,
            var_states_abnormal,
            self._model_prob.abnormal,
        )

    def rts_smoother(self, time_step: int):
        """
        RTS smoother for each sub-model
        """

        self.norm_model.rts_smoother(time_step)
        self.norm_to_abnorm_model.rts_smoother(time_step)
        self.abnorm_model.rts_smoother(time_step)
        self.abnorm_to_norm_model.rts_smoother(time_step)

        epsilon = 1e-10
        U = ModelTransition()
        U.norm_to_norm = (
            self.model_prob[time_step, 0] * self.transition_prob.norm_to_norm
        )
        U.norm_to_abnorm = (
            self.model_prob[time_step, 0] * self.transition_prob.norm_to_abnorm
        )
        U.abnorm_to_norm = (
            self.model_prob[time_step, 1] * self.transition_prob.abnorm_to_norm
        )
        U.abnorm_to_abnorm = (
            self.model_prob[time_step, 1] * self.transition_prob.abnorm_to_abnorm
        )

        U.norm_to_norm = U.norm_to_norm / np.maximum(
            U.norm_to_norm + U.abnorm_to_norm, epsilon
        )
        U.abnorm_to_norm = U.abnorm_to_norm / np.maximum(
            U.norm_to_norm + U.abnorm_to_norm, epsilon
        )
        U.norm_to_abnorm = U.norm_to_abnorm / np.maximum(
            U.norm_to_abnorm + U.abnorm_to_abnorm, epsilon
        )
        U.abnorm_to_abnorm = U.abnorm_to_abnorm / np.maximum(
            U.norm_to_abnorm + U.abnorm_to_abnorm, epsilon
        )

        M_ = ModelTransition()
        M_.norm_to_norm = U.norm_to_norm * self.model_prob[time_step + 1, 0]
        M_.norm_to_abnorm = U.norm_to_abnorm * self.model_prob[time_step + 1, 1]
        M_.abnorm_to_norm = U.abnorm_to_norm * self.model_prob[time_step + 1, 0]
        M_.abnorm_to_abnorm = U.abnorm_to_abnorm * self.model_prob[time_step + 1, 1]

        M = ProbabilityModel()
        M.normal = M_.norm_to_norm + M_.norm_to_abnorm
        M.abnormal = M_.abnorm_to_norm + M_.abnorm_to_abnorm
        self.model_prob[time_step, 0] = copy.copy(M.normal)
        self.model_prob[time_step, 1] = copy.copy(M.abnormal)

        coef_model = ModelTransition()
        coef_model.norm_to_norm = M_.norm_to_norm / np.maximum(M.normal, epsilon)
        coef_model.norm_to_abnorm = M_.norm_to_abnorm / np.maximum(M.normal, epsilon)
        coef_model.abnorm_to_abnorm = M_.abnorm_to_abnorm / np.maximum(
            M.abnormal, epsilon
        )
        coef_model.abnorm_to_norm = M_.abnorm_to_norm / np.maximum(M.abnormal, epsilon)

        # Collapse 11, 12
        (mu_states_normal, var_states_normal) = SKF.gaussian_mixture(
            self.norm_model.smoother_states.mu_smooth[time_step],
            self.norm_model.smoother_states.var_smooth[time_step],
            coef_model.norm_to_norm,
            self.norm_to_abnorm_model.smoother_states.mu_smooth[time_step],
            self.norm_to_abnorm_model.smoother_states.var_smooth[time_step],
            coef_model.norm_to_abnorm,
        )
        self.norm_model.smoother_states.mu_smooth[time_step] = mu_states_normal
        self.norm_model.smoother_states.var_smooth[time_step] = var_states_normal
        self.norm_to_abnorm_model.smoother_states.mu_smooth[time_step] = (
            mu_states_normal
        )
        self.norm_to_abnorm_model.smoother_states.var_smooth[time_step] = (
            var_states_normal
        )

        # Collapse 21, 22
        (mu_states_abnormal, var_states_abnormal) = SKF.gaussian_mixture(
            self.abnorm_to_norm_model.smoother_states.mu_smooth[time_step],
            self.abnorm_to_norm_model.smoother_states.var_smooth[time_step],
            coef_model.abnorm_to_norm,
            self.abnorm_model.smoother_states.mu_smooth[time_step],
            self.abnorm_model.smoother_states.var_smooth[time_step],
            coef_model.abnorm_to_abnorm,
        )
        self.abnorm_model.smoother_states.mu_smooth[time_step] = mu_states_abnormal
        self.abnorm_model.smoother_states.var_smooth[time_step] = var_states_abnormal
        self.abnorm_to_norm_model.smoother_states.mu_smooth[time_step] = (
            mu_states_abnormal
        )
        self.abnorm_to_norm_model.smoother_states.var_smooth[time_step] = (
            var_states_abnormal
        )

        # Collapse 1 & 2
        (
            self.states.mu[time_step],
            self.states.var[time_step],
        ) = SKF.gaussian_mixture(
            mu_states_normal,
            var_states_normal,
            M.normal,
            mu_states_abnormal,
            var_states_abnormal,
            M.abnormal,
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
        self.model_prob = np.zeros((num_time_steps + 1, 2), dtype=np.float32)

        # Initialize hidden states
        self.initialize_model_states()
        self.states.initialize(num_time_steps + 1, self.norm_model.num_states)

        for time_step, (x, y) in enumerate(zip(data["x"], data["y"])):
            if self.norm_model.lstm_net:
                mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                    self.norm_model.lstm_output_history, x
                )
                mu_lstm_pred, var_lstm_pred = self.norm_model.lstm_net.forward(
                    mu_x=mu_lstm_input, var_x=var_lstm_input
                )

            mu_obs_pred, var_obs_pred = self.forward(y, mu_lstm_pred, var_lstm_pred)
            self.backward(y, time_step + 1)

            if self.norm_model.lstm_net:
                self.norm_model.update_lstm_output_history(
                    self.norm_model.mu_states_posterior[
                        self.norm_model.lstm_states_index
                    ],
                    self.norm_model.var_states_posterior[
                        self.norm_model.lstm_states_index,
                        self.norm_model.lstm_states_index,
                    ],
                )

            if self.SKF_smoother:
                self.save_for_smoother(time_step + 1)

            self.set_states()
            mu_obs_preds.append(mu_obs_pred)
            var_obs_preds.append(var_obs_pred)
            self.model_prob[time_step + 1, 0] = copy.copy(self._model_prob.normal)
            self.model_prob[time_step + 1, 1] = copy.copy(self._model_prob.abnormal)

        return mu_obs_preds, var_obs_preds, self.model_prob[1:, 1], self.states

    def smoother(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smoother for whole time series
        """

        num_time_steps = len(data["y"])
        self.initialize_model_states()
        self.initialize_smoother_states(num_time_steps)

        # Filter
        mu_obs_preds, var_obs_preds, _, _ = self.filter(data)

        # Smoother
        self.initialize_smoother_buffers()
        for time_step in reversed(range(0, num_time_steps)):
            self.rts_smoother(time_step)

        return mu_obs_preds, var_obs_preds, self.model_prob[1:, 1], self.states
