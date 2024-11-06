import numpy as np
import copy
from typing import Tuple, List, Dict, Optional
from src.model import Model
import src.common as common
from src.data_struct import SmootherStates
import pytagi.metric as metric


class SKF:
    """ "
    Switching Kalman Filter
    """

    def __init__(
        self,
        normal_model: Model,
        abnormal_model: Model,
        std_transition_error: Optional[float] = 0.0,
        prob_norm_to_ab: Optional[float] = 1e-3,
        prob_ab_to_norm: Optional[float] = 1e-3,
        prob_norm: Optional[float] = 0.99,
    ):
        self.norm_model = normal_model
        self.abnorm_model = abnormal_model
        self.std_transition_error = std_transition_error
        self.transition_prob_matrix = np.array(
            [
                [1 - prob_norm_to_ab, prob_norm_to_ab],
                [prob_ab_to_norm, 1 - prob_ab_to_norm],
            ]
        )
        self.prob_model = np.array([[prob_norm], [1 - prob_norm]])
        self.transition_prob = np.zeros((2, 2))
        self.likelihood = np.zeros((2, 2))

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, SmootherStates]:
        """
        Train the LstmNetwork of the normal model
        """
        return self.norm_model.lstm_train(train_data, validation_data)

    @staticmethod
    def create_compatible_model(source: Model, target: Model) -> Model:
        """ "
        Create compatiable model by padding zero to states and matrices
        """
        # TODO: should not be like this, this class
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
    def mixture(mu1, var1, coeff1, mu2, var2, coeff2):
        mu_mixture = mu1 * coeff1 + mu2 * coeff2
        m1 = mu1 - mu_mixture
        m2 = mu2 - mu_mixture
        var_mixture = coeff1 * (var1 + m1 @ m1.T) + coeff2 * (var2 + m2 @ m2.T)
        return mu_mixture, var_mixture

    def collapse(self):
        coeff_norm = self.transition_prob[:, 0] / self.prob_model[0]
        mu_mixture_norm, var_mixture_norm = SKF.mixture(
            self.norm_model.mu_states,
            self.norm_model.var_states,
            coeff_norm[0],
            self.abnorm_to_norm_model.mu_states,
            self.abnorm_to_norm_model.var_states,
            coeff_norm[1],
        )
        coeff_abnorm = self.transition_prob[:, 1] / self.prob_model[1]
        mu_mixture_abnorm, var_mixture_abnorm = SKF.mixture(
            self.norm_to_abnorm_model.mu_states,
            self.norm_to_abnorm_model.var_states,
            coeff_abnorm[0],
            self.abnorm_model.mu_states,
            self.abnorm_model.var_states,
            coeff_abnorm[1],
        )
        mu_pred_norm, var_pred_norm = common.calc_observation(
            mu_mixture_norm, var_mixture_norm, self.norm_model.observation_matrix
        )
        mu_pred_abnorm, var_pred_abnorm = common.calc_observation(
            mu_mixture_abnorm, var_mixture_abnorm, self.abnorm_model.observation_matrix
        )

        self.norm_model.mu_states = mu_mixture_norm
        self.norm_model.var_states = var_mixture_norm
        self.norm_to_abnorm_model.mu_states = mu_mixture_norm
        self.norm_to_abnorm_model.var_states = var_mixture_norm
        self.abnorm_model.mu_states = mu_mixture_abnorm
        self.abnorm_model.var_states = var_mixture_abnorm
        self.norm_to_abnorm_model.mu_states = mu_mixture_abnorm
        self.norm_to_abnorm_model.var_states = var_mixture_abnorm

        return mu_pred_norm, var_pred_norm, mu_pred_abnorm, var_pred_abnorm

    def forward(
        self,
        y,
        mu_lstm_pred: Optional[np.ndarray] = None,
        var_lstm_pred: Optional[np.ndarray] = None,
    ):
        mu_pred_norm, var_pred_norm = self.norm_model.forward(
            mu_lstm_pred, var_lstm_pred
        )
        self.norm_model.mu_pred_norm = mu_pred_norm
        self.norm_model.var_pred_norm = var_pred_norm

        mu_pred_norm_to_ab, var_pred_norm_to_ab = self.norm_to_abnorm_model.forward(
            mu_lstm_pred, var_lstm_pred
        )
        self.norm_to_abnorm_model.mu_pred_norm = mu_pred_norm_to_ab
        self.norm_to_abnorm_model.var_pred_norm = var_pred_norm_to_ab

        mu_pred_abnorm, var_pred_abnorm = self.abnorm_model.forward(
            mu_lstm_pred, var_lstm_pred
        )
        self.abnorm_model.mu_pred_norm = mu_pred_abnorm
        self.abnorm_model.var_pred_norm = var_pred_abnorm

        mu_pred_ab_to_norm, var_pred_ab_to_norm = self.abnorm_to_norm_model.forward(
            mu_lstm_pred, var_lstm_pred
        )
        self.abnorm_to_norm_model.mu_pred_norm = mu_pred_ab_to_norm
        self.abnorm_to_norm_model.var_pred_norm = var_pred_ab_to_norm

        self.likelihood[0, 0] = np.exp(
            metric.log_likelihood(mu_pred_norm, y, var_pred_norm**0.5)
        )
        self.likelihood[0, 1] = np.exp(
            metric.log_likelihood(mu_pred_norm_to_ab, y, var_pred_norm_to_ab**0.5)
        )
        self.likelihood[1, 0] = np.exp(
            metric.log_likelihood(mu_pred_ab_to_norm, y, var_pred_ab_to_norm**0.5)
        )
        self.likelihood[1, 1] = np.exp(
            metric.log_likelihood(mu_pred_abnorm, y, var_pred_abnorm**0.5)
        )

        transition_prob = (
            self.likelihood * self.transition_prob_matrix * self.prob_model
        )
        self.transition_prob = transition_prob / np.sum(transition_prob)
        self.prob_model = np.sum(self.transition_prob, axis=0).reshape(-1, 1)

        mu_pred_norm, var_pred_norm, mu_pred_abnorm, var_pred_abnorm = self.collapse()
        mu_pred, var_pred = SKF.mixture(
            mu_pred_norm,
            var_pred_norm,
            self.prob_model[0],
            mu_pred_abnorm,
            var_pred_abnorm,
            self.prob_model[1],
        )
        return mu_pred, var_pred

    def backward(
        self,
        obs: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step in states-space model
        """

        mu_delta_norm, mu_delta_norm = self.norm_model.backward(
            obs, self.norm_model.mu_pred_norm, self.norm_model.var_pred_norm
        )
        self.norm_model.estimate_posterior_states(mu_delta_norm, mu_delta_norm)
        self.norm_model.mu_states = self.norm_model._mu_states_posterior
        self.norm_model.var_states = self.norm_model._var_states_posterior

        mu_delta_norm_to_ab, var_delta_norm_to_ab = self.norm_to_abnorm_model.backward(
            obs,
            self.norm_to_abnorm_model.mu_pred_norm,
            self.norm_to_abnorm_model.var_pred_norm,
        )
        self.norm_to_abnorm_model.estimate_posterior_states(
            mu_delta_norm_to_ab, var_delta_norm_to_ab
        )
        self.norm_to_abnorm_model.mu_states = (
            self.norm_to_abnorm_model._mu_states_posterior
        )
        self.norm_to_abnorm_model.var_states = (
            self.norm_to_abnorm_model._var_states_posterior
        )

        mu_delta_abnorm, var_delta_abnorm = self.abnorm_model.backward(
            obs,
            self.abnorm_model.mu_pred_norm,
            self.abnorm_model.var_pred_norm,
        )
        self.abnorm_model.estimate_posterior_states(mu_delta_abnorm, var_delta_abnorm)
        self.abnorm_model.mu_states = self.abnorm_model._mu_states_posterior
        self.abnorm_model.var_states = self.abnorm_model._var_states_posterior

        mu_delta_ab_to_norm, var_delta_ab_to_norm = self.abnorm_to_norm_model.backward(
            obs,
            self.abnorm_to_norm_model.mu_pred_norm,
            self.abnorm_to_norm_model.var_pred_norm,
        )
        self.abnorm_to_norm_model.estimate_posterior_states(
            mu_delta_ab_to_norm, var_delta_ab_to_norm
        )
        self.abnorm_to_norm_model.mu_states = (
            self.abnorm_to_norm_model._mu_states_posterior
        )
        self.abnorm_to_norm_model.var_states = (
            self.abnorm_to_norm_model._var_states_posterior
        )

        mu_pred_norm, var_pred_norm, mu_pred_abnorm, var_pred_abnorm = self.collapse()

    def filter(
        self,
        data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filtering
        """

        # Initialize sub-models for SKF
        self.norm_model = SKF.create_compatible_model(
            source=self.norm_model, target=self.abnorm_model
        )
        self.abnorm_model.set_states(
            self.norm_model.mu_states, self.norm_model.var_states
        )
        self.norm_to_abnorm_model = self.abnorm_model.duplicate_model()
        self.abnorm_to_norm_model = self.norm_model.duplicate_model()
        index_pad_states = self.norm_model.index_pad_states
        self.norm_to_abnorm_model.process_noise_matrix[
            index_pad_states, index_pad_states
        ] = (self.std_transition_error**2)

        #
        mu_obs_preds = []
        var_obs_preds = []
        mu_lstm_pred = None
        var_lstm_pred = None
        num_time_steps = len(data["y"])
        prob_abnorm = np.zeros((num_time_steps, 1), dtype=np.float32)

        for time_step, (x, y) in enumerate(zip(data["x"], data["y"])):
            if self.norm_model.lstm_net:
                mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                    self.norm_model.lstm_output_history, x
                )
                mu_lstm_pred, var_lstm_pred = self.norm_model.lstm_net.forward(
                    mu_x=mu_lstm_input, var_x=var_lstm_input
                )
            mu_obs_pred, var_obs_pred = self.forward(y, mu_lstm_pred, var_lstm_pred)
            prob_abnorm[time_step] = self.prob_model[1]
            self.backward(y)

            if self.norm_model.lstm_net:
                self.norm_model.update_lstm_output_history(mu_lstm_pred, var_lstm_pred)

        return prob_abnorm
