"""
Emsemble of model
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from canari.model import Model
from canari.data_struct import StatesHistory
import canari.common as common


class ModelEnsemble:
    """
    Ensemble of model for modelling dependencies.
    """

    def __init__(
        self,
        *model: Model,
    ):
        self.model = list(model)

    def forward(
        self,
        input_covariates: Optional[np.ndarray] = None,
    ):
        """ """

        mu_pred = []
        var_pred = []
        noise_var = 0
        # Covariate model
        mu_input_covariates = input_covariates.copy()
        var_input_covariates = np.zeros_like(input_covariates)
        for model in self.model:
            if model.model_type == "covariate":
                # get model input
                x = mu_input_covariates[model.input_col]
                mu_pred_covar, var_pred_covar, *_ = model.forward(x)

                # Obtain white noise variance
                for noise_type in ("white noise", "heteroscedastic noise"):
                    noise_index = model.get_states_index(noise_type)
                    if noise_index is not None:
                        noise_var = model.process_noise_matrix[noise_index, noise_index]
                        break

                # Replace in target model's input
                mu_input_covariates[model.output_col] = mu_pred_covar
                var_input_covariates[model.output_col] = var_pred_covar - noise_var

                # Replace lags TODO: model.mu_pred, not lstm.mu_pred
                if len(model.output_lag_col) > 0:
                    mu_output_lag = model.lstm_output_history.mu[
                        -len(model.output_lag_col) :
                    ]
                    var_output_lag = model.lstm_output_history.var[
                        -len(model.output_lag_col) :
                    ]
                    mu_input_covariates[model.output_lag_col] = mu_output_lag
                    var_input_covariates[model.output_lag_col] = var_output_lag

        # Target model
        for model in self.model:
            if model.model_type == "target":
                mu_pred, var_pred, *_ = model.forward(
                    mu_input_covariates, var_input_covariates
                )
        return mu_pred, var_pred
        # return mu_pred_covar, var_pred_covar

    def backward(
        self,
        obs: float,
        covariates: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ """

        for model in self.model:
            if model.model_type == "covariate":
                obs_temp = covariates[model.output_col].copy()
            else:
                obs_temp = obs.copy()

            (
                delta_mu_states,
                delta_var_states,
                *_,
            ) = model.backward(obs_temp)

            if model.lstm_net:
                lstm_index = model.get_states_index("lstm")
                delta_mu_lstm = np.array(
                    delta_mu_states[lstm_index]
                    / model.var_states_prior[lstm_index, lstm_index]
                )
                delta_var_lstm = np.array(
                    delta_var_states[lstm_index, lstm_index]
                    / model.var_states_prior[lstm_index, lstm_index] ** 2
                )
                model.lstm_net.update_param(
                    np.float32(delta_mu_lstm), np.float32(delta_var_lstm)
                )

    def forecast(
        self, data: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """ """

        mu_obs_preds = []
        std_obs_preds = []
        states = []
        for x in data["x"]:
            (
                mu_obs_pred,
                var_obs_pred,
            ) = self.forward(x)

            for model in self.model:
                if model.lstm_net:
                    lstm_index = model.get_states_index("lstm")
                    model.lstm_output_history.update(
                        model.mu_states_prior[lstm_index],
                        model.var_states_prior[lstm_index, lstm_index],
                    )

                model._set_posterior_states(
                    model.mu_states_prior, model.var_states_prior
                )
                model._save_states_history()
                model.set_states(model.mu_states_prior, model.var_states_prior)

            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)

        for model in self.model:
            if model.model_type == "target":
                states = model.states

        return (
            np.array(mu_obs_preds).flatten(),
            np.array(std_obs_preds).flatten(),
            states,
        )

    def filter(
        self,
        data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """ """

        mu_obs_preds = []
        std_obs_preds = []
        states = []

        for model in self.model:
            model.initialize_states_history()

        for x, y in zip(data["x"], data["y"]):
            (
                mu_obs_pred,
                var_obs_pred,
            ) = self.forward(x)

            self.backward(y, x)

            for model in self.model:
                if model.lstm_net:
                    lstm_index = model.get_states_index("lstm")
                    model.lstm_output_history.update(
                        model.mu_states_posterior[lstm_index],
                        model.var_states_posterior[lstm_index, lstm_index],
                    )
                model._save_states_history()
                model.set_states(model.mu_states_posterior, model.var_states_posterior)

            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)

        for model in self.model:
            if model.model_type == "target":
                states = model.states

        return (
            np.array(mu_obs_preds).flatten(),
            np.array(std_obs_preds).flatten(),
            states,
        )

    def smoother(
        self,
        matrix_inversion_tol: Optional[float] = 1e-12,
        tol_type: Optional[str] = "relative",  # relative of absolute
    ):
        """ """

        for model in self.model:
            model.smoother(matrix_inversion_tol=matrix_inversion_tol, tol_type=tol_type)

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
        white_noise_decay: Optional[bool] = True,
        white_noise_max_std: Optional[float] = 5,
        white_noise_decay_factor: Optional[float] = 0.9,
    ) -> Tuple[np.ndarray, np.ndarray, StatesHistory]:
        """
        Train LSTM
        """

        if white_noise_decay:
            for model in self.model:
                for noise_type in ("white noise", "heteroscedastic noise"):
                    if model.get_states_index(noise_type) is not None:
                        model.white_noise_decay(
                            model._current_epoch,
                            white_noise_max_std,
                            white_noise_decay_factor,
                        )
                        break
        self.filter(train_data)
        self.smoother()
        mu_validation_preds, std_validation_preds, states = self.forecast(
            validation_data
        )

        return (
            np.array(mu_validation_preds),
            np.array(std_validation_preds),
            states,
        )

    def set_memory(self, time_step: int):
        """ """
        if time_step == 0:
            for model in self.model:
                model.set_memory(states=model.states, time_step=time_step)

    def recal_covariates_col(self, covariates_col: np.ndarray):
        """
        Re-estimate model.input_col, model.output_col, and model.output_lag_col
        following data["x"].
        """

        for model in self.model:
            if model.model_type == "covariate":
                covariates_index = np.flatnonzero(covariates_col)
                if len(model.input_col) > 0:
                    model.input_col = [
                        np.where(covariates_index == i)[0][0] for i in model.input_col
                    ]
                model.output_col = [
                    np.where(covariates_index == i)[0][0] for i in model.output_col
                ]
                if len(model.output_lag_col) > 0:
                    model.output_lag_col = [
                        np.where(covariates_index == i)[0][0]
                        for i in model.output_lag_col
                    ]
