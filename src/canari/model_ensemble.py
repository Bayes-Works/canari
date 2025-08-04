"""
Emsemble of model
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from canari.model import Model
from canari.data_struct import StatesHistory, OutputHistory


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
        """
        Forward
        """

        mu_pred = []
        var_pred = []
        # Covariate model
        mu_input_covariates = input_covariates.copy()
        var_input_covariates = np.zeros_like(input_covariates)
        for model in self.model:
            if model.model_type == "covariate":
                # get model input
                x = mu_input_covariates[model.input_col]
                mu_pred_covar, var_pred_covar, *_ = model.forward(x)

                # save output history
                var_pred_covar_wo_noise = var_pred_covar - model.sched_sigma_v**2
                model.output_history.save_output_history(
                    mu_pred_covar, var_pred_covar_wo_noise
                )

                # Replace in target model's input
                mu_input_covariates[model.output_col] = mu_pred_covar
                var_input_covariates[model.output_col] = var_pred_covar_wo_noise

                # Replace lags
                if len(model.output_lag_col) > 0:
                    mu_output_lag = model.output_history.mu[
                        -len(model.output_lag_col) :
                    ]
                    mu_output_lag = np.concatenate(
                        (
                            np.flip(mu_output_lag),
                            np.zeros(len(model.output_lag_col) - len(mu_output_lag)),
                        )
                    )
                    var_output_lag = model.output_history.var[
                        -len(model.output_lag_col) :
                    ]
                    var_output_lag = np.concatenate(
                        (
                            np.flip(var_output_lag),
                            np.zeros(len(model.output_lag_col) - len(var_output_lag)),
                        )
                    )
                    mu_input_covariates[model.output_lag_col] = mu_output_lag
                    var_input_covariates[model.output_lag_col] = var_output_lag

        # Target model
        for model in self.model:
            if model.model_type == "target":
                mu_pred, var_pred, *_ = model.forward(
                    mu_input_covariates, var_input_covariates
                )
                model.output_history.save_output_history(mu_pred, var_pred)

        return mu_pred, var_pred

    def backward(
        self,
        covariates: np.ndarray,
        obs: float,
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
                model.update_lstm_param(delta_mu_states, delta_var_states)

    def forecast(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """ """

        mu_obs_preds = []
        std_obs_preds = []
        for x in data["x"]:
            (
                mu_obs_pred,
                var_obs_pred,
            ) = self.forward(x)

            for model in self.model:
                if model.lstm_net:
                    model.update_lstm_history(
                        model.mu_states_prior, model.var_states_prior
                    )

                model._set_posterior_states(
                    model.mu_states_prior, model.var_states_prior
                )
                model.save_states_history()
                model.set_states(model.mu_states_prior, model.var_states_prior)

            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)

        return (
            np.array(mu_obs_preds).flatten(),
            np.array(std_obs_preds).flatten(),
        )

    def filter(
        self,
        data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ """

        mu_obs_preds = []
        std_obs_preds = []

        for model in self.model:
            model.initialize_states_history()
            model.output_history.initialize()

        for x, y in zip(data["x"], data["y"]):
            (
                mu_obs_pred,
                var_obs_pred,
            ) = self.forward(x)

            self.backward(x, y)

            for model in self.model:
                if model.lstm_net:
                    model.update_lstm_history(
                        model.mu_states_posterior, model.var_states_posterior
                    )
                model.save_states_history()
                model.set_states(model.mu_states_posterior, model.var_states_posterior)

            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)

        return (
            np.array(mu_obs_preds).flatten(),
            np.array(std_obs_preds).flatten(),
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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        mu_validation_preds, std_validation_preds = self.forecast(validation_data)

        return (
            np.array(mu_validation_preds),
            np.array(std_validation_preds),
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
