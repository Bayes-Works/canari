"""
Emsemble of model
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from canari.model import Model


class ModelAssemble:
    """
    Assemble of models for modelling dependencies.
    """

    def __init__(
        self,
        target_model: Model,
        covariate_model: Union[Model, List[Model]],
    ):
        self.target_model = target_model
        self.covariate_model = (
            covariate_model if isinstance(covariate_model, list) else [covariate_model]
        )
        self._recal_covar_col = True

    def forward(
        self,
        input_covariates: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make a one-step-ahead prediction using the prediction step of the Kalman filter.
        """

        mu_pred = []
        var_pred = []
        mu_input_covariates = input_covariates.copy()
        var_input_covariates = np.zeros_like(input_covariates)

        # Covariate model
        for model in self.covariate_model:
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
                mu_output_lag = model.output_history.mu[-len(model.output_lag_col) :]
                mu_output_lag = np.concatenate(
                    (
                        np.flip(mu_output_lag),
                        np.zeros(len(model.output_lag_col) - len(mu_output_lag)),
                    )
                )
                var_output_lag = model.output_history.var[-len(model.output_lag_col) :]
                var_output_lag = np.concatenate(
                    (
                        np.flip(var_output_lag),
                        np.zeros(len(model.output_lag_col) - len(var_output_lag)),
                    )
                )
                mu_input_covariates[model.output_lag_col] = mu_output_lag
                var_input_covariates[model.output_lag_col] = var_output_lag

        # Target model
        mu_pred, var_pred, *_ = self.target_model.forward(
            mu_input_covariates, var_input_covariates
        )
        self.target_model.output_history.save_output_history(mu_pred, var_pred)

        return mu_pred, var_pred

    def backward(
        self,
        covariates: np.ndarray,
        obs: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Update step in the Kalman filter for one time step.
        """

        # Covariate models
        for model in self.covariate_model:
            obs_temp = covariates[model.output_col].copy()
            (
                delta_mu_states_covar,
                delta_var_states_covar,
                *_,
            ) = model.backward(obs_temp)
            if model.lstm_net:
                model.update_lstm_param(delta_mu_states_covar, delta_var_states_covar)

        # Target model
        (
            delta_mu_states,
            delta_var_states,
            *_,
        ) = self.target_model.backward(obs)
        if self.target_model.lstm_net:
            self.target_model.update_lstm_param(delta_mu_states, delta_var_states)

    def forecast(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform multi-step-ahead forecast over an entire dataset by recursively making
        one-step-ahead predictions, i.e., reapeatly apply the
        Kalman prediction step over multiple time steps.
        """

        mu_obs_preds = []
        std_obs_preds = []

        for x in data["x"]:
            (
                mu_obs_pred,
                var_obs_pred,
            ) = self.forward(x)

            for model in [self.target_model] + self.covariate_model:
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
        """
        Run the Kalman filter over an entire dataset, i.e., repeatly apply the Kalman prediction and
        update steps over multiple time steps.
        """

        mu_obs_preds = []
        std_obs_preds = []

        for model in [self.target_model] + self.covariate_model:
            model.initialize_states_history()
            model.output_history.initialize()

        for x, y in zip(data["x"], data["y"]):
            (
                mu_obs_pred,
                var_obs_pred,
            ) = self.forward(x)

            self.backward(x, y)

            for model in [self.target_model] + self.covariate_model:
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
        """
        Run the Kalman smoother over an entire time series data, i.e., repeatly apply the
        RTS smoothing equation over multiple time steps.
        """

        for model in [self.target_model] + self.covariate_model:
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
        Train both the target model and covariate models on the provided
        training set, then evaluate on the validation set.
        """

        self._recal_covariates_col(
            train_data["covariates_col"], train_data["data_col_names"]
        )

        if white_noise_decay:
            for model in [self.target_model] + self.covariate_model:
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
        """
        Set memory
        """
        if time_step == 0:
            for model in [self.target_model] + self.covariate_model:
                model.set_memory(states=model.states, time_step=time_step)

    def _recal_covariates_col(self, covariates_col: np.ndarray, data_col_names: list):
        """
        Re-estimate model.input_col, model.output_col, and model.output_lag_col
        following data["x"].
        """

        if self._recal_covar_col:
            for model in self.covariate_model:
                output_name = data_col_names[model.output_col[0]]
                model.output_lag_col = [
                    i
                    for i, col in enumerate(data_col_names)
                    if f"{output_name}_lag" in col
                ]
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

            # set to False, do recal_covariates_col() only once
            self._recal_covar_col = False
