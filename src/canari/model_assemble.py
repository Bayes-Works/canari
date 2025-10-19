"""
Assemble of multiple :class:~canari.model.Model instances. This class is
used when the target model depends on a covariate that is itself modeled
by another :class:`~canari.model.Model`.
"""

import copy
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from canari.model import Model
import canari.common as common


class ModelAssemble:
    """
    Assemble of :class:`~canari.model.Model` instances.

    Args:
        target_model (Model): the target (dependent) model.
        covariate_model(List[Model]): the independent model used as input for the target one.
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
        posterior_covariate: Optional[bool] = True,
        update_param_covar_model: Optional[bool] = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make a one-step-ahead prediction using the prediction step of the Kalman filter.
        If no `input_covariates` for LSTM, use an empty `np.ndarray`.
        Recall :meth:`~canari.model.Model.forward` for each model.

        This function is used at the one-time-step level.

        Args:
            input_covariates (Optional[np.ndarray]): Input covariates for LSTM at time `t`.
            posterior_covariate (Optional[bool]): Estimate the posterior for covariate model or not.
                        If True, the posteriors for covariates are used as input for the target model.
                        Otherwise, priors are used. Defaults to True.
            update_param_covar_model (Optional[bool]): Update the network parameters for the covariate models
                                                        or not. Defaults to True.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                A tuple containing:

                - :attr:`mu_pred` (np.ndarray):
                    The predictive mean of the observation at `t+1`.
                - :attr:`var_pred` (np.ndarray):
                    The predictive variance of the observation at `t+1`.
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
            var_pred_covar = var_pred_covar - model.sched_sigma_v**2

            if posterior_covariate:
                # Backward for covar_models
                y_covar = mu_input_covariates[model.output_col].copy()
                delta_mu, delta_var, mu_pos, var_pos = model.backward(y_covar)
                mu_pred_covar, var_pred_covar = common.calc_observation(
                    mu_pos, var_pos, model.observation_matrix
                )
                if update_param_covar_model and model.lstm_net:
                    model.update_lstm_param(delta_mu, delta_var)

            # save output history
            model.output_history.save_output_history(mu_pred_covar, var_pred_covar)

            # Replace in target model's input
            mu_input_covariates[model.output_col] = mu_pred_covar
            var_input_covariates[model.output_col] = var_pred_covar

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
        obs: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Update step in the Kalman filter for one time step.
        """

        (
            delta_mu_states,
            delta_var_states,
            *_,
        ) = self.target_model.backward(obs)
        if self.target_model.lstm_net:
            self.target_model.update_lstm_param(delta_mu_states, delta_var_states)

    def forecast(
        self,
        data: Dict[str, np.ndarray],
        posterior_covariate: Optional[bool] = False,
        update_param_covar_model: Optional[bool] = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform multi-step-ahead forecast over an entire dataset by recursively making
        one-step-ahead predictions, i.e., reapeatly apply the
        Kalman prediction step over multiple time steps.

        Args:
            data (Dict[str, np.ndarray]): A dictionary containing key 'x' as input covariates,
                                            if exists 'y' (real observations) will not be used.
            posterior_covariate (Optional[bool]): Estimate the posterior for covariate model or not.
                        If True, the posteriors for covariates are used as input for the target model.
                        Otherwise, priors are used. Defaults to False.
            update_param_covar_model (Optional[bool]): Update the network parameters for the covariate models
                                                        or not. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                A tuple containing:

                - **mu_obs_preds** (np.ndarray):
                    The means for forecasts.
                - **std_obs_preds** (np.ndarray):
                    The standard deviations for forecasts.
        """

        mu_obs_preds = []
        std_obs_preds = []

        for x in data["x"]:
            (
                mu_obs_pred,
                var_obs_pred,
            ) = self.forward(x, posterior_covariate, update_param_covar_model)

            for model in [self.target_model] + self.covariate_model:
                if model.lstm_net:
                    model.update_lstm_output_history(
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

        Args:
            data (Dict[str, np.ndarray]): Includes 'x' and 'y'.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                A tuple containing:

                - **mu_obs_preds** (np.ndarray):
                    The means for forecasts.
                - **std_obs_preds** (np.ndarray):
                    The standard deviations for forecasts.
        """

        mu_obs_preds = []
        std_obs_preds = []

        for model in [self.target_model] + self.covariate_model:
            model.initialize_states_history()
            model.output_history.initialize()

        for index, (x, y) in enumerate(zip(data["x"], data["y"])):
            (
                mu_obs_pred,
                var_obs_pred,
            ) = self.forward(x)

            self.backward(y)

            for model in [self.target_model] + self.covariate_model:
                if model.lstm_net:
                    model.update_lstm_states_history(
                        index, last_step=len(data["y"]) - 1
                    )
                    model.update_lstm_output_history(
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

        Args:
            matrix_inversion_tol (float): Numerical stability threshold for matrix
                                            pseudoinversion (pinv). Defaults to 1E-12.
            tol_type (Optional[str]): Tolerance type, "relative" or "absolute".
                                        Defaults to "relative".
        """

        for model in [self.target_model] + self.covariate_model:
            model.smoother(matrix_inversion_tol=matrix_inversion_tol, tol_type=tol_type)

    def lstm_train(
        self,
        train_data: Dict[str, np.ndarray],
        validation_data: Dict[str, np.ndarray],
        use_val_posterior_covariate: Optional[bool] = False,
        update_param_covar_model: Optional[bool] = False,
        white_noise_decay: Optional[bool] = True,
        white_noise_max_std: Optional[float] = 5,
        white_noise_decay_factor: Optional[float] = 0.9,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the :class:`~canari.component.lstm_component.LstmNetwork` component for each model
        on the provided training set, then evaluate on the validation set.
        Optionally apply exponential decay on the white noise standard deviation over epochs.

        Args:
            train_data (Dict[str, np.ndarray]):
                Dictionary with keys `'x'` and `'y'` for training inputs and targets.
            validation_data (Dict[str, np.ndarray]):
                Dictionary with keys `'x'` and `'y'` for validation inputs and targets.
            posterior_covariate (Optional[bool]): Estimate the posterior for covariate model or not.
                        If True, the posteriors for covariates are used as input for the target model.
                        Otherwise, priors are used. Defaults to False.
            update_param_covar_model (Optional[bool]): Update the network parameters for the covariate models
                                                        or not. Defaults to False.
            white_noise_decay (bool, optional):
                If True, apply an exponential decay on the white noise standard deviation
                over epochs, if a white noise component exists. Defaults to True.
            white_noise_max_std (float, optional):
                Upper bound on the white-noise standard deviation when decaying.
                Defaults to 5.
            white_noise_decay_factor (float, optional):
                Multiplicative decay factor applied to the white‐noise standard
                deviation each epoch. Defaults to 0.9.

        Returns:
            Tuple[np.ndarray, np.ndarray, StatesHistory]:
                A tuple containing:

                - **mu_validation_preds** (np.ndarray):
                    The means for multi-step-ahead predictions for the validation set.
                - **std_validation_preds** (np.ndarray):
                    The standard deviations for multi-step-ahead predictions for the validation set.
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

        if self.target_model.lstm_net.smooth:
            self.target_model.pretraining_filter(train_data)

        for model in self.covariate_model:
            if model.lstm_net.smooth:
                pretrain_data = copy.copy(train_data)
                # pretrain_data["cov_names"] = [
                #     pretrain_data["cov_names"][model.input_col[0]]
                # ]
                selected_names = [
                    pretrain_data["cov_names"][i] for i in model.input_col
                ]
                pretrain_data["cov_names"] = selected_names
                model.pretraining_filter(pretrain_data)

        self.filter(train_data)
        mu_validation_preds, std_validation_preds = self.forecast(
            validation_data, use_val_posterior_covariate, update_param_covar_model
        )
        self.smoother()

        for model in [self.target_model] + self.covariate_model:
            model.set_memory(time_step=0)
            model._current_epoch += 1

        return (
            np.array(mu_validation_preds),
            np.array(std_validation_preds),
        )

    def set_memory(self, time_step: int):
        """
        Set memory for each model to a specific time step.
        Recall :meth:`~canari.model.Model.set_memory` for each model.

        Args:
            time_step (int): Time step to set the memory.
        """

        for model in [self.target_model] + self.covariate_model:
            model.set_memory(time_step=time_step)

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
