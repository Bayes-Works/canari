from src import (
    LocalLevel,
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    Periodic,
    Autoregression,
    WhiteNoise,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from examples import DataProcess
from pytagi import Normalizer as normalizer
from typing import Tuple, Dict, Optional, Callable
import src.common as common
import numpy as np
import copy
from src.common import likelihood

class hsl_detection:
    """
    Anomaly detection based on hidden-states likelihood
    """

    def __init__(
            self, 
            base_model: Model,
            data_processor: DataProcess,
            drift_model_process_error_std: Optional[float] = 1e-4,
            ):
        self.base_model = base_model
        self.data_processor = data_processor
        self._create_drift_model(drift_model_process_error_std)
        self.base_model.initialize_states_history()
        self.drift_model.initialize_states_history()
        self.AR_index = base_model.states_name.index("autoregression")
        pass

    def _create_drift_model(self, baseline_process_error_std):
        ar_component = self.base_model.components["autoregression"]
        self.drift_model = Model(
            LocalTrend(
                mu_states=[0, 0], 
                var_states=[baseline_process_error_std**2, baseline_process_error_std**2],
                std_error=baseline_process_error_std
                ),
            Autoregression(
                   std_error=ar_component.std_error, 
                   phi=ar_component.phi, 
                   mu_states=ar_component.mu_states, 
                   var_states=ar_component.var_states
                ),
        )

    def filter(
            self, 
            data,
            state_dist_estimate_window: Optional[np.ndarray] = None,
            ):
        data = common.set_default_input_covariates(data)
        lstm_index = self.base_model.lstm_states_index
        mu_obs_preds, std_obs_preds = [], []
        mu_ar_preds, std_ar_preds = [], []
        LTd_buffer = []
        LTd_pdf = None
        mu_lstm_pred, var_lstm_pred = None, None

        self.p_anm_all = []
        prior_na, prior_a = 0.998, 0.002

        for i, (x, y) in enumerate(zip(data["x"], data["y"])):
            # Estimate likelihoods
            # This step should be done only when user wants to detect anomaly
            if LTd_pdf is not None:
                # Estimate likelihood without intervention
                y_likelihood_na, x_likelihood_na, mu_lstm_pred, var_lstm_pred = self._estimate_likelihoods(obs=y, input_covariates=x, state_dist=LTd_pdf)
                # Estimate likelihood with intervention
                itv_base_model_prior, itv_drift_model_prior = self._intervene_current_priors()
                y_likelihood_a, x_likelihood_a, mu_lstm_pred, var_lstm_pred = self._estimate_likelihoods(obs=y, 
                                                                                                    input_covariates=x, 
                                                                                                    state_dist=LTd_pdf,
                                                                                                    mu_lstm_pred=mu_lstm_pred,
                                                                                                    var_lstm_pred=var_lstm_pred,
                                                                                                    base_model_prior=itv_base_model_prior,
                                                                                                    drift_model_prior=itv_drift_model_prior
                                                                                                    )
                p_yt_I_Yt1 = y_likelihood_na * x_likelihood_na * prior_na + y_likelihood_a * x_likelihood_a * prior_a
                # p_na_I_Yt = y_likelihood_na * x_likelihood_na * p_na_I_Yt1 / p_yt_I_Yt1
                p_a_I_Yt = (y_likelihood_a * x_likelihood_a * prior_a / p_yt_I_Yt1).item()
            else:
                y_likelihood_na, x_likelihood_na, y_likelihood_a, x_likelihood_a = None, None, None, None
                p_a_I_Yt = 0
            self.p_anm_all.append(p_a_I_Yt)

            # Base model filter process, same as in model.py
            mu_obs_pred, var_obs_pred, _, var_states_prior = self.base_model.forward(x,
                                                                                    mu_lstm_pred=mu_lstm_pred,
                                                                                    var_lstm_pred=var_lstm_pred,)
            (
                delta_mu_states,
                delta_var_states,
                mu_states_posterior,
                var_states_posterior,
            ) = self.base_model.backward(y)

            if self.base_model.lstm_net:
                self.base_model.update_lstm_output_history(
                    mu_states_posterior[lstm_index],
                    var_states_posterior[lstm_index, lstm_index],
                )

            self.base_model.save_states_history()
            self.base_model.set_states(mu_states_posterior, var_states_posterior)
            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)

            # Drift model filter process
            mu_ar_pred, var_ar_pred, mu_drift_states_prior, _ = self.drift_model.forward()
            _, _, mu_drift_states_posterior, var_drift_states_posterior = self.drift_model.backward(
                obs=self.base_model.mu_states_prior[self.AR_index], 
                obs_var=self.base_model.var_states_prior[self.AR_index, self.AR_index])
            self.drift_model.save_states_history()
            self.drift_model.set_states(mu_drift_states_posterior, var_drift_states_posterior)
            mu_ar_preds.append(mu_ar_pred)
            std_ar_preds.append(var_ar_pred**0.5)

            if state_dist_estimate_window is not None:
                if i >= state_dist_estimate_window[0] and i < state_dist_estimate_window[1]:
                    LTd_buffer.append(mu_drift_states_prior[1].item())
                if i == state_dist_estimate_window[1]:
                    self.mu_LTd = np.mean(LTd_buffer)
                    LTd_pdf = common.gaussian_pdf(mu = self.mu_LTd, std = np.std(LTd_buffer))
                    # Collect samples from synthetic time series
                    # TODO
                    validation_data_x_unnorm = normalizer.unstandardize(
                                                    self.data_processor.validation_data_norm[:, self.data_processor.covariates_col],
                                                    self.data_processor.norm_const_mean[1],
                                                    data_processor.norm_const_std[1],
                                                )
                    time_covariate_info = {'initial_time_covariate': validation_data_x_unnorm[-1].item(),
                                            'mu': data_processor.norm_const_mean[1], 
                                            'std': data_processor.norm_const_std[1]}
                    generated_ts = self.base_model.generate(num_time_series=1, num_time_steps=52*6, time_covariates=self.data_processor.time_covariates, time_covariate_info=time_covariate_info)
                    # Train neural network to learn intervention
                    # TODO

        return np.array(mu_obs_preds).flatten(), np.array(std_obs_preds).flatten(), np.array(mu_ar_preds).flatten(), np.array(std_ar_preds).flatten()

    def _estimate_likelihoods(
            self, 
            obs: float,
            state_dist: Optional[Callable] = None,
            input_covariates: Optional[np.ndarray] = None,
            mu_lstm_pred: Optional[np.ndarray] = None,
            var_lstm_pred: Optional[np.ndarray] = None,
            base_model_prior: Optional[Dict] = None,
            drift_model_prior: Optional[Dict] = None,
            ):
        """
        Compute the likelihood of observation and hidden states given action
        """
        base_model_copy = copy.deepcopy(self.base_model)
        drift_model_copy = copy.deepcopy(self.drift_model)
        base_model_copy.lstm_net = self.base_model.lstm_net

        if base_model_prior is not None and drift_model_prior is not None:
            base_model_copy.mu_states = base_model_prior['mu']
            base_model_copy.var_states = base_model_prior['var']
            drift_model_copy.mu_states = drift_model_prior['mu']
            drift_model_copy.var_states = drift_model_prior['var']

        if mu_lstm_pred is not None and var_lstm_pred is not None:
            mu_obs_pred, var_obs_pred, _, var_states_prior = base_model_copy.forward(input_covariates = input_covariates, mu_lstm_pred=mu_lstm_pred, var_lstm_pred=var_lstm_pred)
        else:
            mu_obs_pred, var_obs_pred, _, var_states_prior = base_model_copy.forward(input_covariates = input_covariates)

        y_likelihood = likelihood(mu_obs_pred, np.sqrt(var_obs_pred), obs)

        mu_ar_pred, var_ar_pred, mu_d_states_prior, _ = drift_model_copy.forward()
        x_likelihood = state_dist(mu_d_states_prior[1].item())
        return y_likelihood.item(), x_likelihood, base_model_copy.mu_lstm_pred, base_model_copy.var_lstm_pred
    
    def _intervene_current_priors(self):
        base_model_prior = {
            'mu': copy.deepcopy(self.base_model.mu_states),
            'var': copy.deepcopy(self.base_model.var_states)
        }
        drift_model_prior = {
            'mu': copy.deepcopy(self.drift_model.mu_states),
            'var': copy.deepcopy(self.drift_model.var_states)
        }

        LL_index = self.base_model.states_name.index("local level")
        LT_index = self.base_model.states_name.index("local trend")
        AR_index = self.base_model.states_name.index("autoregression")
        base_model_prior['mu'][LL_index] += drift_model_prior['mu'][0]
        base_model_prior['mu'][LT_index] += drift_model_prior['mu'][1]
        base_model_prior['mu'][AR_index] = drift_model_prior['mu'][2]
        base_model_prior['var'][LL_index, LL_index] += drift_model_prior['var'][0, 0]
        base_model_prior['var'][LT_index, LT_index] += drift_model_prior['var'][1, 1]
        base_model_prior['var'][AR_index, AR_index] = drift_model_prior['var'][2, 2]
        drift_model_prior['mu'][0] = 0
        drift_model_prior['mu'][1] = self.mu_LTd
        return base_model_prior, drift_model_prior