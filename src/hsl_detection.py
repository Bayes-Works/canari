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
from typing import Tuple, Dict, Optional
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
            drift_model_process_error_std: Optional[float] = 1e-4,
            ):
        self.base_model = base_model
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

        for i, (x, y) in enumerate(zip(data["x"], data["y"])):

            # Estimate likelihoods
            # This step should be done only when user wants to detect anomaly
            mu_obs_pred2, var_obs_pred2, mu_ar_pred2, var_ar_pred2, mu_lstm_pred, var_lstm_pred = self._estimate_likelihoods(obs=y, input_covariates=x)
            mu_obs_pred3, var_obs_pred3, mu_ar_pred3, var_ar_pred3, mu_lstm_pred, var_lstm_pred = self._estimate_likelihoods(obs=y, 
                                                                                                                             input_covariates=x, 
                                                                                                                             mu_lstm_pred=mu_lstm_pred,
                                                                                                                             var_lstm_pred=var_lstm_pred)

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
                    LTd_pdf = common.gaussian_pdf(mu = np.mean(LTd_buffer), std = np.std(LTd_buffer))

        return np.array(mu_obs_preds).flatten(), np.array(std_obs_preds).flatten(), np.array(mu_ar_preds).flatten(), np.array(std_ar_preds).flatten()

    def _estimate_likelihoods(
            self, 
            obs: float,
            input_covariates: Optional[np.ndarray] = None,
            mu_lstm_pred: Optional[np.ndarray] = None,
            var_lstm_pred: Optional[np.ndarray] = None,
            ):
        """
        Compute the likelihood of observation and hidden states given action
        """
        base_model_copy = copy.deepcopy(self.base_model)
        drift_model_copy = copy.deepcopy(self.drift_model)
        base_model_copy.lstm_net = self.base_model.lstm_net
        # TODO
        # if intervention == True:
        #     base_model_copy.mu_states = base_model_prior['mu']
        #     base_model_copy.var_states = base_model_prior['var']
        #     drift_model_copy.mu_states = drift_model_prior['mu']
        #     drift_model_copy.var_states = drift_model_prior['var']

        if mu_lstm_pred is not None and var_lstm_pred is not None:
            mu_obs_pred, var_obs_pred, _, var_states_prior = base_model_copy.forward(input_covariates = input_covariates, mu_lstm_pred=mu_lstm_pred, var_lstm_pred=var_lstm_pred)
        else:
            mu_obs_pred, var_obs_pred, _, var_states_prior = base_model_copy.forward(input_covariates = input_covariates)

        y_likelihood = likelihood(mu_obs_pred, np.sqrt(var_obs_pred), obs)

        # # TODO
        mu_ar_pred, var_ar_pred, mu_d_states_prior, _ = drift_model_copy.forward()
        # x_likelihood = dist_hidden_state(mu_d_states_prior[1].item())
        return mu_obs_pred, var_obs_pred, mu_ar_pred, var_ar_pred, base_model_copy.mu_lstm_pred, base_model_copy.var_lstm_pred