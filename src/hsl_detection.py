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

class hsl_detection:
    """
    Anomaly detection based on hidden-states likelihood
    """

    def __init__(
            self, 
            base_model: Model,
            drift_model_process_error_std: Optional[float] = 0.0,
            ):
        self.base_model = base_model
        self._create_drift_model(drift_model_process_error_std)
        self.base_model.initialize_states_history()
        self.drift_model.initialize_states_history()
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
            run_drift_model=False
            ):
        # Base model filter process, same as in model.py
        data = common.set_default_input_covariates(data)
        lstm_index = self.base_model.lstm_states_index
        mu_obs_preds = []
        std_obs_preds = []

        for x, y in zip(data["x"], data["y"]):
            mu_obs_pred, var_obs_pred, _, var_states_prior = self.base_model.forward(x)
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

        # TODO
        if run_drift_model:
            self.drift_model.initialize_states_history()

        return np.array(mu_obs_preds).flatten(), np.array(std_obs_preds).flatten()

    def estimate_likelihoods(self):
        """
        Compute the likelihood of observation and hidden states given action
        """
        pass