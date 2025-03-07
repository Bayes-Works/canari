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

class hsl_detection:
    """
    Anomaly detection based on hidden-states likelihood
    """

    def __init__(
            self, 
            base_model, 
            drift_model_process_error_std=1e-8,
            ):
        self.base_model = base_model

        self._create_drift_model(drift_model_process_error_std)
        pass

    def _create_drift_model(self, baseline_process_error_std):
        # Get AR component from the base model
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
        pass

    def filter(self, data):
        self.base_model.filter(data, train_lstm=False)
        pass

    def compute_likelihoods(self):
        """
        Compute the likelihood of observation and hidden states given action
        """
        pass