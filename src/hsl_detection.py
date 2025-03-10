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
        self.ar_component = self.base_model.components["autoregression"]
        self.drift_model = Model(
            LocalTrend(
                mu_states=[0, 0], 
                var_states=[baseline_process_error_std**2, baseline_process_error_std**2],
                std_error=baseline_process_error_std
                ),
            Autoregression(
                   std_error=self.ar_component.std_error, 
                   phi=self.ar_component.phi, 
                   mu_states=self.ar_component.mu_states, 
                   var_states=self.ar_component.var_states
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
                    self._collect_synthetic_samples()
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
    
    def _collect_synthetic_samples(self, num_samples: int = 10):
        # Collect samples from synthetic time series
        # Anomly feature range define
        ts_len = 52*6
        stationary_ar_std = self.ar_component.std_error/(1-self.ar_component.phi**2)**0.5
        anm_mag_range = [-stationary_ar_std/13, stationary_ar_std/13]
        anm_begin_range = [int(ts_len/3), ts_len]

        # # Generate synthetic time series
        covariate_col = self.data_processor.covariates_col
        time_covariate_info = {'initial_time_covariate': self.data_processor.validation_data[-1, covariate_col].item(),
                                'mu': self.data_processor.norm_const_mean[covariate_col], 
                                'std': self.data_processor.norm_const_std[covariate_col]}
        generated_ts, time_covariate, anm_mag_list, anm_begin_list = self.base_model.generate(num_time_series=num_samples, num_time_steps=ts_len, 
                                                                time_covariates=self.data_processor.time_covariates, 
                                                                time_covariate_info=time_covariate_info,
                                                                add_anomaly=True, anomaly_mag_range=anm_mag_range, anomaly_begin_range=anm_begin_range)
        # Plot generated time series
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 1)
        ax0 = plt.subplot(gs[0])
        for j in range(len(generated_ts)):
            ax0.plot(np.concatenate((self.data_processor.train_data_norm[:, self.data_processor.output_col].reshape(-1), 
                                        self.data_processor.validation_data_norm[:, self.data_processor.output_col].reshape(-1), 
                                        generated_ts[j])))
        ax0.axvline(x=len(self.data_processor.train_data_norm[:, self.data_processor.output_col].reshape(-1))+len(self.data_processor.validation_data_norm[:, self.data_processor.output_col].reshape(-1)), color='r', linestyle='--')
        ax0.set_title("Data generation")
        plt.show()

        # # Run the current model on the synthetic time series
        lstm_index = self.base_model.lstm_states_index
        for k in range(len(generated_ts)):
            base_model_copy = copy.deepcopy(self.base_model)
            base_model_copy.lstm_net = self.base_model.lstm_net
            base_model_copy.lstm_net.reset_lstm_states()
            drift_model_copy = copy.deepcopy(self.drift_model)

            mu_obs_preds, std_obs_preds = [], []
            mu_ar_preds, std_ar_preds = [], []
            base_model_copy.initialize_states_history()
            drift_model_copy.initialize_states_history()
            for i, (x, y) in enumerate(zip(time_covariate, generated_ts[k])):
                mu_obs_pred, var_obs_pred, _, _ = base_model_copy.forward(x)
                (
                    _, _,
                    mu_states_posterior,
                    var_states_posterior,
                ) = base_model_copy.backward(y)

                base_model_copy.update_lstm_output_history(
                    mu_states_posterior[lstm_index],
                    var_states_posterior[lstm_index, lstm_index],
                )

                base_model_copy.save_states_history()
                base_model_copy.set_states(mu_states_posterior, var_states_posterior)
                mu_obs_preds.append(mu_obs_pred)
                std_obs_preds.append(var_obs_pred**0.5)

                mu_ar_pred, var_ar_pred, mu_drift_states_prior, _ = drift_model_copy.forward()
                _, _, mu_drift_states_posterior, var_drift_states_posterior = drift_model_copy.backward(
                    obs=base_model_copy.mu_states_prior[self.AR_index], 
                    obs_var=base_model_copy.var_states_prior[self.AR_index, self.AR_index])
                drift_model_copy.save_states_history()
                drift_model_copy.set_states(mu_drift_states_posterior, var_drift_states_posterior)
                mu_ar_preds.append(mu_ar_pred)
                std_ar_preds.append(var_ar_pred**0.5)

                # Collect sample input
                # TODO
                # Label sample output: using anm_mag_list, anm_begin_list
                # TODO

            states_mu_prior = np.array(base_model_copy.states.mu_prior)
            states_var_prior = np.array(base_model_copy.states.var_prior)
            states_drift_mu_prior = np.array(drift_model_copy.states.mu_prior)
            states_drift_var_prior = np.array(drift_model_copy.states.var_prior)

            fig = plt.figure(figsize=(10, 9))
            gs = gridspec.GridSpec(7, 1)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[2])
            ax3 = plt.subplot(gs[3])
            ax4 = plt.subplot(gs[4])
            ax5 = plt.subplot(gs[5])
            ax6 = plt.subplot(gs[6])
            # print(base_model_copy.states.mu_prior)
            ax0.plot(states_mu_prior[:, 0].flatten(), label='local level')
            ax0.fill_between(np.arange(len(states_mu_prior[:, 0])),
                            states_mu_prior[:, 0].flatten() - states_var_prior[:, 0, 0]**0.5,
                            states_mu_prior[:, 0].flatten() + states_var_prior[:, 0, 0]**0.5,
                            alpha=0.5)
            ax0.plot(generated_ts[k])

            ax1.plot(states_mu_prior[:, 1].flatten(), label='local trend')
            ax1.fill_between(np.arange(len(states_mu_prior[:, 1])),
                            states_mu_prior[:, 1].flatten() - states_var_prior[:, 1, 1]**0.5,
                            states_mu_prior[:, 1].flatten() + states_var_prior[:, 1, 1]**0.5,
                            alpha=0.5)
            
            ax2.plot(states_mu_prior[:, 2].flatten(), label='lstm')
            ax2.fill_between(np.arange(len(states_mu_prior[:, 2])),
                            states_mu_prior[:, 2].flatten() - states_var_prior[:, 2, 2]**0.5,
                            states_mu_prior[:, 2].flatten() + states_var_prior[:, 2, 2]**0.5,
                            alpha=0.5)
            
            ax3.plot(states_mu_prior[:, 3].flatten(), label='autoregression')
            ax3.fill_between(np.arange(len(states_mu_prior[:, 3])),
                            states_mu_prior[:, 3].flatten() - states_var_prior[:, 3, 3]**0.5,
                            states_mu_prior[:, 3].flatten() + states_var_prior[:, 3, 3]**0.5,
                            alpha=0.5)
            ax4.plot(np.array(mu_ar_preds).flatten(), label='obs')
            ax4.fill_between(np.arange(len(mu_ar_preds)),
                            np.array(mu_ar_preds).flatten() - np.array(std_ar_preds).flatten(),
                            np.array(mu_ar_preds).flatten() + np.array(std_ar_preds).flatten(),
                            alpha=0.5)
            ax4.plot(states_drift_mu_prior[:, 0].flatten())
            ax4.fill_between(np.arange(len(states_drift_mu_prior[:, 0])),
                            states_drift_mu_prior[:, 0].flatten() - states_drift_var_prior[:, 0, 0]**0.5,
                            states_drift_mu_prior[:, 0].flatten() + states_drift_var_prior[:, 0, 0]**0.5,
                            alpha=0.5)
            ax4.set_ylabel('LLd')
            ax5.plot(states_drift_mu_prior[:, 1].flatten())
            ax5.fill_between(np.arange(len(states_drift_mu_prior[:, 1])),
                            states_drift_mu_prior[:, 1].flatten() - states_drift_var_prior[:, 1, 1]**0.5,
                            states_drift_mu_prior[:, 1].flatten() + states_drift_var_prior[:, 1, 1]**0.5,
                            alpha=0.5)
            ax5.set_ylabel('LTd')
            ax6.plot(states_drift_mu_prior[:, 2].flatten())
            ax6.fill_between(np.arange(len(states_drift_mu_prior[:, 2])),
                            states_drift_mu_prior[:, 2].flatten() - states_drift_var_prior[:, 2, 2]**0.5,
                            states_drift_mu_prior[:, 2].flatten() + states_drift_var_prior[:, 2, 2]**0.5,
                            alpha=0.5)
            ax6.set_ylabel('ARd')
            plt.show()