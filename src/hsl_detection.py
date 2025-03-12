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
import pandas as pd
from tqdm import tqdm
from pytagi.nn import Linear, OutputUpdater, Sequential, ReLU, EvenExp

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
        self.LTd_buffer = []
        self.p_anm_all = []
        self.prior_na, self.prior_a = 0.998, 0.002
        self.detection_threshold = 0.5
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

    def filter(self, data, buffer_LTd: Optional[bool] = False):
        data = common.set_default_input_covariates(data)
        lstm_index = self.base_model.lstm_states_index
        mu_obs_preds, std_obs_preds = [], []
        mu_ar_preds, std_ar_preds = [], []
        mu_lstm_pred, var_lstm_pred = None, None

        for i, (x, y) in enumerate(zip(data["x"], data["y"])):
            # Base model filter process, same as in model.py
            mu_obs_pred, var_obs_pred, _, _ = self.base_model.forward(x,
                                                                      mu_lstm_pred=mu_lstm_pred,
                                                                      var_lstm_pred=var_lstm_pred,)
            (
                _, _,
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

            if buffer_LTd:
                self.LTd_buffer.append(mu_drift_states_prior[1].item())
            
            # Dummy values for p_anm when it is not in detect mode
            self.p_anm_all.append(0)

        return np.array(mu_obs_preds).flatten(), np.array(std_obs_preds).flatten(), np.array(mu_ar_preds).flatten(), np.array(std_ar_preds).flatten()
    
    def estimate_LTd_dist(self):
        self.mu_LTd = np.mean(self.LTd_buffer)
        self.LTd_pdf = common.gaussian_pdf(mu = self.mu_LTd, std = np.std(self.LTd_buffer))
        print('LTd_mean:', self.mu_LTd)
        print('LTd_std:', np.std(self.LTd_buffer))

    def detect(
            self, 
            data,
            ):
        data = common.set_default_input_covariates(data)
        lstm_index = self.base_model.lstm_states_index
        mu_obs_preds, std_obs_preds = [], []
        mu_ar_preds, std_ar_preds = [], []
        mu_lstm_pred, var_lstm_pred = None, None

        for i, (x, y) in enumerate(zip(data["x"], data["y"])):
            # Estimate likelihoods
            # Estimate likelihood without intervention
            y_likelihood_na, x_likelihood_na, mu_lstm_pred, var_lstm_pred = self._estimate_likelihoods(base_model=self.base_model, drift_model=self.drift_model,
                                                                                                        obs=y, input_covariates=x, state_dist=self.LTd_pdf)
            # Estimate likelihood with intervention
            itv_base_model_prior, itv_drift_model_prior = self._intervene_current_priors(base_model=self.base_model, drift_model=self.drift_model,)
            y_likelihood_a, x_likelihood_a, mu_lstm_pred, var_lstm_pred = self._estimate_likelihoods(
                                                                                                base_model=self.base_model, drift_model=self.drift_model,
                                                                                                obs=y, input_covariates=x, state_dist=self.LTd_pdf,
                                                                                                mu_lstm_pred=mu_lstm_pred, var_lstm_pred=var_lstm_pred,
                                                                                                base_model_prior=itv_base_model_prior, drift_model_prior=itv_drift_model_prior
                                                                                                )
            p_yt_I_Yt1 = y_likelihood_na * x_likelihood_na * self.prior_na + y_likelihood_a * x_likelihood_a * self.prior_a
            # p_na_I_Yt = y_likelihood_na * x_likelihood_na * p_na_I_Yt1 / p_yt_I_Yt1
            p_a_I_Yt = (y_likelihood_a * x_likelihood_a * self.prior_a / p_yt_I_Yt1).item()
            self.p_anm_all.append(p_a_I_Yt)

            if p_a_I_Yt > self.detection_threshold:
                # Get LTd history
                LTd_mu_prior = np.array(self.drift_model.states.mu_prior)[:, 1].flatten()
                LTd_history = self._hidden_states_collector(i - 1, LTd_mu_prior)
                LTd_history = np.array(LTd_history.tolist(), dtype=np.float32)
                LTd_history = (LTd_history - self.mean_train) / self.std_train
                LTd_history = np.repeat(LTd_history[np.newaxis, :], self.batch_size, axis=0)
                # LTd_history = np.tile(LTd_history, (10, 1))

                output_pred_mu, output_pred_var = self.model.net(LTd_history)
                output_pred_mu = output_pred_mu.reshape(self.batch_size, len(self.target_list)*2)
                output_pred_var = output_pred_var.reshape(self.batch_size, len(self.target_list)*2)
                itv_pred_mu = output_pred_mu[0, [0, 2, 4]]
                itv_pred_var = output_pred_mu[0, [1, 3, 5]]
                itv_pred_mu_denorm = itv_pred_mu * self.std_target + self.mean_target
                itv_pred_var_denorm = itv_pred_var * self.std_target ** 2

                # Apply intervention on base_model hidden states
                LL_index = self.base_model.states_name.index("local level")
                LT_index = self.base_model.states_name.index("local trend")
                AR_index = self.base_model.states_name.index("autoregression")
                self.base_model.mu_states[LL_index] += itv_pred_mu_denorm[1]
                self.base_model.mu_states[LT_index] += itv_pred_mu_denorm[0]
                self.base_model.mu_states[AR_index] -= itv_pred_mu_denorm[1]
                self.base_model.var_states[LL_index, LL_index] += itv_pred_var_denorm[1]
                self.base_model.var_states[LT_index, LT_index] += itv_pred_var_denorm[0]

            # Base model filter process, same as in model.py
            mu_obs_pred, var_obs_pred, _, _ = self.base_model.forward(x,
                                                                      mu_lstm_pred=mu_lstm_pred,
                                                                      var_lstm_pred=var_lstm_pred,)
            (
                _, _,
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

        return np.array(mu_obs_preds).flatten(), np.array(std_obs_preds).flatten(), np.array(mu_ar_preds).flatten(), np.array(std_ar_preds).flatten()

    def _estimate_likelihoods(
            self, 
            base_model: Model,
            drift_model: Model,
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
        base_model_copy = copy.deepcopy(base_model)
        drift_model_copy = copy.deepcopy(drift_model)
        base_model_copy.lstm_net = base_model.lstm_net

        if base_model_prior is not None and drift_model_prior is not None:
            base_model_copy.mu_states = base_model_prior['mu']
            base_model_copy.var_states = base_model_prior['var']
            drift_model_copy.mu_states = drift_model_prior['mu']
            drift_model_copy.var_states = drift_model_prior['var']

        if mu_lstm_pred is not None and var_lstm_pred is not None:
            mu_obs_pred, var_obs_pred, _, _ = base_model_copy.forward(input_covariates = input_covariates, mu_lstm_pred=mu_lstm_pred, var_lstm_pred=var_lstm_pred)
        else:
            mu_obs_pred, var_obs_pred, _, _ = base_model_copy.forward(input_covariates = input_covariates)

        y_likelihood = likelihood(mu_obs_pred, np.sqrt(var_obs_pred), obs)

        _, _, mu_d_states_prior, _ = drift_model_copy.forward()
        x_likelihood = state_dist(mu_d_states_prior[1].item())
        return y_likelihood.item(), x_likelihood, base_model_copy.mu_lstm_pred, base_model_copy.var_lstm_pred
    
    def _intervene_current_priors(self, base_model, drift_model):
        base_model_prior = {
            'mu': copy.deepcopy(base_model.mu_states),
            'var': copy.deepcopy(base_model.var_states)
        }
        drift_model_prior = {
            'mu': copy.deepcopy(drift_model.mu_states),
            'var': copy.deepcopy(drift_model.var_states)
        }

        LL_index = base_model.states_name.index("local level")
        LT_index = base_model.states_name.index("local trend")
        AR_index = base_model.states_name.index("autoregression")
        base_model_prior['mu'][LL_index] += drift_model_prior['mu'][0]
        base_model_prior['mu'][LT_index] += drift_model_prior['mu'][1]
        base_model_prior['mu'][AR_index] = drift_model_prior['mu'][2]
        base_model_prior['var'][LL_index, LL_index] += drift_model_prior['var'][0, 0]
        base_model_prior['var'][LT_index, LT_index] += drift_model_prior['var'][1, 1]
        base_model_prior['var'][AR_index, AR_index] = drift_model_prior['var'][2, 2]
        drift_model_prior['mu'][0] = 0
        drift_model_prior['mu'][1] = self.mu_LTd
        return base_model_prior, drift_model_prior
    
    def collect_synthetic_samples(self, num_time_series: int = 10, save_to_path: Optional[str] = 'data/hsl_tsad_training_samples/hsl_tsad_train_samples.csv'):
        # Collect samples from synthetic time series
        samples = {'LTd_history': [], 'itv_LT': [], 'itv_LL': [], 'anm_develop_time': []}

        # Anomly feature range define
        ts_len = 52*6
        stationary_ar_std = self.ar_component.std_error/(1-self.ar_component.phi**2)**0.5
        anm_mag_range = [-stationary_ar_std/8, stationary_ar_std/8]
        anm_begin_range = [int(ts_len/3), int(ts_len/3*2)]

        # # Generate synthetic time series
        covariate_col = self.data_processor.covariates_col
        time_covariate_info = {'initial_time_covariate': self.data_processor.validation_data[-1, covariate_col].item(),
                                'mu': self.data_processor.norm_const_mean[covariate_col], 
                                'std': self.data_processor.norm_const_std[covariate_col]}
        generated_ts, time_covariate, anm_mag_list, anm_begin_list = self.base_model.generate(num_time_series=num_time_series, num_time_steps=ts_len, 
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
        for k in tqdm(range(len(generated_ts))):
            base_model_copy = copy.deepcopy(self.base_model)
            base_model_copy.lstm_net = self.base_model.lstm_net
            base_model_copy.lstm_net.reset_lstm_states()
            drift_model_copy = copy.deepcopy(self.drift_model)

            mu_obs_preds, std_obs_preds = [], []
            mu_ar_preds, std_ar_preds = [], []
            p_anm_one_syn_ts = []
            y_likelihood_a_one_ts, y_likelihood_na_one_ts = [], []
            x_likelihood_a_one_ts, x_likelihood_na_one_ts = [], []
            base_model_copy.initialize_states_history()
            drift_model_copy.initialize_states_history()
            anomaly_detected = False
            for i, (x, y) in enumerate(zip(time_covariate, generated_ts[k])):
                # Estimate likelihood without intervention
                y_likelihood_na, x_likelihood_na, mu_lstm_pred, var_lstm_pred = self._estimate_likelihoods(base_model=base_model_copy, drift_model=drift_model_copy,
                                                                                                           obs=y, input_covariates=x, state_dist=self.LTd_pdf)
                # Estimate likelihood with intervention
                itv_base_model_prior, itv_drift_model_prior = self._intervene_current_priors(base_model=base_model_copy, drift_model=drift_model_copy,)
                y_likelihood_a, x_likelihood_a, mu_lstm_pred, var_lstm_pred = self._estimate_likelihoods(
                                                                                                    base_model=base_model_copy, drift_model=drift_model_copy,
                                                                                                    obs=y, input_covariates=x, state_dist=self.LTd_pdf,
                                                                                                    mu_lstm_pred=mu_lstm_pred, var_lstm_pred=var_lstm_pred,
                                                                                                    base_model_prior=itv_base_model_prior, drift_model_prior=itv_drift_model_prior
                                                                                                    )
                p_yt_I_Yt1 = y_likelihood_na * x_likelihood_na * self.prior_na + y_likelihood_a * x_likelihood_a * self.prior_a
                p_a_I_Yt = (y_likelihood_a * x_likelihood_a * self.prior_a / p_yt_I_Yt1).item()
                p_anm_one_syn_ts.append(p_a_I_Yt)
                y_likelihood_a_one_ts.append(y_likelihood_a)
                y_likelihood_na_one_ts.append(y_likelihood_na)
                x_likelihood_a_one_ts.append(x_likelihood_a)
                x_likelihood_na_one_ts.append(x_likelihood_na)

                # Collect sample input
                if i >= anm_begin_list[k]:
                    LTd_mu_prior = np.array(drift_model_copy.states.mu_prior)[:, 1].flatten()
                    mu_LTd_history = self._hidden_states_collector(i - 1, LTd_mu_prior)
                    itv_LT = anm_mag_list[k]
                    itv_anm_dev_time = i - anm_begin_list[k]
                    itv_LL = itv_LT * itv_anm_dev_time
                    # # Change labels when anomaly is detected
                    # if anomaly_detected:
                    #     itv_LT = 0
                    #     itv_LL = 0
                    #     itv_anm_dev_time = 0
                    samples['LTd_history'].append(mu_LTd_history.tolist())
                    samples['itv_LT'].append(itv_LT)
                    samples['itv_LL'].append(itv_LL)
                    samples['anm_develop_time'].append(itv_anm_dev_time)

                # if p_a_I_Yt > self.detection_threshold:
                #     anomaly_detected = True
                #     # Intervene the model using true anomaly features
                #     LL_index = base_model_copy.states_name.index("local level")
                #     LT_index = base_model_copy.states_name.index("local trend")
                #     base_model_copy.mu_states[LT_index] += anm_mag_list[k]
                #     base_model_copy.mu_states[LL_index] += anm_mag_list[k] * (i - anm_begin_list[k])
                #     base_model_copy.mu_states[self.AR_index] = drift_model_copy.mu_states[2]
                #     drift_model_copy.mu_states[0] = 0
                #     drift_model_copy.mu_states[1] = self.mu_LTd

                mu_obs_pred, var_obs_pred, _, _ = base_model_copy.forward(x, mu_lstm_pred=mu_lstm_pred, var_lstm_pred=var_lstm_pred,)
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

                mu_ar_pred, var_ar_pred, _, _ = drift_model_copy.forward()
                _, _, mu_drift_states_posterior, var_drift_states_posterior = drift_model_copy.backward(
                    obs=base_model_copy.mu_states_prior[self.AR_index], 
                    obs_var=base_model_copy.var_states_prior[self.AR_index, self.AR_index])
                drift_model_copy.save_states_history()
                drift_model_copy.set_states(mu_drift_states_posterior, var_drift_states_posterior)
                mu_ar_preds.append(mu_ar_pred)
                std_ar_preds.append(var_ar_pred**0.5)

            # states_mu_prior = np.array(base_model_copy.states.mu_prior)
            # states_var_prior = np.array(base_model_copy.states.var_prior)
            # states_drift_mu_prior = np.array(drift_model_copy.states.mu_prior)
            # states_drift_var_prior = np.array(drift_model_copy.states.var_prior)

            # fig = plt.figure(figsize=(10, 9))
            # gs = gridspec.GridSpec(10, 1)
            # ax0 = plt.subplot(gs[0])
            # ax1 = plt.subplot(gs[1])
            # ax2 = plt.subplot(gs[2])
            # ax3 = plt.subplot(gs[3])
            # ax4 = plt.subplot(gs[4])
            # ax5 = plt.subplot(gs[5])
            # ax6 = plt.subplot(gs[6])
            # ax7 = plt.subplot(gs[7])
            # ax8 = plt.subplot(gs[8])
            # ax9 = plt.subplot(gs[9])
            # # print(base_model_copy.states.mu_prior)
            # ax0.plot(states_mu_prior[:, 0].flatten(), label='local level')
            # ax0.fill_between(np.arange(len(states_mu_prior[:, 0])),
            #                 states_mu_prior[:, 0].flatten() - states_var_prior[:, 0, 0]**0.5,
            #                 states_mu_prior[:, 0].flatten() + states_var_prior[:, 0, 0]**0.5,
            #                 alpha=0.5)
            # ax0.axvline(x=anm_begin_list[k], color='r', linestyle='--')
            # ax0.plot(generated_ts[k])

            # ax1.plot(states_mu_prior[:, 1].flatten(), label='local trend')
            # ax1.fill_between(np.arange(len(states_mu_prior[:, 1])),
            #                 states_mu_prior[:, 1].flatten() - states_var_prior[:, 1, 1]**0.5,
            #                 states_mu_prior[:, 1].flatten() + states_var_prior[:, 1, 1]**0.5,
            #                 alpha=0.5)
            
            # ax2.plot(states_mu_prior[:, 2].flatten(), label='lstm')
            # ax2.fill_between(np.arange(len(states_mu_prior[:, 2])),
            #                 states_mu_prior[:, 2].flatten() - states_var_prior[:, 2, 2]**0.5,
            #                 states_mu_prior[:, 2].flatten() + states_var_prior[:, 2, 2]**0.5,
            #                 alpha=0.5)
            
            # ax3.plot(states_mu_prior[:, 3].flatten(), label='autoregression')
            # ax3.fill_between(np.arange(len(states_mu_prior[:, 3])),
            #                 states_mu_prior[:, 3].flatten() - states_var_prior[:, 3, 3]**0.5,
            #                 states_mu_prior[:, 3].flatten() + states_var_prior[:, 3, 3]**0.5,
            #                 alpha=0.5)
            # ax4.plot(np.array(mu_ar_preds).flatten(), label='obs')
            # ax4.fill_between(np.arange(len(mu_ar_preds)),
            #                 np.array(mu_ar_preds).flatten() - np.array(std_ar_preds).flatten(),
            #                 np.array(mu_ar_preds).flatten() + np.array(std_ar_preds).flatten(),
            #                 alpha=0.5)
            # ax4.plot(states_drift_mu_prior[:, 0].flatten())
            # ax4.fill_between(np.arange(len(states_drift_mu_prior[:, 0])),
            #                 states_drift_mu_prior[:, 0].flatten() - states_drift_var_prior[:, 0, 0]**0.5,
            #                 states_drift_mu_prior[:, 0].flatten() + states_drift_var_prior[:, 0, 0]**0.5,
            #                 alpha=0.5)
            # ax4.set_ylabel('LLd')
            # ax5.plot(states_drift_mu_prior[:, 1].flatten())
            # ax5.fill_between(np.arange(len(states_drift_mu_prior[:, 1])),
            #                 states_drift_mu_prior[:, 1].flatten() - states_drift_var_prior[:, 1, 1]**0.5,
            #                 states_drift_mu_prior[:, 1].flatten() + states_drift_var_prior[:, 1, 1]**0.5,
            #                 alpha=0.5)
            # ax5.set_ylabel('LTd')
            # ax6.plot(states_drift_mu_prior[:, 2].flatten())
            # ax6.fill_between(np.arange(len(states_drift_mu_prior[:, 2])),
            #                 states_drift_mu_prior[:, 2].flatten() - states_drift_var_prior[:, 2, 2]**0.5,
            #                 states_drift_mu_prior[:, 2].flatten() + states_drift_var_prior[:, 2, 2]**0.5,
            #                 alpha=0.5)
            # ax6.set_ylabel('ARd')
            # ax7.plot(p_anm_one_syn_ts)
            # ax7.axvline(x=anm_begin_list[k], color='r', linestyle='--')
            # ax7.set_ylim(-0.05, 1.05)
            # ax7.set_ylabel('p_anm')
            # ax8.plot(y_likelihood_a_one_ts, label='itv')
            # ax8.plot(y_likelihood_na_one_ts, label='no itv')
            # ax8.set_ylabel('y_likelihood')
            # ax9.plot(x_likelihood_a_one_ts, label='itv')
            # ax9.plot(x_likelihood_na_one_ts, label='no itv')
            # ax9.set_ylabel('x_likelihood')
            # plt.show()
        
        samples_df = pd.DataFrame(samples)
        samples_df.to_csv(save_to_path, index=False)

    def _get_look_back_time_steps(self, current_step, step_look_back = 64):
        look_back_step_list = [0]
        current = 1
        while current <=  step_look_back:
            look_back_step_list.append(current)
            current *= 2
        look_back_step_list = [current_step - i for i in look_back_step_list]
        return look_back_step_list

    def _hidden_states_collector(self, current_step, hidden_states_all_step):
        hidden_states_all_step_numpy = np.array(np.copy(hidden_states_all_step))
        look_back_steps_list = self._get_look_back_time_steps(current_step)
        hidden_states_collected = hidden_states_all_step_numpy[look_back_steps_list]
        return hidden_states_collected
    
    def learn_intervention(self, training_samples_path):
        samples = pd.read_csv(training_samples_path)
        samples['LTd_history'] = samples['LTd_history'].apply(lambda x: list(map(float, x[1:-1].split(','))))
        # Convert samples['anm_develop_time'] to float
        samples['anm_develop_time'] = samples['anm_develop_time'].apply(lambda x: float(x))

        # Shuffle samples
        samples = samples.sample(frac=1).reset_index(drop=True)

        # Target list8
        self.target_list = ['itv_LT', 'itv_LL', 'anm_develop_time']

        self.model = TAGI_Net(len(samples['LTd_history'][0]), len(self.target_list))

        # Train the model using 50% of the samples
        n_samples = len(samples)
        n_train = int(n_samples * 0.5)
        train_samples = samples.iloc[:n_train]
        train_X = np.array(train_samples['LTd_history'].values.tolist(), dtype=np.float32)
        train_y = np.array(train_samples[self.target_list].values, dtype=np.float32)
        # Get the moments of training set, and use them to normalize the validation set and test set
        self.mean_train = train_X.mean()
        self.std_train = train_X.std()
        train_X = (train_X - self.mean_train) / self.std_train
        self.mean_target = train_y.mean(axis=0)
        self.std_target = train_y.std(axis=0)
        # # Remove when using time series with different anomaly magnitude
        # std_target[0] = 1
        # mean_target[0] = 0
        train_y = (train_y - self.mean_target) / self.std_target

        # Validation set 10% of the samples
        n_val = int(n_samples * 0.1)
        val_samples = samples.iloc[n_train:n_train+n_val]
        val_X = np.array(val_samples['LTd_history'].values.tolist(), dtype=np.float32)
        val_y = np.array(val_samples[self.target_list].values, dtype=np.float32)
        val_X = (val_X - self.mean_train) / self.std_train
        val_y = (val_y - self.mean_target) / self.std_target

        # Train the model with batch size 20
        self.batch_size = 20
        n_batch_train = n_train // self.batch_size
        n_batch_val = n_val // self.batch_size
        patience = 10
        best_loss = float('inf')
        max_training_epoch = 10
        # for epoch in range(max_training_epoch):
        for epoch in range(max_training_epoch):
            for i in range(n_batch_train):
                prediction_mu, _ = self.model.net(train_X[i*self.batch_size:(i+1)*self.batch_size])
                prediction_mu = prediction_mu.reshape(self.batch_size, len(self.target_list)*2)

                # Update model
                out_updater = OutputUpdater(self.model.net.device)
                out_updater.update_heteros(
                    output_states = self.model.net.output_z_buffer,
                    mu_obs = train_y[i*self.batch_size:(i+1)*self.batch_size].flatten(),
                    # var_obs = np.zeros_like(train_y[i*self.batch_size:(i+1)*self.batch_size].flatten()),
                    # var_obs = np.zeros_like(train_y[i*self.batch_size:(i+1)*self.batch_size].flatten()),
                    delta_states = self.model.net.input_delta_z_buffer,
                )
                self.model.net.backward()
                self.model.net.step()

            loss_val = 0
            for j in range(n_batch_val):
                val_pred_mu, _ = self.model.net(val_X[j*self.batch_size:(j+1)*self.batch_size])
                val_pred_mu = val_pred_mu.reshape(self.batch_size, len(self.target_list)*2)
                val_pred_y_mu = val_pred_mu[:, [0, 2, 4]]
                val_y_batch = val_y[j*self.batch_size:(j+1)*self.batch_size]
                # Compute the mse between val_pred_y_mu and val_y_batch
                loss_val += ((val_pred_y_mu - val_y_batch)**2).mean()
            loss_val /= n_batch_val

            print(f'Epoch {epoch}: {loss_val}')
            # Early stopping with patience 10
            if loss_val < best_loss:
                best_loss = loss_val
                patience = 10
            else:
                patience -= 1
                if patience == 0:
                    break

        # Test the model using 10% of the samples
        n_test = int(n_samples * 0.1)
        test_samples = samples.iloc[n_train+n_val:n_train+n_val+n_test]
        test_X = np.array(test_samples['LTd_history'].tolist(), dtype=np.float32)
        test_y = np.array(test_samples[self.target_list].values, dtype=np.float32)
        test_X = (test_X - self.mean_train) / self.std_train
        test_y = (test_y - self.mean_target) / self.std_target

        n_batch_test = n_test // self.batch_size

        loss_test = 0
        for j in range(n_batch_val):
            test_pred_mu, test_pred_var = self.model.net(test_X[j*self.batch_size:(j+1)*self.batch_size])
            test_pred_mu = test_pred_mu.reshape(self.batch_size, len(self.target_list)*2)
            test_pred_y_mu = test_pred_mu[:, [0, 2, 4]]
            test_pred_y_var = test_pred_mu[:, [1, 3, 5]]
            test_y_batch = test_y[j*self.batch_size:(j+1)*self.batch_size]
            # Compute the mse between test_pred_y_mu and test_y_batch
            loss_test += ((test_pred_y_mu - test_y_batch)**2).mean()
        loss_test /= n_batch_test

        print(f'Test loss {loss_test.item()}')

        # Denormalize the prediction
        y_pred_denorm = test_pred_y_mu * self.std_target + self.mean_target
        y_pred_var_denorm = test_pred_y_var * self.std_target ** 2
        y_test_denorm = test_y_batch * self.std_target + self.mean_target
        print(y_test_denorm.tolist())
        print(y_pred_denorm.tolist())
        print(np.sqrt(y_pred_var_denorm))


class TAGI_Net():
    def __init__(self, n_observations, n_actions):
        super(TAGI_Net, self).__init__()
        self.net = Sequential(
                    Linear(n_observations, 64),
                    # Linear(n_observations, 64, gain_weight=0.1, gain_bias=0.1),
                    ReLU(),
                    Linear(64, 32),
                    # Linear(64, 32, gain_weight=0.1, gain_bias=0.1),
                    ReLU(),
                    Linear(32, n_actions * 2),
                    # Linear(32, n_actions * 2, gain_weight=0.1, gain_bias=0.1),
                    EvenExp()
                    )
        self.n_actions = n_actions
        self.n_observations = n_observations
    def forward(self, mu_x, var_x):
        return self.net.forward(mu_x, var_x)