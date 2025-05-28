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
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch


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
        self.mu_obs_preds, self.std_obs_preds = [], []
        self.mu_ar_preds, self.std_ar_preds = [], []
        self.prior_na, self.prior_a = 0.998, 0.002
        self.detection_threshold = 0.5
        self.mu_itv_all, self.std_itv_all = [], []
        self.nn_train_with = 'tagiv'
        self.current_time_step = 0
        self.lstm_history = []
        self.lstm_cell_states = []
        self.mean_train, self.std_train, self.mean_target, self.std_target = None, None, None, None

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

        lstm_index = self.base_model.get_states_index("lstm")
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
                self.base_model.lstm_output_history.update(
                    mu_states_posterior[lstm_index],
                    var_states_posterior[lstm_index, lstm_index],
                )

            self.base_model._save_states_history()
            self.base_model.set_states(mu_states_posterior, var_states_posterior)
            mu_obs_preds.append(mu_obs_pred)
            std_obs_preds.append(var_obs_pred**0.5)

            # Drift model filter process
            mu_ar_pred, var_ar_pred, mu_drift_states_prior, _ = self.drift_model.forward()
            _, _, mu_drift_states_posterior, var_drift_states_posterior = self.drift_model.backward(
                obs=self.base_model.mu_states_prior[self.AR_index], 
                obs_var=self.base_model.var_states_prior[self.AR_index, self.AR_index])
            self.drift_model._save_states_history()
            self.drift_model.set_states(mu_drift_states_posterior, var_drift_states_posterior)
            mu_ar_preds.append(mu_ar_pred)
            std_ar_preds.append(var_ar_pred**0.5)

            if buffer_LTd:
                self.LTd_buffer.append(mu_drift_states_prior[1].item())
            
            # Dummy values for p_anm when it is not in detect mode
            self.p_anm_all.append(0)
            self.mu_itv_all.append([np.nan, np.nan, np.nan])
            self.std_itv_all.append([np.nan, np.nan, np.nan])

            self.current_time_step += 1

        return np.array(mu_obs_preds).flatten(), np.array(std_obs_preds).flatten(), np.array(mu_ar_preds).flatten(), np.array(std_ar_preds).flatten()
    
    def estimate_LTd_dist(self):
        print('mean and std before roll out synthetic data', np.mean(self.LTd_buffer), np.std(self.LTd_buffer))
        # Roll out ten synthetic time series
        # # Generate synthetic time series
        covariate_col = self.data_processor.covariates_col
        train_index, val_index, test_index = self.data_processor.get_split_indices()
        time_covariate_info = {'initial_time_covariate': self.data_processor.data.values[val_index[-1], self.data_processor.covariates_col].item(),
                                'mu': self.data_processor.norm_const_mean[covariate_col], 
                                'std': self.data_processor.norm_const_std[covariate_col]}
        generated_ts, time_covariate, _, _ = self.base_model.generate_time_series(num_time_series=1, num_time_steps=52*10, 
                                                                time_covariates=self.data_processor.time_covariates, 
                                                                time_covariate_info=time_covariate_info,
                                                                add_anomaly=False, sample_from_lstm_pred=False)
        # # Run the current model on the synthetic time series
        if "lstm" in self.base_model.states_name:
            lstm_index = self.base_model.get_states_index("lstm")
            output_history_temp = copy.deepcopy(self.base_model.lstm_output_history)
            cell_states_temp = copy.deepcopy(self.base_model.lstm_net.get_lstm_states())
        for k in tqdm(range(len(generated_ts))):
            base_model_copy = copy.deepcopy(self.base_model)
            if "lstm" in self.base_model.states_name:
                base_model_copy.lstm_net = self.base_model.lstm_net
                base_model_copy.lstm_output_history = copy.deepcopy(output_history_temp)
                base_model_copy.lstm_net.set_lstm_states(cell_states_temp)
            drift_model_copy = copy.deepcopy(self.drift_model)

            base_model_copy.initialize_states_history()
            drift_model_copy.initialize_states_history()
            mu_ar_preds, std_ar_preds = [], []

            for i, (x, y) in enumerate(zip(time_covariate, generated_ts[k])):

                _, _, _, _ = base_model_copy.forward(x)
                (
                    _, _,
                    mu_states_posterior,
                    var_states_posterior,
                ) = base_model_copy.backward(y)

                if "lstm" in base_model_copy.states_name:
                    base_model_copy.lstm_output_history.update(
                        mu_states_posterior[lstm_index],
                        var_states_posterior[lstm_index, lstm_index],
                    )

                base_model_copy._save_states_history()
                base_model_copy.set_states(mu_states_posterior, var_states_posterior)

                mu_ar_pred, var_ar_pred, mu_drift_states_prior, _ = drift_model_copy.forward()
                _, _, mu_drift_states_posterior, var_drift_states_posterior = drift_model_copy.backward(
                    obs=base_model_copy.mu_states_prior[self.AR_index], 
                    obs_var=base_model_copy.var_states_prior[self.AR_index, self.AR_index])
                drift_model_copy._save_states_history()
                drift_model_copy.set_states(mu_drift_states_posterior, var_drift_states_posterior)
                self.LTd_buffer.append(mu_drift_states_prior[1].item())
                mu_ar_preds.append(mu_ar_pred)
                std_ar_preds.append(var_ar_pred**0.5)

            # states_mu_prior = np.array(base_model_copy.states.mu_prior)
            # states_var_prior = np.array(base_model_copy.states.var_prior)
            # states_drift_mu_prior = np.array(drift_model_copy.states.mu_prior)
            # states_drift_var_prior = np.array(drift_model_copy.states.var_prior)

            # fig = plt.figure(figsize=(10, 9))
            # gs = gridspec.GridSpec(7, 1)
            # ax0 = plt.subplot(gs[0])
            # ax1 = plt.subplot(gs[1])
            # ax2 = plt.subplot(gs[2])
            # ax3 = plt.subplot(gs[3])
            # ax4 = plt.subplot(gs[4])
            # ax5 = plt.subplot(gs[5])
            # ax6 = plt.subplot(gs[6])
            # # print(base_model_copy.states.mu_prior)
            # ax0.plot(states_mu_prior[:, 0].flatten(), label='local level')
            # ax0.fill_between(np.arange(len(states_mu_prior[:, 0])),
            #                 states_mu_prior[:, 0].flatten() - states_var_prior[:, 0, 0]**0.5,
            #                 states_mu_prior[:, 0].flatten() + states_var_prior[:, 0, 0]**0.5,
            #                 alpha=0.5)
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
            # plt.show()

        self.mu_LTd = np.mean(self.LTd_buffer)
        self.LTd_std = np.std(self.LTd_buffer)
        self.LTd_pdf = common.gaussian_pdf(mu = self.mu_LTd, std = self.LTd_std)
        print('mean and std after roll out synthetic data',self.mu_LTd, self.LTd_std)

    def tune(self, decay_factor: Optional[float] = 0.9):
        '''
        Tune the std_LTd using synthetic time series
        '''
        covariate_col = self.data_processor.covariates_col
        train_index, val_index, test_index = self.data_processor.get_split_indices()
        time_covariate_info = {'initial_time_covariate': self.data_processor.data.values[val_index[-1], self.data_processor.covariates_col].item(),
                                'mu': self.data_processor.norm_const_mean[covariate_col], 
                                'std': self.data_processor.norm_const_std[covariate_col]}
        generated_ts, time_covariate, anm_mag_list, anm_begin_list = self.base_model.generate_time_series(num_time_series=10, num_time_steps=52*10, 
                                                                time_covariates=self.data_processor.time_covariates, 
                                                                time_covariate_info=time_covariate_info,
                                                                add_anomaly=False)
                                                                # add_anomaly=True, anomaly_mag_range=[-1/52, 1/52], 
                                                                # anomaly_begin_range=[int(52*10/4), int(52*10*3/8)], sample_from_lstm_pred=False)

        std_LTd_coeff = 1 / decay_factor
        LTd_std_original = copy.deepcopy(self.LTd_std)
        false_alarm = False

        while not false_alarm:
            std_LTd_coeff *= decay_factor
            self.LTd_std = LTd_std_original * std_LTd_coeff
            self.LTd_pdf = common.gaussian_pdf(mu = self.mu_LTd, std = self.LTd_std)
            print(f'Tuning LTd_std with coefficient of {std_LTd_coeff}; LTd_std is {self.LTd_std} ...')
        
            # # Run the current model on the synthetic time series
            if "lstm" in self.base_model.states_name:
                lstm_index = self.base_model.get_states_index("lstm")
                output_history_temp = copy.deepcopy(self.base_model.lstm_output_history)
                cell_states_temp = copy.deepcopy(self.base_model.lstm_net.get_lstm_states())
            for k in range(len(generated_ts)):
                base_model_copy = copy.deepcopy(self.base_model)
                if "lstm" in self.base_model.states_name:
                    base_model_copy.lstm_net = self.base_model.lstm_net
                    base_model_copy.lstm_output_history = copy.deepcopy(output_history_temp)
                    base_model_copy.lstm_net.set_lstm_states(cell_states_temp)
                drift_model_copy = copy.deepcopy(self.drift_model)

                base_model_copy.initialize_states_history()
                drift_model_copy.initialize_states_history()
                mu_ar_preds, std_ar_preds = [], []
                p_anm_one_syn_ts = []

                for i, (x, y) in enumerate(zip(time_covariate, generated_ts[k])):

                    # Estimate likelihood without intervention
                    y_likelihood_na, x_likelihood_na = self._estimate_likelihoods(base_model=base_model_copy, drift_model=drift_model_copy,
                                                                                    obs=y, input_covariates=x, state_dist=self.LTd_pdf)
                    # Estimate likelihood with intervention
                    itv_base_model_prior, itv_drift_model_prior = self._intervene_current_priors(base_model=base_model_copy, drift_model=drift_model_copy,)
                    y_likelihood_a, x_likelihood_a = self._estimate_likelihoods(
                                                                                base_model=base_model_copy, drift_model=drift_model_copy,
                                                                                obs=y, input_covariates=x, state_dist=self.LTd_pdf,
                                                                                base_model_prior=itv_base_model_prior, drift_model_prior=itv_drift_model_prior
                                                                                )
                    p_yt_I_Yt1 = y_likelihood_na * x_likelihood_na * self.prior_na + y_likelihood_a * x_likelihood_a * self.prior_a
                    p_a_I_Yt = (y_likelihood_a * x_likelihood_a * self.prior_a / p_yt_I_Yt1).item()
                    p_anm_one_syn_ts.append(p_a_I_Yt)

                    mu_obs_pred, var_obs_pred, _, _ = base_model_copy.forward(x)
                    (
                        _, _,
                        mu_states_posterior,
                        var_states_posterior,
                    ) = base_model_copy.backward(y)

                    if "lstm" in base_model_copy.states_name:
                        base_model_copy.lstm_output_history.update(
                            mu_states_posterior[lstm_index],
                            var_states_posterior[lstm_index, lstm_index],
                        )

                    base_model_copy._save_states_history()
                    base_model_copy.set_states(mu_states_posterior, var_states_posterior)

                    mu_ar_pred, var_ar_pred, _, _ = drift_model_copy.forward()
                    _, _, mu_drift_states_posterior, var_drift_states_posterior = drift_model_copy.backward(
                        obs=base_model_copy.mu_states_prior[self.AR_index], 
                        obs_var=base_model_copy.var_states_prior[self.AR_index, self.AR_index])
                    drift_model_copy._save_states_history()
                    drift_model_copy.set_states(mu_drift_states_posterior, var_drift_states_posterior)
                    mu_ar_preds.append(mu_ar_pred)
                    std_ar_preds.append(var_ar_pred**0.5)

                # states_mu_prior = np.array(base_model_copy.states.mu_prior)
                # states_var_prior = np.array(base_model_copy.states.var_prior)
                # states_drift_mu_prior = np.array(drift_model_copy.states.mu_prior)
                # states_drift_var_prior = np.array(drift_model_copy.states.var_prior)

                # fig = plt.figure(figsize=(10, 9))
                # gs = gridspec.GridSpec(8, 1)
                # ax0 = plt.subplot(gs[0])
                # ax1 = plt.subplot(gs[1])
                # ax2 = plt.subplot(gs[2])
                # ax3 = plt.subplot(gs[3])
                # ax4 = plt.subplot(gs[4])
                # ax5 = plt.subplot(gs[5])
                # ax6 = plt.subplot(gs[6])
                # ax7 = plt.subplot(gs[7])
                # # print(base_model_copy.states.mu_prior)
                # ax0.plot(states_mu_prior[:, 0].flatten(), label='local level')
                # ax0.fill_between(np.arange(len(states_mu_prior[:, 0])),
                #                 states_mu_prior[:, 0].flatten() - states_var_prior[:, 0, 0]**0.5,
                #                 states_mu_prior[:, 0].flatten() + states_var_prior[:, 0, 0]**0.5,
                #                 alpha=0.5)
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
                # ax7.set_ylim(-0.05, 1.05)
                # ax7.set_ylabel('p_anm')
                # plt.show()

                if (len(np.where(np.array(p_anm_one_syn_ts) > 0.5)[0]) > 0.5):
                    false_alarm = True
                    break
        
        self.LTd_std = self.LTd_std/decay_factor # Revert the LTd_std to the last one before false alarm is raised
        self.LTd_pdf = common.gaussian_pdf(mu = self.mu_LTd, std = self.LTd_std)
        print(f'LTd_std is tuned to coefficient of {std_LTd_coeff/decay_factor}; LTd_std is tuned to {self.LTd_std} ...')

    def detect(
            self, 
            data,
            apply_intervention: Optional[bool] = False,
            ):

        lstm_index = self.base_model.get_states_index("lstm")
        mu_lstm_pred, var_lstm_pred = None, None

        # # # Set drift model rigid prior
        # self.drift_model.var_states = np.diag([1e-12, 1e-12, self.ar_component.var_states.item()])

        # for i, (x, y) in enumerate(zip(data["x"], data["y"])):
        i = 0
        i_before_retract = 0
        trigger = False
        rerun_kf = False
        while i < len(data["x"]):
            if i > i_before_retract:
                rerun_kf = False
            if rerun_kf is False:
                # Estimate likelihoods
                # Estimate likelihood without intervention
                y_likelihood_na, x_likelihood_na = self._estimate_likelihoods(base_model=self.base_model, drift_model=self.drift_model,
                                                                            obs=data["y"][i], input_covariates=data["x"][i], state_dist=self.LTd_pdf)
                # Estimate likelihood with intervention
                itv_base_model_prior, itv_drift_model_prior = self._intervene_current_priors(base_model=self.base_model, drift_model=self.drift_model,)
                y_likelihood_a, x_likelihood_a = self._estimate_likelihoods(
                                                                            base_model=self.base_model, drift_model=self.drift_model,
                                                                            obs=data["y"][i], input_covariates=data["x"][i], state_dist=self.LTd_pdf,
                                                                            base_model_prior=itv_base_model_prior, drift_model_prior=itv_drift_model_prior
                                                                            )
                p_yt_I_Yt1 = y_likelihood_na * x_likelihood_na * self.prior_na + y_likelihood_a * x_likelihood_a * self.prior_a
                # p_na_I_Yt = y_likelihood_na * x_likelihood_na * p_na_I_Yt1 / p_yt_I_Yt1
                p_a_I_Yt = (y_likelihood_a * x_likelihood_a * self.prior_a / p_yt_I_Yt1).item()
                self.p_anm_all.append(p_a_I_Yt)

                # Track what NN learns
                LTd_mu_prior = np.array(self.drift_model.states.mu_prior)[:, 1].flatten()
                # LTd_history = self._hidden_states_collector(i - 1, LTd_mu_prior)
                LTd_history = self._hidden_states_collector(self.current_time_step - 1, LTd_mu_prior)
                LTd_history = np.array(LTd_history.tolist(), dtype=np.float32)
                LTd_history = (LTd_history - self.mean_train) / self.std_train
                if self.nn_train_with == 'tagiv':
                    LTd_history = np.repeat(LTd_history[np.newaxis, :], self.batch_size, axis=0)

                    output_pred_mu, output_pred_var = self.model.net(LTd_history)
                    output_pred_mu = output_pred_mu.reshape(self.batch_size, len(self.target_list)*2)
                    output_pred_var = output_pred_var.reshape(self.batch_size, len(self.target_list)*2)
                    itv_pred_mu = output_pred_mu[0, [0, 2, 4]]
                    itv_pred_var = output_pred_mu[0, [1, 3, 5]]
                elif self.nn_train_with == 'backprop':
                    LTd_history = torch.tensor(LTd_history)
                    itv_pred_mu = self.model(LTd_history)
                    itv_pred_mu = itv_pred_mu.detach().numpy()
                    itv_pred_var = np.zeros_like(itv_pred_mu)
                    
                itv_pred_mu_denorm = itv_pred_mu * self.std_target + self.mean_target
                itv_pred_var_denorm = itv_pred_var * self.std_target ** 2

                self.mu_itv_all.append(itv_pred_mu_denorm.tolist())
                self.std_itv_all.append(np.sqrt(itv_pred_var_denorm).tolist())

            if "lstm" in self.base_model.states_name:
                self._save_lstm_input()

            if apply_intervention:
                if rerun_kf is False:
                    if p_a_I_Yt > self.detection_threshold:
                        rerun_kf = True
                        # To control that during the rerun from the past, the agent cannnot trigger again
                        i_before_retract = copy.copy(i)
                        # Retract agent
                        step_back = max(int(itv_pred_mu_denorm[2]), 1) if max(int(itv_pred_mu_denorm[2]), 1) < i else i - 2
                        self._retract_agent(time_step_back=step_back)
                        i = i - step_back
                        self.current_time_step = self.current_time_step - step_back

                        # Apply intervention on base_model hidden states
                        LL_index = self.base_model.states_name.index("local level")
                        LT_index = self.base_model.states_name.index("local trend")
                        AR_index = self.base_model.states_name.index("autoregression")
                        self.base_model.mu_states[LL_index] += itv_pred_mu_denorm[1]
                        self.base_model.mu_states[LT_index] += itv_pred_mu_denorm[0]
                        self.base_model.var_states[LL_index, LL_index] += itv_pred_var_denorm[1]
                        self.base_model.var_states[LT_index, LT_index] += itv_pred_var_denorm[0]

                        self.drift_model.mu_states[0] = 0
                        self.drift_model.mu_states[1] = self.mu_LTd
                        trigger = True

            # if trigger is False:
            #     if i == len(data["x"]) - 1:
            #         self._retract_agent(time_step_back=len(data["x"]) - 1 - 100)
            #         # current_step_before_retract = copy.copy(i)
            #         i = 100
            #         self.current_time_step = self.current_time_step - (len(data["x"]) - 1 - 100)
            #         trigger = True

            # Base model filter process, same as in model.py
            # mu_obs_pred, var_obs_pred, _, _ = self.base_model.forward(data["x"][i])
            mu_obs_pred, var_obs_pred, _, _ = self.base_model.forward(data["x"][i])

            (
                _, _,
                mu_states_posterior,
                var_states_posterior,
            ) = self.base_model.backward(data["y"][i])

            if self.base_model.lstm_net:
                self.base_model.lstm_output_history.update(
                    mu_states_posterior[lstm_index],
                    var_states_posterior[lstm_index, lstm_index],
                )

            self.base_model._save_states_history()
            self.base_model.set_states(mu_states_posterior, var_states_posterior)
            self.mu_obs_preds.append(mu_obs_pred)
            self.std_obs_preds.append(var_obs_pred**0.5)

            # Drift model filter process
            mu_ar_pred, var_ar_pred, mu_drift_states_prior, _ = self.drift_model.forward()
            _, _, mu_drift_states_posterior, var_drift_states_posterior = self.drift_model.backward(
                obs=self.base_model.mu_states_prior[self.AR_index], 
                obs_var=self.base_model.var_states_prior[self.AR_index, self.AR_index])
            self.drift_model._save_states_history()
            self.drift_model.set_states(mu_drift_states_posterior, var_drift_states_posterior)
            self.mu_ar_preds.append(mu_ar_pred)
            self.std_ar_preds.append(var_ar_pred**0.5)

            self.current_time_step += 1
            i += 1

        return np.array(self.mu_obs_preds).flatten(), np.array(self.std_obs_preds).flatten(), np.array(self.mu_ar_preds).flatten(), np.array(self.std_ar_preds).flatten()

    def _estimate_likelihoods(
            self, 
            base_model: Model,
            drift_model: Model,
            obs: float,
            state_dist: Optional[Callable] = None,
            input_covariates: Optional[np.ndarray] = None,
            base_model_prior: Optional[Dict] = None,
            drift_model_prior: Optional[Dict] = None,
            ):
        """
        Compute the likelihood of observation and hidden states given action
        """
        base_model_copy = copy.deepcopy(base_model)
        drift_model_copy = copy.deepcopy(drift_model)
        if "lstm" in base_model.states_name:
            output_history_temp = copy.deepcopy(base_model.lstm_output_history)
            cell_states_temp = copy.deepcopy(base_model.lstm_net.get_lstm_states())
            base_model_copy.lstm_net = base_model.lstm_net
            base_model_copy.lstm_output_history = copy.deepcopy(output_history_temp)
            base_model_copy.lstm_net.set_lstm_states(cell_states_temp)

        if base_model_prior is not None and drift_model_prior is not None:
            base_model_copy.mu_states = base_model_prior['mu']
            base_model_copy.var_states = base_model_prior['var']
            drift_model_copy.mu_states = drift_model_prior['mu']
            drift_model_copy.var_states = drift_model_prior['var']

        mu_obs_pred, var_obs_pred, _, _ = base_model_copy.forward(input_covariates = input_covariates)

        y_likelihood = likelihood(mu_obs_pred, np.sqrt(var_obs_pred), obs)

        _, _, mu_d_states_prior, _ = drift_model_copy.forward()
        x_likelihood = state_dist(mu_d_states_prior[1].item())

        if "lstm" in base_model.states_name:
            # Set the base_model back to the original state
            base_model.lstm_output_history = copy.deepcopy(output_history_temp)
            base_model.lstm_net.set_lstm_states(cell_states_temp)
        return y_likelihood.item(), x_likelihood
    
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
        samples = {'LTd_history': [], 'itv_LT': [], 'itv_LL': [], 'anm_develop_time': [], 'p_anm': []}

        # Anomly feature range define
        ts_len = 52*6
        stationary_ar_std = self.ar_component.std_error/(1-self.ar_component.phi**2)**0.5
        # anm_mag_range = [stationary_ar_std/80, stationary_ar_std/80]      # Same anm mag
        # anm_mag_range = [-1/52, 1/52]       # LT anm mag
        anm_mag_range = [0, 0]       # LT anm mag
        # anm_mag_range = [-10*stationary_ar_std, 10*stationary_ar_std]       # LL anm mag
        anm_begin_range = [int(ts_len/4), int(ts_len*3/8)]

        # # Generate synthetic time series
        covariate_col = self.data_processor.covariates_col
        train_index, val_index, test_index = self.data_processor.get_split_indices()
        time_covariate_info = {'initial_time_covariate': self.data_processor.data.values[val_index[-1], self.data_processor.covariates_col].item(),
                                'mu': self.data_processor.norm_const_mean[covariate_col], 
                                'std': self.data_processor.norm_const_std[covariate_col]}
        generated_ts, time_covariate, anm_mag_list, anm_begin_list = self.base_model.generate_time_series(num_time_series=num_time_series, num_time_steps=ts_len, 
                                                                time_covariates=self.data_processor.time_covariates, 
                                                                time_covariate_info=time_covariate_info,
                                                                add_anomaly=True, anomaly_mag_range=anm_mag_range, 
                                                                anomaly_begin_range=anm_begin_range, sample_from_lstm_pred=False)
        # Plot generated time series
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 1)
        ax0 = plt.subplot(gs[0])
        norm_data = self.data_processor.normalize_data()
        for j in range(len(generated_ts)):
            ax0.plot(np.concatenate((norm_data[train_index, self.data_processor.output_col].reshape(-1), 
                                        norm_data[val_index, self.data_processor.output_col].reshape(-1), 
                                        generated_ts[j])))
        ax0.axvline(x=len(self.data_processor.data.values[train_index, self.data_processor.output_col].reshape(-1))+len(self.data_processor.data.values[val_index, self.data_processor.output_col].reshape(-1)), color='r', linestyle='--')
        ax0.set_title("Data generation")
        plt.show()

        # # Run the current model on the synthetic time series
        if "lstm" in self.base_model.states_name:
            lstm_index = self.base_model.get_states_index("lstm")
            lstm_cell_states = copy.deepcopy(self.base_model.lstm_net.get_lstm_states())
            output_history_temp = copy.deepcopy(self.base_model.lstm_output_history)
        for k in tqdm(range(len(generated_ts))):
            base_model_copy = copy.deepcopy(self.base_model)
            if "lstm" in self.base_model.states_name:
                base_model_copy.lstm_net = self.base_model.lstm_net
                base_model_copy.lstm_output_history = copy.deepcopy(output_history_temp)
                base_model_copy.lstm_net.set_lstm_states(lstm_cell_states)
            drift_model_copy = copy.deepcopy(self.drift_model)

            mu_obs_preds, std_obs_preds = [], []
            mu_ar_preds, std_ar_preds = [], []
            p_anm_one_syn_ts = []
            y_likelihood_a_one_ts, y_likelihood_na_one_ts = [], []
            x_likelihood_a_one_ts, x_likelihood_na_one_ts = [], []
            base_model_copy.initialize_states_history()
            drift_model_copy.initialize_states_history()

            for i, (x, y) in enumerate(zip(time_covariate, generated_ts[k])):
                # Estimate likelihood without intervention
                y_likelihood_na, x_likelihood_na = self._estimate_likelihoods(base_model=base_model_copy, drift_model=drift_model_copy,
                                                                                obs=y, input_covariates=x, state_dist=self.LTd_pdf)
                # Estimate likelihood with intervention
                itv_base_model_prior, itv_drift_model_prior = self._intervene_current_priors(base_model=base_model_copy, drift_model=drift_model_copy,)
                y_likelihood_a, x_likelihood_a = self._estimate_likelihoods(
                                                                            base_model=base_model_copy, drift_model=drift_model_copy,
                                                                            obs=y, input_covariates=x, state_dist=self.LTd_pdf,
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
                if i > 65:
                    LTd_mu_prior = np.array(drift_model_copy.states.mu_prior)[:, 1].flatten()
                    mu_LTd_history = self._hidden_states_collector(i - 1, LTd_mu_prior)
                    samples['LTd_history'].append(mu_LTd_history.tolist())
                if i > 65 and i < anm_begin_list[k]:
                    samples['itv_LT'].append(0.)
                    samples['itv_LL'].append(0.)
                    samples['anm_develop_time'].append(0.)
                    samples['p_anm'].append(0.)
                elif i >= anm_begin_list[k]:
                    # LT anomaly label
                    itv_LT = anm_mag_list[k]
                    itv_anm_dev_time = i - anm_begin_list[k]
                    itv_LL = itv_LT * itv_anm_dev_time
                    # # LL anomaly label
                    # itv_LT = 0
                    # itv_anm_dev_time = i - anm_begin_list[k]
                    # itv_LL = anm_mag_list[k]
                    
                    samples['itv_LT'].append(itv_LT)
                    samples['itv_LL'].append(itv_LL)
                    samples['anm_develop_time'].append(itv_anm_dev_time)
                    samples['p_anm'].append(p_a_I_Yt)

                # if p_a_I_Yt > self.detection_threshold:
                #     anomaly_detected = True
                #     break
                #     # Intervene the model using true anomaly features
                #     LL_index = base_model_copy.states_name.index("local level")
                #     LT_index = base_model_copy.states_name.index("local trend")
                #     base_model_copy.mu_states[LT_index] += anm_mag_list[k]
                #     base_model_copy.mu_states[LL_index] += anm_mag_list[k] * (i - anm_begin_list[k])
                #     base_model_copy.mu_states[self.AR_index] = drift_model_copy.mu_states[2]
                #     drift_model_copy.mu_states[0] = 0
                #     drift_model_copy.mu_states[1] = self.mu_LTd

                mu_obs_pred, var_obs_pred, _, _ = base_model_copy.forward(x)
                (
                    _, _,
                    mu_states_posterior,
                    var_states_posterior,
                ) = base_model_copy.backward(y)

                if "lstm" in base_model_copy.states_name:
                    base_model_copy.lstm_output_history.update(
                        mu_states_posterior[lstm_index],
                        var_states_posterior[lstm_index, lstm_index],
                    )

                base_model_copy._save_states_history()
                base_model_copy.set_states(mu_states_posterior, var_states_posterior)
                mu_obs_preds.append(mu_obs_pred)
                std_obs_preds.append(var_obs_pred**0.5)

                mu_ar_pred, var_ar_pred, _, _ = drift_model_copy.forward()
                _, _, mu_drift_states_posterior, var_drift_states_posterior = drift_model_copy.backward(
                    obs=base_model_copy.mu_states_prior[self.AR_index], 
                    obs_var=base_model_copy.var_states_prior[self.AR_index, self.AR_index])
                drift_model_copy._save_states_history()
                drift_model_copy.set_states(mu_drift_states_posterior, var_drift_states_posterior)
                mu_ar_preds.append(mu_ar_pred)
                std_ar_preds.append(var_ar_pred**0.5)

            states_mu_prior = np.array(base_model_copy.states.mu_prior)
            states_var_prior = np.array(base_model_copy.states.var_prior)
            states_drift_mu_prior = np.array(drift_model_copy.states.mu_prior)
            states_drift_var_prior = np.array(drift_model_copy.states.var_prior)

            fig = plt.figure(figsize=(10, 9))
            gs = gridspec.GridSpec(10, 1)
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[2])
            ax3 = plt.subplot(gs[3])
            ax4 = plt.subplot(gs[4])
            ax5 = plt.subplot(gs[5])
            ax6 = plt.subplot(gs[6])
            ax7 = plt.subplot(gs[7])
            ax8 = plt.subplot(gs[8])
            ax9 = plt.subplot(gs[9])
            # print(base_model_copy.states.mu_prior)
            ax0.plot(states_mu_prior[:, 0].flatten(), label='local level')
            ax0.fill_between(np.arange(len(states_mu_prior[:, 0])),
                            states_mu_prior[:, 0].flatten() - states_var_prior[:, 0, 0]**0.5,
                            states_mu_prior[:, 0].flatten() + states_var_prior[:, 0, 0]**0.5,
                            alpha=0.5)
            ax0.axvline(x=anm_begin_list[k], color='r', linestyle='--')
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
            ax7.plot(p_anm_one_syn_ts)
            ax7.axvline(x=anm_begin_list[k], color='r', linestyle='--')
            ax7.set_ylim(-0.05, 1.05)
            ax7.set_ylabel('p_anm')
            ax8.plot(y_likelihood_a_one_ts, label='itv')
            ax8.plot(y_likelihood_na_one_ts, label='no itv')
            ax8.set_ylabel('y_likelihood')
            ax9.plot(x_likelihood_a_one_ts, label='itv')
            ax9.plot(x_likelihood_na_one_ts, label='no itv')
            ax9.set_ylabel('x_likelihood')
            plt.show()
        
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
    
    def learn_intervention(self, training_samples_path, save_model_path=None, load_model_path=None, max_training_epoch=10):
        samples = pd.read_csv(training_samples_path)
        samples['LTd_history'] = samples['LTd_history'].apply(lambda x: list(map(float, x[1:-1].split(','))))
        # Convert samples['anm_develop_time'] to float
        samples['anm_develop_time'] = samples['anm_develop_time'].apply(lambda x: float(x))

        # Shuffle samples
        samples = samples.sample(frac=1).reset_index(drop=True)

        # Target list
        self.target_list = ['itv_LT', 'itv_LL', 'anm_develop_time']

        samples_input = np.array(samples['LTd_history'].values.tolist(), dtype=np.float32)
        samples_target = np.array(samples[self.target_list].values, dtype=np.float32)
        samples_p_anm = np.array(samples['p_anm'].values.tolist(), dtype=np.float32)

        # Find where samples_p_anm is 0
        zero_indices = np.where(samples_p_anm == 0)[0]
        # Remove those samples
        samples_input = np.delete(samples_input, zero_indices, axis=0)
        samples_target = np.delete(samples_target, zero_indices, axis=0)
        samples_p_anm = np.delete(samples_p_anm, zero_indices, axis=0)

        # panm_b5_indices = np.where(samples_p_anm > 0.5)[0]
        # samples_p_anm = np.delete(samples_p_anm, panm_b5_indices, axis=0)
        # samples_input = np.delete(samples_input, panm_b5_indices, axis=0)
        # samples_target = np.delete(samples_target, panm_b5_indices, axis=0)

        # Train the model using 80% of the samples
        n_samples = len(samples_input)
        n_train = int(n_samples * 0.8)
        # train_samples = samples.iloc[:n_train]
        train_X = samples_input[:n_train]
        train_y = samples_target[:n_train]
        # Get the moments of training set, and use them to normalize the validation set and test set
        if self.mean_train is None or self.std_train is None or self.mean_target is None or self.std_target is None:
            print('skip')
            self.mean_train = train_X.mean()
            self.std_train = train_X.std()
            self.mean_target = train_y.mean(axis=0)
            self.std_target = train_y.std(axis=0)
        print('mean and std of training input', self.mean_train, self.std_train)
        print('mean and std of training target', self.mean_target, self.std_target)

        train_X = (train_X - self.mean_train) / self.std_train

        # # Remove when using time series with different anomaly magnitude
        # self.mean_target[0] = 0
        # self.std_target[0] = 1
        # self.mean_target = np.zeros_like(self.mean_target)
        # self.std_target = np.ones_like(self.std_target)
        train_y = (train_y - self.mean_target) / self.std_target

        # Validation set 10% of the samples
        n_val = int(n_samples * 0.1)
        val_X = samples_input[n_train:n_train+n_val]
        val_y = samples_target[n_train:n_train+n_val]
        val_X = (val_X - self.mean_train) / self.std_train
        val_y = (val_y - self.mean_target) / self.std_target

        # Test the model using 10% of the samples
        n_test = int(n_samples * 0.1)
        test_X = samples_input[n_train+n_val:n_train+n_val+n_test]
        test_y = samples_target[n_train+n_val:n_train+n_val+n_test]
        test_X = (test_X - self.mean_train) / self.std_train
        test_y = (test_y - self.mean_target) / self.std_target

        if self.nn_train_with == 'tagiv':
            self.model = TAGI_Net(len(samples['LTd_history'][0]), len(self.target_list))
        elif self.nn_train_with == 'backprop':
            self.model = NN(input_size = len(samples['LTd_history'][0]), output_size = len(self.target_list))
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            train_X = torch.tensor(train_X)
            train_y = torch.tensor(train_y)
            val_X = torch.tensor(val_X)
            val_y = torch.tensor(val_y)
            test_X = torch.tensor(test_X)
            test_y = torch.tensor(test_y)

        self.batch_size = 20

        if load_model_path is not None:
            if self.nn_train_with == 'tagiv':
                with open(load_model_path, 'rb') as f:
                    param_dict = pickle.load(f)
                self.model.net.load_state_dict(param_dict)
            elif self.nn_train_with == 'backprop':
                pass    # TODO
        else:
            # Train the model with batch size 20
            n_batch_train = n_train // self.batch_size
            n_batch_val = n_val // self.batch_size
            patience = 10
            best_loss = float('inf')
            # for epoch in range(max_training_epoch):
            for epoch in range(max_training_epoch):
                for i in range(n_batch_train):
                    if self.nn_train_with == 'tagiv':
                        prediction_mu, _ = self.model.net(train_X[i*self.batch_size:(i+1)*self.batch_size])
                        prediction_mu = prediction_mu.reshape(self.batch_size, len(self.target_list)*2)

                        # Update model
                        out_updater = OutputUpdater(self.model.net.device)
                        out_updater.update_heteros(
                            output_states = self.model.net.output_z_buffer,
                            mu_obs = train_y[i*self.batch_size:(i+1)*self.batch_size].flatten(),
                            delta_states = self.model.net.input_delta_z_buffer,
                        )
                        self.model.net.backward()
                        self.model.net.step()
                    elif self.nn_train_with == 'backprop':
                        optimizer.zero_grad()
                        y_pred = self.model(train_X[i*self.batch_size:(i+1)*self.batch_size].float())
                        loss_train = loss_fn(y_pred, train_y[i*self.batch_size:(i+1)*self.batch_size].float())
                        loss_train.backward()
                        optimizer.step()

                loss_val = 0
                if self.nn_train_with == 'tagiv':
                    for j in range(n_batch_val):
                        val_pred_mu, _ = self.model.net(val_X[j*self.batch_size:(j+1)*self.batch_size])
                        val_pred_mu = val_pred_mu.reshape(self.batch_size, len(self.target_list)*2)
                        val_pred_y_mu = val_pred_mu[:, [0, 2, 4]]
                        val_y_batch = val_y[j*self.batch_size:(j+1)*self.batch_size]
                        # Compute the mse between val_pred_y_mu and val_y_batch
                        loss_val += ((val_pred_y_mu - val_y_batch)**2).mean()
                    loss_val /= n_batch_val
                elif self.nn_train_with == 'backprop':
                    y_pred = self.model(val_X.float())
                    loss_val = loss_fn(y_pred, val_y.float())

                print(f'Epoch {epoch}: {loss_val}')
                # Early stopping with patience 10
                if loss_val < best_loss:
                    best_loss = loss_val
                    patience = 10
                else:
                    patience -= 1
                    if patience == 0:
                        break

            n_batch_test = n_test // self.batch_size

            loss_test = 0
            if self.nn_train_with == 'tagiv':
                for j in range(n_batch_val):
                    test_pred_mu, test_pred_var = self.model.net(test_X[j*self.batch_size:(j+1)*self.batch_size])
                    test_pred_mu = test_pred_mu.reshape(self.batch_size, len(self.target_list)*2)
                    test_pred_y_mu = test_pred_mu[:, [0, 2, 4]]
                    test_pred_y_var = test_pred_mu[:, [1, 3, 5]]
                    test_y_batch = test_y[j*self.batch_size:(j+1)*self.batch_size]
                    # Compute the mse between test_pred_y_mu and test_y_batch
                    loss_test += ((test_pred_y_mu - test_y_batch)**2).mean()
                loss_test /= n_batch_test
            elif self.nn_train_with == 'backprop':
                y_pred = self.model(test_X.float())
                loss_test = loss_fn(y_pred, test_y.float())
                # difference = y_pred - test_y.float()
                # # Convert to numpy
                # difference = difference.detach().numpy()

            print(f'Test loss {loss_test}')

        # # Denormalize the prediction
        # y_pred = y_pred.detach().numpy()
        # y_pred_denorm = y_pred * self.std_target + self.mean_target
        # # y_pred_var_denorm = test_pred_y_var * self.std_target ** 2
        # y_test_denorm = test_y * self.std_target + self.mean_target
        # print(y_test_denorm.tolist()[:20])
        # print(y_pred_denorm.tolist()[:20])
        # print(np.sqrt(y_pred_var_denorm))

        if save_model_path is not None:
            if self.nn_train_with == 'tagiv':
                param_dict = self.model.net.state_dict()
                # Save dictionary to file
                with open(save_model_path, 'wb') as f:
                    pickle.dump(param_dict, f)
            elif self.nn_train_with == 'backprop':
                pass    # TODO

    def _save_lstm_input(self):
        self.lstm_history.append(copy.deepcopy(self.base_model.lstm_output_history))
        self.lstm_cell_states.append(self.base_model.lstm_net.get_lstm_states())
        pass


    def _erase_history(self, num_steps_to_erase):
        # Erase the last num_steps_to_erase steps of the lstm history
        remove_until_index = -(num_steps_to_erase)
        self.base_model.states.mu_prior = self.base_model.states.mu_prior[:remove_until_index]
        self.base_model.states.var_prior = self.base_model.states.var_prior[:remove_until_index]
        self.base_model.states.mu_posterior = self.base_model.states.mu_posterior[:remove_until_index]
        self.base_model.states.var_posterior = self.base_model.states.var_posterior[:remove_until_index]
        self.base_model.states.cov_states = self.base_model.states.cov_states[:remove_until_index]
        self.base_model.states.mu_smooth = self.base_model.states.mu_smooth[:remove_until_index]
        self.base_model.states.var_smooth = self.base_model.states.var_smooth[:remove_until_index]

        self.drift_model.states.mu_prior = self.drift_model.states.mu_prior[:remove_until_index]
        self.drift_model.states.var_prior = self.drift_model.states.var_prior[:remove_until_index]
        self.drift_model.states.mu_posterior = self.drift_model.states.mu_posterior[:remove_until_index]
        self.drift_model.states.var_posterior = self.drift_model.states.var_posterior[:remove_until_index]
        self.drift_model.states.cov_states = self.drift_model.states.cov_states[:remove_until_index]
        self.drift_model.states.mu_smooth = self.drift_model.states.mu_smooth[:remove_until_index]
        self.drift_model.states.var_smooth = self.drift_model.states.var_smooth[:remove_until_index]
        self.lstm_history = self.lstm_history[:remove_until_index]
        self.lstm_cell_states = self.lstm_cell_states[:remove_until_index]
        # self.p_anm_all = self.p_anm_all[:remove_until_index]
        # self.mu_itv_all = self.mu_itv_all[:remove_until_index]
        # self.std_itv_all = self.std_itv_all[:remove_until_index]
        self.mu_obs_preds = self.mu_obs_preds[:remove_until_index]
        self.std_obs_preds = self.std_obs_preds[:remove_until_index]
        self.mu_ar_preds = self.mu_ar_preds[:remove_until_index]
        self.std_ar_preds = self.std_ar_preds[:remove_until_index]

    def _retract_agent(self, time_step_back):
        self._erase_history(time_step_back)
        new_base_mu_states = self.base_model.states.mu_posterior[-1]
        new_base_var_states = self.base_model.states.var_posterior[-1]
        new_drift_mu_states = self.drift_model.states.mu_posterior[-1]
        new_drift_var_states = self.drift_model.states.var_posterior[-1]
        self.base_model.set_states(new_base_mu_states, new_base_var_states)
        self.drift_model.set_states(new_drift_mu_states, new_drift_var_states)
        self.base_model.lstm_output_history = self.lstm_history[-1]
        self.base_model.lstm_net.set_lstm_states(self.lstm_cell_states[-1])

        # pass

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
    
class NN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x