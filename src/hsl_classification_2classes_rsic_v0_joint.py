from canari.component import LocalTrend, Autoregression
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
    common,
)
from pytagi import Normalizer as normalizer
from typing import Tuple, Dict, Optional, Callable
import numpy as np
import copy
from canari.common import likelihood, likelihood_laplace_approx
import pandas as pd
from tqdm import tqdm
from pytagi.nn import Linear, OutputUpdater, Sequential, ReLU, EvenExp, Remax, MixtureReLU
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import stumpy
from src.convert_to_class import hierachical_softmax
from typing import List
from scipy.stats import beta


class hsl_classification:
    """
    Anomaly detection based on hidden-states likelihood
    """

    def __init__(
            self, 
            base_model: Model,
            generate_model: Model,
            data_processor: DataProcess,
            drift_model_process_error_std: Optional[float] = 1e-5,
            y_std_scale: Optional[float] = 1.0,
            start_idx_mp: Optional[int] = 52*2+1,
            m_mp: Optional[int] = 52,
            ):
        self.base_model = base_model
        self.generate_model = generate_model
        self.data_processor = data_processor
        self._create_drift_model(drift_model_process_error_std)
        self.base_model.initialize_states_history()
        self.generate_model.initialize_states_history()
        self.drift_model.initialize_states_history()
        self.AR_index = base_model.states_name.index("autoregression")
        self.LTd_buffer = []
        self.p_anm_all = []
        self.mu_obs_preds, self.std_obs_preds = [], []
        self.prior_na, self.prior_a = 0.998, 0.002
        self.detection_threshold = 0.5
        self.mu_itv_all, self.std_itv_all = [], []
        self.nn_train_with = 'tagiv'
        self.current_time_step = 0
        self.lstm_history = []
        self.lstm_cell_states = []
        self.mean_target_lt_model, self.std_target_lt_model, self.mean_target_ll_model, self.std_target_ll_model = None, None, None, None
        self.mean_LTd_class, self.std_LTd_class = None, None
        self.class_prob_moments = []
        self.itv_decisions = []
        self.ll_itv_all, self.lt_itv_all = [], []
        self.y_std_scale = y_std_scale
        self._copy_initial_models()
        self.start_idx_mp = start_idx_mp
        self.posterior_mu_states_no_itv, self.posterior_var_states_no_itv = [], []

    def _copy_initial_models(self):
        """
        Create copies of the base and generate models to avoid modifying the original models.
        """
        self.init_base_model = copy.deepcopy(self.base_model)
        self.init_drift_model = copy.deepcopy(self.drift_model)
        if "lstm" in self.base_model.states_name:
            self.init_base_model.lstm_net = self.base_model.lstm_net
            self.init_base_model.lstm_output_history = copy.deepcopy(self.base_model.lstm_output_history)
            self.init_base_model.lstm_net.set_lstm_states(copy.deepcopy(self.base_model.lstm_net.get_lstm_states()))
        self.init_base_model.initialize_states_history()
        self.init_drift_model.initialize_states_history()

    def _create_drift_model(self, baseline_process_error_std):
        ar_component_key = [key for key in self.base_model.components.keys() if 'autoregression' in key][0]
        self.ar_component = self.base_model.components[ar_component_key]
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
        lstm_index_gen = self.generate_model.get_states_index("lstm")
        mu_obs_preds, std_obs_preds = [], []
        mu_ar_preds, std_ar_preds = [], []
        mu_ar_preds2, std_ar_preds2 = [], []
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

            #################################### Repeat with generate_model ###################################
            # Base model filter process, same as in model.py
            _,_,_,_ = self.generate_model.forward(x,
                                                    mu_lstm_pred=mu_lstm_pred,
                                                    var_lstm_pred=var_lstm_pred,)
            (
                _, _,
                mu_states_posterior_gen,
                var_states_posterior_gen,
            ) = self.generate_model.backward(y)

            if self.generate_model.lstm_net:
                self.generate_model.lstm_output_history.update(
                    mu_states_posterior_gen[lstm_index_gen],
                    var_states_posterior_gen[lstm_index_gen, lstm_index_gen],
                )

            self.generate_model._save_states_history()
            self.generate_model.set_states(mu_states_posterior, var_states_posterior)
            #########################################################################################################

            # Drift model filter process
            mu_ar_pred, var_ar_pred, mu_drift_states_prior, _ = self.drift_model.forward()
            _, _, mu_drift_states_posterior, var_drift_states_posterior = self.drift_model.backward(
                obs=self.base_model.mu_states_posterior[self.AR_index], 
                obs_var=self.base_model.var_states_posterior[self.AR_index, self.AR_index])
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
            self.class_prob_moments.append([0.5, 0.5, 0, 0])
            self.itv_decisions.append(0)
            self.ll_itv_all.append(0)
            self.lt_itv_all.append(0)
            self.posterior_mu_states_no_itv.append(np.zeros_like(self.base_model.states.mu_posterior[0])*np.nan)
            self.posterior_var_states_no_itv.append(np.zeros_like(self.base_model.states.mu_posterior[0])*np.nan)

            self.current_time_step += 1

        return np.array(mu_obs_preds).flatten(), np.array(std_obs_preds).flatten(), np.array(mu_ar_preds).flatten(), np.array(std_ar_preds).flatten()
    
    def estimate_LTd_dist(self):
        print('mean and std before roll out synthetic data', np.mean(self.LTd_buffer), np.std(self.LTd_buffer))
        # Roll out ten synthetic time series
        # # Generate synthetic time series
        covariate_col = self.data_processor.covariates_col
        train_index, val_index, test_index = self.data_processor.get_split_indices()
        time_covariate_info = {'initial_time_covariate': self.data_processor.data.values[val_index[-1], self.data_processor.covariates_col].item(),
                                'mu': self.data_processor.scale_const_mean[covariate_col], 
                                'std': self.data_processor.scale_const_std[covariate_col]}
        gen_model_copy = copy.deepcopy(self.generate_model)
        if "lstm" in self.generate_model.states_name:
            gen_model_copy.lstm_net = self.generate_model.lstm_net
            gen_model_copy.lstm_output_history = copy.deepcopy(self.generate_model.lstm_output_history)
            gen_model_copy.lstm_net.set_lstm_states(copy.deepcopy(self.generate_model.lstm_net.get_lstm_states()))
        generated_ts, time_covariate, _, _ = gen_model_copy.generate_time_series(num_time_series=10, num_time_steps=52*6, 
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
                    obs=base_model_copy.mu_states_posterior[self.AR_index], 
                    obs_var=base_model_copy.var_states_posterior[self.AR_index, self.AR_index])
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
            # # ax4.plot(states_drift_mu_prior[:, 0].flatten())
            # # ax4.fill_between(np.arange(len(states_drift_mu_prior[:, 0])),
            # #                 states_drift_mu_prior[:, 0].flatten() - states_drift_var_prior[:, 0, 0]**0.5,
            # #                 states_drift_mu_prior[:, 0].flatten() + states_drift_var_prior[:, 0, 0]**0.5,
            # #                 alpha=0.5)
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

    def tune_panm_threshold(self, data: pd.DataFrame):
        lstm_index = self.init_base_model.get_states_index("lstm")
        i = 0
        p_anm_to_tune = []
        while i < len(data["x"]):
            # Estimate likelihoods
            # Estimate likelihood without intervention
            y_likelihood_na, x_likelihood_na = self._estimate_likelihoods(base_model=self.init_base_model, drift_model=self.init_drift_model,
                                                                        obs=data["y"][i], input_covariates=data["x"][i], state_dist=self.LTd_pdf)
            # Estimate likelihood with intervention
            itv_base_model_prior, itv_drift_model_prior = self._intervene_current_priors(base_model=self.init_base_model, drift_model=self.init_drift_model)
            y_likelihood_a, x_likelihood_a = self._estimate_likelihoods(
                                                                        base_model=self.init_base_model, drift_model=self.init_drift_model,
                                                                        obs=data["y"][i], input_covariates=data["x"][i], state_dist=self.LTd_pdf,
                                                                        base_model_prior=itv_base_model_prior, drift_model_prior=itv_drift_model_prior
                                                                        )
            p_yt_I_Yt1 = y_likelihood_na * x_likelihood_na * self.prior_na + y_likelihood_a * x_likelihood_a * self.prior_a
            p_a_I_Yt = (y_likelihood_a * x_likelihood_a * self.prior_a / p_yt_I_Yt1).item()
            p_anm_to_tune.append(p_a_I_Yt)

            mu_obs_pred, var_obs_pred, _, _ = self.init_base_model.forward(data["x"][i])

            (
                _, _,
                mu_states_posterior,
                var_states_posterior,
            ) = self.init_base_model.backward(data["y"][i])

            if self.init_base_model.lstm_net:
                self.init_base_model.lstm_output_history.update(
                    mu_states_posterior[lstm_index],
                    var_states_posterior[lstm_index, lstm_index],
                )

            self.init_base_model._save_states_history()
            self.init_base_model.set_states(mu_states_posterior, var_states_posterior)

            # Drift model filter process
            mu_ar_pred, var_ar_pred, mu_drift_states_prior, _ = self.init_drift_model.forward()
            _, _, mu_drift_states_posterior, var_drift_states_posterior = self.init_drift_model.backward(
                obs=self.init_base_model.mu_states_posterior[self.AR_index], 
                obs_var=self.init_base_model.var_states_posterior[self.AR_index, self.AR_index])
            self.init_drift_model._save_states_history()
            self.init_drift_model.set_states(mu_drift_states_posterior, var_drift_states_posterior)

            i += 1

        self.detection_threshold = max(np.nanmax(p_anm_to_tune) * 1.1, 0.1)
        # self.detection_threshold = np.nanmax(p_anm_to_tune) * 1.1
        print(f"Detection threshold tuned to: {self.detection_threshold}")

        states_mu_prior = np.array(self.init_base_model.states.mu_posterior)
        states_var_prior = np.array(self.init_base_model.states.var_posterior)
        states_drift_mu_prior = np.array(self.init_drift_model.states.mu_posterior)
        states_drift_var_prior = np.array(self.init_drift_model.states.var_posterior)

        # #  Plot
        #  Plot states from pretrained model
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(8, 1)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        ax2 = plt.subplot(gs[2])
        ax3 = plt.subplot(gs[3])
        ax4 = plt.subplot(gs[4])
        ax5 = plt.subplot(gs[5])
        ax6 = plt.subplot(gs[6])
        ax7 = plt.subplot(gs[7])

        ax0.plot(states_mu_prior[:, 0].flatten(), label='local level')
        ax0.fill_between(np.arange(len(states_mu_prior[:, 0])),
                        states_mu_prior[:, 0].flatten() - states_var_prior[:, 0, 0]**0.5,
                        states_mu_prior[:, 0].flatten() + states_var_prior[:, 0, 0]**0.5,
                        alpha=0.5)
        ax0.plot(data["y"].flatten(), label='observed')

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
        
        ax4.plot(states_drift_mu_prior[:, 0].flatten(), label='LLd')
        ax4.fill_between(np.arange(len(states_drift_mu_prior[:, 0])),
                        states_drift_mu_prior[:, 0].flatten() - states_drift_var_prior[:, 0, 0]**0.5,
                        states_drift_mu_prior[:, 0].flatten() + states_drift_var_prior[:, 0, 0]**0.5,
                        alpha=0.5)
        
        ax5.plot(states_drift_mu_prior[:, 1].flatten(), label='LTd')
        ax5.fill_between(np.arange(len(states_drift_mu_prior[:, 1])),
                        states_drift_mu_prior[:, 1].flatten() - states_drift_var_prior[:, 1, 1]**0.5,
                        states_drift_mu_prior[:, 1].flatten() + states_drift_var_prior[:, 1, 1]**0.5,
                        alpha=0.5)
        
        ax6.plot(states_drift_mu_prior[:, 2].flatten(), label='ARd')
        ax6.fill_between(np.arange(len(states_drift_mu_prior[:, 2])),
                        states_drift_mu_prior[:, 2].flatten() - states_drift_var_prior[:, 2, 2]**0.5,
                        states_drift_mu_prior[:, 2].flatten() + states_drift_var_prior[:, 2, 2]**0.5,
                        alpha=0.5)
        
        ax7.plot(p_anm_to_tune, label='p_anm')
        ax7.axhline(y=self.detection_threshold, color='r', linestyle='--', label='detection threshold')
        ax7.set_ylim(0, 1)
        ax7.set_xlabel('Time step')


    def tune(self, decay_factor: Optional[float] = 0.9, begin_std_LTd: Optional[float] = 1):
        '''
        Tune the std_LTd using synthetic time series
        '''
        covariate_col = self.data_processor.covariates_col
        train_index, val_index, test_index = self.data_processor.get_split_indices()
        time_covariate_info = {'initial_time_covariate': self.data_processor.data.values[val_index[-1], self.data_processor.covariates_col].item(),
                                'mu': self.data_processor.scale_const_mean[covariate_col], 
                                'std': self.data_processor.scale_const_std[covariate_col]}
        gen_model_copy = copy.deepcopy(self.generate_model)
        if "lstm" in self.generate_model.states_name:
            gen_model_copy.lstm_net = self.generate_model.lstm_net
            gen_model_copy.lstm_output_history = copy.deepcopy(self.generate_model.lstm_output_history)
            gen_model_copy.lstm_net.set_lstm_states(copy.deepcopy(self.generate_model.lstm_net.get_lstm_states()))
        generated_ts, time_covariate, anm_mag_list, anm_begin_list = gen_model_copy.generate_time_series(num_time_series=10, num_time_steps=52*6, 
                                                                time_covariates=self.data_processor.time_covariates, 
                                                                time_covariate_info=time_covariate_info,
                                                                add_anomaly=False, sample_from_lstm_pred=False)
                                                                # add_anomaly=True, anomaly_mag_range=[-1/52, 1/52], 
                                                                # anomaly_begin_range=[int(52*10/4), int(52*10*3/8)], sample_from_lstm_pred=False)

        std_LTd_coeff = begin_std_LTd / decay_factor
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
                        obs=base_model_copy.mu_states_posterior[self.AR_index], 
                        obs_var=base_model_copy.var_states_posterior[self.AR_index, self.AR_index])
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
        print(f'LTd_std is tuned to coefficient of {std_LTd_coeff/decay_factor}; LTd_std is tuned to {self.LTd_std}.')

    def detect(
            self, 
            data,
            anm_type: Optional[str] = "LL",
            anm_magnitude: Optional[float] = 17,
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
        first_time_trigger = False

        self.num_before_detect = len(self.p_anm_all)
        itv_log = [] # 0: LL itv, 1: LT itv
        itv_applied_times = []
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

            if "lstm" in self.base_model.states_name:
                self._save_lstm_input()

            if first_time_trigger and self.current_time_step - trigger_time >= 52 * 5:
                first_time_trigger = False

            # Apply intervention to estimate data likelihood
            # if self.current_time_step > 52 *5:
            #     p_a_I_Yt = 0.95
            if p_a_I_Yt > self.detection_threshold and rerun_kf is False:
                # # Track what classifier learns
                LTd_mu_prior = np.array(self.drift_model.states.mu_prior)[:, 1].flatten()
                # LTd_history = self._hidden_states_collector(i - 1, LTd_mu_prior)
                LTd_history = self._hidden_states_collector(self.current_time_step - 1, LTd_mu_prior, step_look_back=128)

                # Normalize the histories
                LTd_history = (LTd_history - self.mean_LTd_class) / self.std_LTd_class
               
                # Get interventions predicted by the model
                self.lt_itv_model.net.eval()
                self.ll_itv_model.net.eval()

                itv_input_history = np.array(LTd_history.tolist())
                itv_input_history = itv_input_history.astype(np.float32)
                output_pred_lt_mu, output_pred_lt_var = self.lt_itv_model.net(itv_input_history)
                itv_pred_lt_mu = output_pred_lt_mu[::2]
                itv_pred_lt_var = output_pred_lt_mu[1::2] + output_pred_lt_var[::2]
                output_pred_ll_mu, output_pred_ll_var = self.ll_itv_model.net(itv_input_history)
                itv_pred_ll_mu = output_pred_ll_mu[::2]
                itv_pred_ll_var = output_pred_ll_mu[1::2] + output_pred_ll_var[::2]
                    
                itv_pred_lt_mu_denorm = itv_pred_lt_mu * self.std_target_lt_model + self.mean_target_lt_model
                itv_pred_lt_var_denorm = itv_pred_lt_var * self.std_target_lt_model ** 2
                itv_pred_ll_mu_denorm = itv_pred_ll_mu * self.std_target_ll_model + self.mean_target_ll_model
                itv_pred_ll_var_denorm = itv_pred_ll_var * self.std_target_ll_model ** 2

                trend_itv = itv_pred_lt_mu_denorm[0]
                llclt_itv = itv_pred_lt_mu_denorm[1]
                var_trend_itv = itv_pred_lt_var_denorm[0]
                var_llclt_itv = itv_pred_lt_var_denorm[1]
                level_itv = itv_pred_ll_mu_denorm[0]
                var_level_itv = itv_pred_ll_var_denorm[0]

                if first_time_trigger:
                    self.p_anm_all.append(0)
                else:
                    self.p_anm_all.append(p_a_I_Yt)
                
                if first_time_trigger is False:
                    trigger_time = self.current_time_step
                    first_time_trigger = True
                itvtime_from_det = self.current_time_step - trigger_time + 1
                
                # Get the trend intervention at the detection time
                # Option #1: Inverse transition of the intervention variables
                mu_lt_t = np.array([[llclt_itv], [trend_itv]])
                var_lt_t = np.array([[var_llclt_itv, 0], [0, var_trend_itv]])
                transition_matrix_itv = np.array([[1, 1], [0, 1]])
                itv_at_trigger, var_itv_at_trigger = reverse_lt_states(mu_lt_t, var_lt_t, transition_matrix_itv, itvtime_from_det)
                llclt_itv_at_trigger = itv_at_trigger[0, 0]
                var_llclt_itv_at_trigger = var_itv_at_trigger[0, 0]
                trend_itv_at_trigger = itv_at_trigger[1, 0]
                var_trend_itv_at_trigger = var_itv_at_trigger[1, 1]
                # # For verify the forward gives me the same states:
                # mu_lt_t, var_lt_t = transition_lt_states(llclt_itv_at_trigger, var_llclt_itv_at_trigger, transition_matrix_itv, itvtime_from_det)

                # # Option #2: Naive propagation of uncertainties
                # llclt_itv_at_trigger = llclt_itv - trend_itv * itvtime_from_det
                # var_llclt_itv_at_trigger = var_llclt_itv + var_trend_itv * itvtime_from_det**2

                # Intervention time step option to choose:
                num_steps_retract = itvtime_from_det

                self.likelihoods_log_mask = []
                data_likelihoods_ll, hs_likelihoods_ll, itv_LL, _, ll_itv_baseline = self._estimate_likelihoods_with_intervention(
                    ssm=self.base_model,
                    drift_model=self.drift_model,
                    level_intervention = [level_itv, var_level_itv],
                    trend_intervention = [0, 0],
                    num_steps_retract = num_steps_retract,
                    data = data,
                    make_mask=True
                )
                self.ll_itv_all.append(ll_itv_baseline[-1])
                gamma = 0.95
                # gamma = 1
                decay_weights = np.array([gamma**i for i in range(len(data_likelihoods_ll)-1, -1, -1)])
                decay_weights_op = np.array([gamma**i for i in range(len(data_likelihoods_ll))])
                data_likelihoods_lt, hs_likelihoods_lt, _, itv_LT, lt_itv_baseline = self._estimate_likelihoods_with_intervention(
                    ssm=self.base_model,
                    drift_model=self.drift_model,
                    level_intervention = [llclt_itv_at_trigger, var_llclt_itv_at_trigger],
                    trend_intervention = [trend_itv_at_trigger, var_trend_itv_at_trigger],
                    num_steps_retract = num_steps_retract,
                    data = data,
                    make_mask=False
                )
                self.lt_itv_all.append(lt_itv_baseline[-1])   


                gen_ar_phi = self.generate_model.components["autoregression 2"].phi
                gen_ar_sigma = self.generate_model.components["autoregression 2"].std_error
                ar_phi = self.base_model.components["autoregression 2"].phi
                ar_sigma = self.base_model.components["autoregression 2"].std_error

                # stationary_ar_std = np.sqrt(gen_ar_sigma**2 / (1 - gen_ar_phi**2))
                stationary_ar_std = np.sqrt(ar_sigma**2 / (1 - ar_phi**2))

                # Measure the impact of the intervention v.s. the AR
                # # Option 1: compare the variance of them
                # itv_baselines_std_n = np.std(ll_itv_baseline - lt_itv_baseline)
                # ratio_baseline_res = itv_baselines_std_n**2 / (itv_baselines_std_n**2 + stationary_ar_std**2)

                # # Option 2: drop likelihoods
                # ll_lt_diff = ll_itv_baseline - lt_itv_baseline
                # indices_insignificant = np.where(np.abs(ll_lt_diff) < stationary_ar_std)[0]
                # data_likelihoods_ll = [data_likelihoods_ll[j] for j in range(len(data_likelihoods_ll)) if j not in indices_insignificant]
                # data_likelihoods_lt = [data_likelihoods_lt[j] for j in range(len(data_likelihoods_lt)) if j not in indices_insignificant]

                # Compute the average of data_likelihoods_ll and data_likelihoods_lt
                if len(data_likelihoods_ll) > 0 and len(data_likelihoods_lt) > 0:
                    # # Joint likelihood
                    # log_likelihood_ll = np.sum(np.log(data_likelihoods_ll))
                    # log_likelihood_lt = np.sum(np.log(data_likelihoods_lt))

                    # # Compare var(ll_itv_baseline-lt_itv_baseline) with stationary_ar_std**2 to decide whether to keep the likelihood or not
                    # ll_lt_diff = ll_itv_baseline - lt_itv_baseline
                    # if np.var(ll_lt_diff) < stationary_ar_std**2:
                    #     log_likelihood_ll = 1
                    #     log_likelihood_lt = 1
                    # else:
                    # EBMS
                    data_ll_post_n = np.array(data_likelihoods_ll) / (np.array(data_likelihoods_ll) + np.array(data_likelihoods_lt))
                    data_lt_post_n = np.array(data_likelihoods_lt) / (np.array(data_likelihoods_ll) + np.array(data_likelihoods_lt))
                    # data_ll_post_sum = np.nansum(data_ll_post_n)
                    # data_lt_post_sum = np.nansum(data_lt_post_n)

                    hs_ll_post_n = np.array(hs_likelihoods_ll) / (np.array(hs_likelihoods_ll) + np.array(hs_likelihoods_lt))
                    hs_lt_post_n = np.array(hs_likelihoods_lt) / (np.array(hs_likelihoods_ll) + np.array(hs_likelihoods_lt))
                    # hs_ll_post_sum = np.nansum(hs_ll_post_n)
                    # hs_lt_post_sum = np.nansum(hs_lt_post_n)

                    joint_data_hs_ll_post_n = data_ll_post_n * hs_ll_post_n
                    joint_data_hs_lt_post_n = data_lt_post_n * hs_lt_post_n
                    joint_data_hs_ll_post_sum = np.nansum(joint_data_hs_ll_post_n)
                    joint_data_hs_lt_post_sum = np.nansum(joint_data_hs_lt_post_n)
                else:
                    # data_ll_post_sum = 1
                    # data_lt_post_sum = 1
                    # hs_ll_post_sum = 1
                    # hs_lt_post_sum = 1
                    joint_data_hs_ll_post_sum = 1
                    joint_data_hs_lt_post_sum = 1

                # data_llitv_prob_mean = data_ll_post_sum / (data_ll_post_sum + data_lt_post_sum)
                # data_ltitv_prob_mean = data_lt_post_sum / (data_ll_post_sum + data_lt_post_sum)
                # data_llitv_prob_std = np.sqrt(data_ll_post_sum * data_lt_post_sum / (data_ll_post_sum + data_lt_post_sum)**2/(data_ll_post_sum + data_lt_post_sum + 1))

                # hs_llitv_prob_mean = hs_ll_post_sum / (hs_ll_post_sum + hs_lt_post_sum)
                # hs_ltitv_prob_mean = hs_lt_post_sum / (hs_ll_post_sum + hs_lt_post_sum)
                # hs_llitv_prob_std = np.sqrt(hs_ll_post_sum * hs_lt_post_sum / (hs_ll_post_sum + hs_lt_post_sum)**2/(hs_ll_post_sum + hs_lt_post_sum + 1))

                # # Get mixture coefficient
                # data_prob_coeff = get_data_dist_coeff(len(data_likelihoods_ll))
                # hs_prob_coeff = 1 - data_prob_coeff
                # # Combine data and hidden states likelihoods
                # # llitv_prob_mean = data_llitv_prob_mean * data_prob_coeff + hs_llitv_prob_mean * hs_prob_coeff
                # # ltitv_prob_mean = data_ltitv_prob_mean * data_prob_coeff + hs_ltitv_prob_mean * hs_prob_coeff
                # # llitv_prob_std = np.sqrt(data_llitv_prob_std**2 * data_prob_coeff + hs_llitv_prob_std**2 * hs_prob_coeff 
                # #                          + data_prob_coeff * hs_prob_coeff * (data_llitv_prob_mean - hs_llitv_prob_mean)**2)
                
                # # llitv_prob_mean = hs_llitv_prob_mean
                # # ltitv_prob_mean = hs_ltitv_prob_mean
                # # llitv_prob_std = hs_llitv_prob_std

                # # llitv_prob_mean = data_llitv_prob_mean
                # # ltitv_prob_mean = data_ltitv_prob_mean
                # # llitv_prob_std = data_llitv_prob_std

                llitv_prob_mean = joint_data_hs_ll_post_sum/ (joint_data_hs_ll_post_sum + joint_data_hs_lt_post_sum)
                ltitv_prob_mean = joint_data_hs_lt_post_sum/ (joint_data_hs_ll_post_sum + joint_data_hs_lt_post_sum)
                llitv_prob_std = np.sqrt(joint_data_hs_ll_post_sum * joint_data_hs_lt_post_sum / (joint_data_hs_ll_post_sum + joint_data_hs_lt_post_sum)**2/(joint_data_hs_ll_post_sum + joint_data_hs_lt_post_sum + 1))


                # Store the log-likelihoods
                self.class_prob_moments.append([llitv_prob_mean, ltitv_prob_mean, llitv_prob_std, llitv_prob_std])
                itv_decision = choose_by_credible_interval(alpha=joint_data_hs_ll_post_sum, beta_=joint_data_hs_lt_post_sum)
                # itv_decision = choose_by_certainty(alpha=joint_data_hs_ll_post_sum, beta_=joint_data_hs_lt_post_sum)
                self.itv_decisions.append(itv_decision)
            elif rerun_kf is False:
                self.class_prob_moments.append([0.5, 0.5, 0, 0])
                self.ll_itv_all.append(0)
                self.lt_itv_all.append(0)
                self.p_anm_all.append(p_a_I_Yt)
                self.itv_decisions.append(0)

            # cond_ll = all(
            #     m[0] - m[1] > 4 * m[2]
            #     for m in self.class_prob_moments[-1:]
            # )

            # cond_lt = all(
            #     m[1] - m[0] > 4 * m[2] 
            #     for m in self.class_prob_moments[-1:]
            # )

            # cond_ll = True if self.itv_decisions[-1] == 1 else False
            # cond_lt = True if self.itv_decisions[-1] == 2 else False

            cond_ll = all([decision == 1 for decision in self.itv_decisions[-13:]])
            cond_lt = all([decision == 2 for decision in self.itv_decisions[-13:]])

            # cond_ll = False
            # cond_lt = False

            if cond_ll and rerun_kf is False:
                apply_intervention = True
                # ll_intervened_mu = itv_LL[0]
                ll_intervened_mu = level_itv
                ll_intervened_var = var_level_itv
                itv_log.append(0)
                itv_applied_times.append(trigger_time)
                print(f"LL intervention {itv_LL} is applied at time step {self.current_time_step}.")
            elif cond_lt and rerun_kf is False:
                apply_intervention = True
                # ll_intervened_mu = itv_LT[0]
                # lt_intervened_mu = itv_LT[1]
                ll_intervened_mu = llclt_itv_at_trigger
                lt_intervened_mu = trend_itv_at_trigger
                ll_intervened_var = var_llclt_itv_at_trigger
                lt_intervened_var = var_trend_itv_at_trigger
                itv_log.append(1)
                itv_applied_times.append(trigger_time)
                print(f"LT intervention {itv_LT} is applied at time step {self.current_time_step}.")

            if rerun_kf is False:
                self.posterior_mu_states_no_itv.append(np.zeros_like(self.base_model.states.mu_posterior[0])*np.nan)
                self.posterior_var_states_no_itv.append(np.zeros_like(self.base_model.states.mu_posterior[0])*np.nan)

            if apply_intervention:
                if rerun_kf is False:
                    rerun_kf = True
                    # To control that during the rerun from the past, the agent cannnot trigger again
                    i_before_retract = copy.copy(i)
                    # Retract agent
                    step_back = num_steps_retract
                    self._retract_agent(time_step_back=step_back)
                    i = i - step_back
                    self.current_time_step = self.current_time_step - step_back

                    # Apply intervention on base_model hidden states
                    LL_index = self.base_model.states_name.index("level")
                    LT_index = self.base_model.states_name.index("trend")
                    self.base_model.mu_states[LL_index] += ll_intervened_mu
                    self.base_model.var_states[LL_index, LL_index] += ll_intervened_var
                    if cond_lt:
                        self.base_model.mu_states[LT_index] += lt_intervened_mu
                        self.base_model.var_states[LT_index, LT_index] += lt_intervened_var

                    self.drift_model.mu_states[0] = 0
                    self.drift_model.mu_states[1] = self.mu_LTd

                    trigger = True
                    apply_intervention = False
                    first_time_trigger = False
            # else:
            #     if rerun_kf is False:
            #         self.posterior_mu_states_no_itv.append(np.zeros_like(self.base_model.states.mu_posterior[0])*np.nan)
            #     else:
            #         self.posterior_var_states_no_itv.append(np.zeros_like(self.base_model.states.mu_posterior[0])*np.nan)

            # if rerun_kf is False:
            #     self.posterior_mu_states_no_itv.append(np.zeros_like(self.base_model.states.mu_posterior[0])*np.nan)
            # else:
            #     self.posterior_var_states_no_itv.append(np.zeros_like(self.base_model.states.mu_posterior[0])*np.nan)


            # Base model filter process
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
                obs=self.base_model.mu_states_posterior[self.AR_index], 
                obs_var=self.base_model.var_states_posterior[self.AR_index, self.AR_index])
            self.drift_model._save_states_history()
            self.drift_model.set_states(mu_drift_states_posterior, var_drift_states_posterior)

            self.current_time_step += 1
            i += 1

        return np.array(self.mu_obs_preds).flatten(), np.array(self.std_obs_preds).flatten(), np.array(itv_log).flatten(), np.array(itv_applied_times).flatten()

    def _estimate_likelihoods_with_intervention(self, ssm: Model, drift_model: Model, level_intervention: List[float], trend_intervention: List[float], num_steps_retract: int, data, make_mask=False):
        """
        Compute the likelihood of observation and hidden states given action
        """
        ssm_copy = copy.deepcopy(ssm)
        data_all = copy.deepcopy(data)
        original_mu_obs_preds = copy.deepcopy(self.mu_obs_preds)
        original_std_obs_preds = copy.deepcopy(self.std_obs_preds)

        mu_y_preds = copy.deepcopy(self.mu_obs_preds)
        std_y_preds = copy.deepcopy(self.std_obs_preds)
        current_preds_num = len(mu_y_preds)
        if "lstm" in ssm.states_name:
            output_history_temp = copy.deepcopy(ssm.lstm_output_history)
            cell_states_temp = copy.deepcopy(ssm.lstm_net.get_lstm_states())
            ssm_copy.lstm_net = ssm.lstm_net
            ssm_copy.lstm_output_history = copy.deepcopy(output_history_temp)
            ssm_copy.lstm_net.set_lstm_states(cell_states_temp)
        
        # Retract SSM to the time of intervention
        # Constrain num_steps_retract to be no larger than current_preds_num
        num_steps_retract = min(num_steps_retract, current_preds_num)
        remove_until_index = -(num_steps_retract)
        ssm_copy.states.mu_prior = ssm_copy.states.mu_prior[:remove_until_index]
        ssm_copy.states.var_prior = ssm_copy.states.var_prior[:remove_until_index]
        ssm_copy.states.mu_posterior = ssm_copy.states.mu_posterior[:remove_until_index]
        ssm_copy.states.var_posterior = ssm_copy.states.var_posterior[:remove_until_index]
        ssm_copy.states.cov_states = ssm_copy.states.cov_states[:remove_until_index]
        ssm_copy.states.mu_smooth = ssm_copy.states.mu_smooth[:remove_until_index]
        ssm_copy.states.var_smooth = ssm_copy.states.var_smooth[:remove_until_index]
        mu_y_preds = mu_y_preds[:remove_until_index]
        std_y_preds = std_y_preds[:remove_until_index]
        lstm_history_copy = copy.deepcopy(self.lstm_history)
        lstm_cell_states_copy = copy.deepcopy(self.lstm_cell_states)
        if "lstm" in ssm.states_name:
            retracted_lstm_history = lstm_history_copy[:remove_until_index]
            retracted_lstm_cell_states = lstm_cell_states_copy[:remove_until_index]
        # Keep the data after intervention time
        data_all["y"] = data_all["y"][current_preds_num-num_steps_retract:]
        data_all["x"] = data_all["x"][current_preds_num-num_steps_retract:]

        ssm_copy.set_states(ssm_copy.states.mu_posterior[-1], ssm_copy.states.var_posterior[-1])
        ssm_copy.lstm_output_history = retracted_lstm_history[-1]
        ssm_copy.lstm_net.set_lstm_states(retracted_lstm_cell_states[-1])
        
        # Apply intervention
        LL_index = ssm_copy.states_name.index("level")
        LT_index = ssm_copy.states_name.index("trend")
        ssm_copy.mu_states[LL_index] += level_intervention[0]
        ssm_copy.mu_states[LT_index] += trend_intervention[0]
        ssm_copy.var_states[LL_index, LL_index] += level_intervention[1]
        ssm_copy.var_states[LT_index, LT_index] += trend_intervention[1]
        y_likelihood_all = []
        for i in range(num_steps_retract):

            mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = ssm_copy.forward(data_all["x"][i])
            _, _, mu_states_posterior, var_states_posterior = ssm_copy.backward(obs=data_all["y"][i])
            if "lstm" in ssm_copy.states_name:
                lstm_index = ssm_copy.get_states_index("lstm")
                ssm_copy.lstm_output_history.update(
                    mu_states_posterior[lstm_index],
                    var_states_posterior[lstm_index, lstm_index],
                    # mu_states_prior[lstm_index],
                    # var_states_prior[lstm_index, lstm_index],
                )
            ssm_copy._save_states_history()
            ssm_copy.set_states(mu_states_posterior, var_states_posterior)

            mu_y_preds.append(mu_obs_pred)
            std_y_preds.append(var_obs_pred**0.5)

            # Regular likelihood
            y_likelihood = likelihood(mu_obs_pred, 
                                    np.sqrt(var_obs_pred), 
                                    data_all["y"][i])
            y_likelihood_all.append(y_likelihood.item())

        # Option 1: use smoothed values as the deterministic intervention
        # Perform smoother
        ssm_copy.smoother()

        # Get the smoothed value at -(num_steps_retract)
        mu_LL_deterministic_itv = ssm_copy.states.mu_smooth[-num_steps_retract][LL_index]
        mu_LT_deterministic_itv = ssm_copy.states.mu_smooth[-num_steps_retract][LT_index]
        LLcLT_deterministic_itv = mu_LL_deterministic_itv

        # # Option 2: use filtered values as the deterministic intervention
        # mu_LL_deterministic_itv = mu_states_prior[LL_index]
        # mu_LT_deterministic_itv = mu_states_prior[LT_index]
        # LLcLT_deterministic_itv = mu_LL_deterministic_itv - mu_LT_deterministic_itv * num_steps_retract

        # Do the intervention again with the smoothed states
        ssm_copy = copy.deepcopy(ssm)
        drift_model_copy = copy.deepcopy(drift_model)
        data_all = copy.deepcopy(data)
        original_mu_obs_preds = copy.deepcopy(self.mu_obs_preds)
        original_std_obs_preds = copy.deepcopy(self.std_obs_preds)

        mu_y_preds = copy.deepcopy(self.mu_obs_preds)
        std_y_preds = copy.deepcopy(self.std_obs_preds)
        current_preds_num = len(mu_y_preds)
        if "lstm" in ssm.states_name:
            output_history_temp = copy.deepcopy(ssm.lstm_output_history)
            cell_states_temp = copy.deepcopy(ssm.lstm_net.get_lstm_states())
            ssm_copy.lstm_net = ssm.lstm_net
            ssm_copy.lstm_output_history = copy.deepcopy(output_history_temp)
            ssm_copy.lstm_net.set_lstm_states(cell_states_temp)
        
        # Retract SSM to the time of intervention
        # Constrain num_steps_retract to be no larger than current_preds_num
        num_steps_retract = min(num_steps_retract, current_preds_num)
        remove_until_index = -(num_steps_retract)
        ssm_copy.states.mu_prior = ssm_copy.states.mu_prior[:remove_until_index]
        ssm_copy.states.var_prior = ssm_copy.states.var_prior[:remove_until_index]
        ssm_copy.states.mu_posterior = ssm_copy.states.mu_posterior[:remove_until_index]
        ssm_copy.states.var_posterior = ssm_copy.states.var_posterior[:remove_until_index]
        ssm_copy.states.cov_states = ssm_copy.states.cov_states[:remove_until_index]
        ssm_copy.states.mu_smooth = ssm_copy.states.mu_smooth[:remove_until_index]
        ssm_copy.states.var_smooth = ssm_copy.states.var_smooth[:remove_until_index]
        drift_model_copy.states.mu_prior = drift_model_copy.states.mu_prior[:remove_until_index]
        drift_model_copy.states.var_prior = drift_model_copy.states.var_prior[:remove_until_index]
        drift_model_copy.states.mu_posterior = drift_model_copy.states.mu_posterior[:remove_until_index]
        drift_model_copy.states.var_posterior = drift_model_copy.states.var_posterior[:remove_until_index]
        drift_model_copy.states.cov_states = drift_model_copy.states.cov_states[:remove_until_index]
        drift_model_copy.states.mu_smooth = drift_model_copy.states.mu_smooth[:remove_until_index]
        drift_model_copy.states.var_smooth = drift_model_copy.states.var_smooth[:remove_until_index]
        mu_y_preds = mu_y_preds[:remove_until_index]
        std_y_preds = std_y_preds[:remove_until_index]
        lstm_history_copy = copy.deepcopy(self.lstm_history)
        lstm_cell_states_copy = copy.deepcopy(self.lstm_cell_states)
        if "lstm" in ssm.states_name:
            retracted_lstm_history = lstm_history_copy[:remove_until_index]
            retracted_lstm_cell_states = lstm_cell_states_copy[:remove_until_index]
        # Keep the data after intervention time
        data_all["y"] = data_all["y"][current_preds_num-num_steps_retract:]
        data_all["x"] = data_all["x"][current_preds_num-num_steps_retract:]

        ssm_copy.set_states(ssm_copy.states.mu_posterior[-1], ssm_copy.states.var_posterior[-1])
        drift_model_copy.set_states(drift_model_copy.states.mu_posterior[-1], drift_model_copy.states.var_posterior[-1])
        ssm_copy.lstm_output_history = retracted_lstm_history[-1]
        ssm_copy.lstm_net.set_lstm_states(retracted_lstm_cell_states[-1])
        
        # Apply intervention
        LL_index = ssm_copy.states_name.index("level")
        LT_index = ssm_copy.states_name.index("trend")
        if trend_intervention[0] != 0:
            ssm_copy.mu_states[LL_index] = LLcLT_deterministic_itv
            ssm_copy.mu_states[LT_index] = mu_LT_deterministic_itv
        else:
            ssm_copy.mu_states[LL_index] = mu_LL_deterministic_itv
        LL_intervened_value = [mu_LL_deterministic_itv]
        LT_intervened_value = [LLcLT_deterministic_itv, mu_LT_deterministic_itv]
        y_likelihood_all = []
        x_likelihood_all = []
        LL_track = [mu_LL_deterministic_itv.item()]
        for i in range(num_steps_retract):

            mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = ssm_copy.forward(data_all["x"][i])
            (_, _, mu_states_posterior, var_states_posterior) = ssm_copy.backward(data_all["y"][i])
            _, _, mu_d_states_prior, _ = drift_model_copy.forward()
            _, _, mu_drift_states_posterior, var_drift_states_posterior = drift_model_copy.backward(
                obs=ssm_copy.mu_states_posterior[self.AR_index], 
                obs_var=ssm_copy.var_states_posterior[self.AR_index, self.AR_index])
            x_likelihood = likelihood(self.mu_LTd, self.LTd_std,
                mu_drift_states_posterior[1].item())
            if "lstm" in ssm_copy.states_name:
                lstm_index = ssm_copy.get_states_index("lstm")
                ssm_copy.lstm_output_history.update(
                    mu_states_posterior[lstm_index],
                    var_states_posterior[lstm_index, lstm_index],
                    # mu_states_prior[lstm_index],
                    # var_states_prior[lstm_index, lstm_index],
                )
            ssm_copy._save_states_history()
            ssm_copy.set_states(mu_states_posterior, var_states_posterior)
            drift_model_copy._save_states_history()
            drift_model_copy.set_states(mu_drift_states_posterior, var_drift_states_posterior)
            LL_track.append(mu_states_prior[LL_index].item())
            # ssm_copy.set_states(mu_states_prior, var_states_prior)

            mu_y_preds.append(mu_obs_pred)
            std_y_preds.append(var_obs_pred**0.5)

            # Regular likelihood
            y_likelihood = likelihood(mu_obs_pred, 
                                    np.sqrt(var_obs_pred), 
                                    data_all["y"][i])
            
            # # Laplace approximated likelihood
            # y_likelihood = likelihood_laplace_approx(
            #                     mu_obs_pred, 
            #                     np.sqrt(var_obs_pred), 
            #                     data_all["y"][i])

            # # Clip likelihood at 3 standard deviations
            # y_ll_lb = likelihood(mu_obs_pred, 
            #                     np.sqrt(var_obs_pred), 
            #                     mu_obs_pred - 3 * np.sqrt(var_obs_pred))
            # y_likelihood = max(y_likelihood, y_ll_lb)

            # print(mu_drift_states_posterior[1].item(), self.mu_LTd, self.LTd_std)
            # print(data_all["y"][i], mu_obs_pred, np.sqrt(var_obs_pred))
            # print(y_likelihood.item(), x_likelihood.item())

            y_likelihood_all.append(y_likelihood.item())
            x_likelihood_all.append(x_likelihood.item())

        # # Plot retracted states after intervention
        # mu_level_plot = ssm_copy.states.get_mean(states_type="posterior", states_name="level", standardization=True)
        # std_level_plot = ssm_copy.states.get_std(states_type="posterior", states_name="level", standardization=True)
        # mu_trend_plot = ssm_copy.states.get_mean(states_type="posterior", states_name="trend", standardization=True)
        # std_trend_plot = ssm_copy.states.get_std(states_type="posterior", states_name="trend", standardization=True)
        # mu_lstm_plot = ssm_copy.states.get_mean(states_type="posterior", states_name="lstm", standardization=True)
        # std_lstm_plot = ssm_copy.states.get_std(states_type="posterior", states_name="lstm", standardization=True)
        # mu_ar_plot = ssm_copy.states.get_mean(states_type="posterior", states_name="autoregression", standardization=True)
        # std_ar_plot = ssm_copy.states.get_std(states_type="posterior", states_name="autoregression", standardization=True)
        # mu_arlevel_plot = drift_model_copy.states.get_mean(states_type="posterior", states_name="level", standardization=True)
        # mu_artrend_plot = drift_model_copy.states.get_mean(states_type="posterior", states_name="trend", standardization=True)
        # # Remove the first self.num_before_detect points to align with mu_y_preds
        # mu_level_plot = mu_level_plot[self.num_before_detect:]
        # std_level_plot = std_level_plot[self.num_before_detect:]
        # mu_trend_plot = mu_trend_plot[self.num_before_detect:]
        # std_trend_plot = std_trend_plot[self.num_before_detect:]
        # mu_lstm_plot = mu_lstm_plot[self.num_before_detect:]
        # std_lstm_plot = std_lstm_plot[self.num_before_detect:]
        # mu_ar_plot = mu_ar_plot[self.num_before_detect:]
        # std_ar_plot = std_ar_plot[self.num_before_detect:]
        # mu_arlevel_plot = mu_arlevel_plot[self.num_before_detect:]
        # mu_artrend_plot = mu_artrend_plot[self.num_before_detect:]

        # fig = plt.figure(figsize=(10, 5))
        # gs = gridspec.GridSpec(5, 1)
        # ax0 = plt.subplot(gs[0])
        # ax1 = plt.subplot(gs[1])
        # ax2 = plt.subplot(gs[2])
        # ax3 = plt.subplot(gs[3])
        # ax4 = plt.subplot(gs[4])
        # ax0.plot(range(len(mu_y_preds)), np.array(mu_y_preds).flatten(), label='After itv')
        # ax0.fill_between(range(len(mu_y_preds)),
        #                  np.array(mu_y_preds).flatten() - np.array(std_y_preds).flatten(),
        #                  np.array(mu_y_preds).flatten() + np.array(std_y_preds).flatten(),
        #                  color='gray', alpha=0.5)
        # ax0.plot(range(len(mu_level_plot)), np.array(mu_level_plot).flatten(),color='tab:blue')
        # ax0.fill_between(range(len(mu_level_plot)),
        #                  mu_level_plot.flatten() - std_level_plot.flatten(),
        #                  mu_level_plot.flatten() + std_level_plot.flatten(),
        #                  color='tab:blue', alpha=0.2)
        # # ax0.plot(range(len(mu_y_preds)), np.array(original_mu_obs_preds).flatten(), label='No itv')
        # # ax0.fill_between(range(len(original_mu_obs_preds)),
        # #                  np.array(original_mu_obs_preds).flatten() - np.array(original_std_obs_preds).flatten(),
        # #                  np.array(original_mu_obs_preds).flatten() + np.array(original_std_obs_preds).flatten(),
        # #                  color='gray', alpha=0.5)
        # ax0.plot(range(len(data["y"])), data["y"], color='k')
        # ax0.axvline(x=len(mu_y_preds)-1, color='green', linestyle='--', label='Detection Point')
        # ax0.axvline(x=len(mu_y_preds)-num_steps_retract-1, color='red', linestyle='--', label='Intervention Point')
        # # # Plot hidden states at index 0
        # # ax0.plot(range(len(mu_level_plot)), np.array(mu_level_plot).flatten(),color='tab:blue')
        # # ax0.fill_between(range(len(mu_level_plot)),
        # #                  mu_level_plot.flatten() - std_level_plot.flatten(),
        # #                  mu_level_plot.flatten() + std_level_plot.flatten(),
        # #                  color='tab:blue', alpha=0.2)
        # # ax0.set_title('Level State with and without Intervention')
        # # ax0.legend(ncol=2, loc='upper left')
        # if level_intervention[0] != 0:
        #     ax0.set_title('LL itv')
        # if trend_intervention[0] != 0:
        #     ax0.set_title('LT itv')
        # ax1.plot(range(len(mu_trend_plot)), np.array(mu_trend_plot).flatten(),color='tab:blue')
        # ax1.fill_between(range(len(mu_trend_plot)),
        #                  mu_trend_plot.flatten() - std_trend_plot.flatten(),
        #                  mu_trend_plot.flatten() + std_trend_plot.flatten(),
        #                  color='tab:blue', alpha=0.2)
        # ax1.set_xlim(ax0.get_xlim())
        # ax1.set_ylabel('LT')
        # ax2.plot(range(len(mu_lstm_plot)), np.array(mu_lstm_plot).flatten(),color='tab:blue')
        # ax2.fill_between(range(len(mu_lstm_plot)),
        #                  mu_lstm_plot.flatten() - std_lstm_plot.flatten(),
        #                  mu_lstm_plot.flatten() + std_lstm_plot.flatten(),
        #                  color='tab:blue', alpha=0.2)
        # ax2.set_xlim(ax0.get_xlim())
        # ax2.set_ylabel('LSTM')
        # ax3.plot(range(len(mu_ar_plot)), np.array(mu_ar_plot).flatten(),color='tab:blue')
        # ax3.fill_between(range(len(mu_ar_plot)),
        #                  mu_ar_plot.flatten() - std_ar_plot.flatten(),
        #                  mu_ar_plot.flatten() + std_ar_plot.flatten(),
        #                  color='tab:blue', alpha=0.2)
        # ax3.plot(range(len(mu_arlevel_plot)), np.array(mu_arlevel_plot).flatten(), color='tab:orange', label='AR Level Drift')
        # ax3.set_xlim(ax0.get_xlim())
        # ax3.set_ylabel('AR')
        # # if trend_intervention[0] == 0:
        # #     print('LL intervention applied.')
        # #     print(y_likelihood_all)
        # #     # print(np.prod(y_likelihood_all))
        # # if trend_intervention[0] != 0:
        # #     print('LT intervention applied.')
        # #     print(y_likelihood_all)
        # #     # print(np.prod(y_likelihood_all))

        # ax4.plot(range(len(mu_artrend_plot)), np.array(mu_artrend_plot).flatten(), color='tab:orange', label='AR LT Drift')
        # ax4.fill_between(range(len(mu_artrend_plot)), self.mu_LTd-self.LTd_std, self.mu_LTd+self.LTd_std, color='tab:orange', alpha=0.2)
        # ax4.set_xlim(ax0.get_xlim())
        # plt.show()

        if "lstm" in ssm.states_name:
            # Set the base_model back to the original state
            ssm.lstm_output_history = copy.deepcopy(output_history_temp)
            ssm.lstm_net.set_lstm_states(cell_states_temp)

        return y_likelihood_all, x_likelihood_all, LL_intervened_value, LT_intervened_value, np.array(LL_track).flatten()
    
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
        _, _, _, _ = base_model_copy.backward(obs)

        y_likelihood = likelihood(mu_obs_pred, np.sqrt(var_obs_pred) * self.y_std_scale, obs)

        _, _, mu_d_states_prior, _ = drift_model_copy.forward()
        _, _, mu_drift_states_posterior, _ = drift_model_copy.backward(
                obs=base_model_copy.mu_states_posterior[self.AR_index], 
                obs_var=base_model_copy.var_states_posterior[self.AR_index, self.AR_index])
        
        x_likelihood = state_dist(mu_drift_states_posterior[1].item())

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

        LL_index = base_model.states_name.index("level")
        LT_index = base_model.states_name.index("trend")
        # ar_index = base_model.states_name.index("autoregression")
        base_model_prior['mu'][LL_index] += drift_model_prior['mu'][0]
        base_model_prior['mu'][LT_index] += drift_model_prior['mu'][1]
        # base_model_prior['mu'][ar_index] = drift_model_prior['mu'][2]
        base_model_prior['var'][LL_index, LL_index] += drift_model_prior['var'][0, 0]
        base_model_prior['var'][LT_index, LT_index] += drift_model_prior['var'][1, 1]
        # base_model_prior['var'][ar_index, ar_index] = drift_model_prior['var'][2, 2]
        drift_model_prior['mu'][0] = 0
        drift_model_prior['mu'][1] = self.mu_LTd
        return base_model_prior, drift_model_prior
    
    def collect_anmtype_samples(self, num_time_series: int = 10, save_to_path: Optional[str] = 'data/hsl_tsad_training_samples/hsl_tsad_train_samples.csv'):
        # Collect samples from synthetic time series
        samples = {'LTd_history': [], 'anm_type': [], 'itv_LT': [], 'itv_LL': [], 'anm_develop_time': [], 'p_anm': []}
        # # # Anomaly type: no_anomaly = 0, LT = 1, LL = 2, PD = 3

        # Anomly feature range define
        ts_len = 52*8
        anm_type_log = []
        anm_mag_list = []
        anm_begin_list = []

        # # Generate synthetic time series
        covariate_col = self.data_processor.covariates_col
        train_index, val_index, test_index = self.data_processor.get_split_indices()
        time_covariate_info = {'initial_time_covariate': self.data_processor.data.values[val_index[-1], self.data_processor.covariates_col].item(),
                                'mu': self.data_processor.scale_const_mean[covariate_col], 
                                'std': self.data_processor.scale_const_std[covariate_col]}
        gen_model_copy = copy.deepcopy(self.generate_model)
        if "lstm" in self.generate_model.states_name:
            gen_model_copy.lstm_net = self.generate_model.lstm_net
            gen_model_copy.lstm_output_history = copy.deepcopy(self.generate_model.lstm_output_history)
            gen_model_copy.lstm_net.set_lstm_states(copy.deepcopy(self.generate_model.lstm_net.get_lstm_states()))
        generated_ts, time_covariate, _, _ = gen_model_copy.generate_time_series(num_time_series=num_time_series, num_time_steps=ts_len, 
                                                                time_covariates=self.data_processor.time_covariates, 
                                                                time_covariate_info=time_covariate_info,
                                                                add_anomaly=False, sample_from_lstm_pred=False) 
                                                                # anomaly_mag_range=anm_mag_range, 
                                                                # anomaly_begin_range=anm_begin_range, sample_from_lstm_pred=False
        # Apply anomalies to the generated time series
        for j in range(len(generated_ts)):
            anm_type = np.random.choice(['no_anm', 'LT', 'LL'])
            if anm_type == 'no_anm':
                anm_type_log.append(0)
                anm_mag_list.append(0)
                anm_begin_list.append(len(generated_ts[j]))
            elif anm_type == 'LT':
                anm_begin = np.random.randint(130, int(52*5))
                anm_mag = np.random.uniform(-1/52, 1/52)
                anm_baseline = np.arange(ts_len) * anm_mag
                anm_baseline[anm_begin:] -= anm_baseline[anm_begin]
                anm_baseline[:anm_begin] = 0
                generated_ts[j] += anm_baseline
                anm_type_log.append(1)
                anm_mag_list.append(anm_mag)
                anm_begin_list.append(anm_begin)
            elif anm_type == 'LL':
                anm_begin = np.random.randint(130, int(52*5))
                anm_mag = np.random.uniform(-2, 2)
                anm_baseline = np.ones(ts_len) * anm_mag
                anm_baseline[:anm_begin] = 0
                generated_ts[j] += anm_baseline
                anm_type_log.append(2)
                anm_mag_list.append(anm_mag)
                anm_begin_list.append(anm_begin)
        # Plot generated time series
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 1)
        ax0 = plt.subplot(gs[0])
        norm_data = self.data_processor.standardize_data()
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
                p_yt_I_Yt1 = np.maximum(y_likelihood_na * x_likelihood_na * self.prior_na, 1e-12) + y_likelihood_a * x_likelihood_a * self.prior_a
                p_a_I_Yt = (y_likelihood_a * x_likelihood_a * self.prior_a / p_yt_I_Yt1).item()
                p_anm_one_syn_ts.append(p_a_I_Yt)
                y_likelihood_a_one_ts.append(y_likelihood_a)
                y_likelihood_na_one_ts.append(y_likelihood_na)
                x_likelihood_a_one_ts.append(x_likelihood_a)
                x_likelihood_na_one_ts.append(x_likelihood_na)

                # Collect sample input
                if i > 129:
                    LTd_mu_prior = np.array(drift_model_copy.states.mu_prior)[:, 1].flatten()
                    mu_LTd_history = self._hidden_states_collector(i - 1, LTd_mu_prior, step_look_back=128)
                    samples['LTd_history'].append(mu_LTd_history.tolist())
                if i > 129 and i < anm_begin_list[k]:
                    samples['anm_type'].append(0)  # No anomaly
                    samples['p_anm'].append(0.)
                    samples['itv_LT'].append(0.)
                    samples['itv_LL'].append(0.)
                    samples['anm_develop_time'].append(0.)
                elif i >= anm_begin_list[k]:
                    samples['anm_type'].append(anm_type_log[k])
                    samples['p_anm'].append(p_a_I_Yt)
                    if anm_type_log[k] == 1:
                        samples['itv_LT'].append(anm_mag_list[k])
                        samples['itv_LL'].append(0.)
                    elif anm_type_log[k] == 2:
                        samples['itv_LT'].append(0.)
                        samples['itv_LL'].append(anm_mag_list[k])
                    elif anm_type_log[k] == 0:
                        samples['itv_LT'].append(0.)
                        samples['itv_LL'].append(0.)
                    samples['anm_develop_time'].append(i - anm_begin_list[k])

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
                    obs=base_model_copy.mu_states_posterior[self.AR_index], 
                    obs_var=base_model_copy.var_states_posterior[self.AR_index, self.AR_index])
                drift_model_copy._save_states_history()
                drift_model_copy.set_states(mu_drift_states_posterior, var_drift_states_posterior)
                mu_ar_preds.append(mu_ar_pred)
                std_ar_preds.append(var_ar_pred**0.5)

            # states_mu_posterior = np.array(base_model_copy.states.mu_posterior)
            # states_var_posterior = np.array(base_model_copy.states.var_posterior)
            # states_drift_mu_posterior = np.array(drift_model2_copy.states.mu_posterior)
            # states_drift_var_posterior = np.array(drift_model2_copy.states.var_posterior)

            # fig = plt.figure(figsize=(10, 9))
            # gs = gridspec.GridSpec(9, 1)
            # ax0 = plt.subplot(gs[0])
            # ax1 = plt.subplot(gs[1])
            # ax2 = plt.subplot(gs[2])
            # ax3 = plt.subplot(gs[3])
            # ax4 = plt.subplot(gs[4])
            # ax5 = plt.subplot(gs[5])
            # ax6 = plt.subplot(gs[6])
            # ax7 = plt.subplot(gs[7])
            # ax8 = plt.subplot(gs[8])
            # # ax9 = plt.subplot(gs[9])
            # # print(base_model_copy.states.mu_prior)
            # ax0.plot(states_mu_posterior[:, 0].flatten(), label='local level')
            # ax0.fill_between(np.arange(len(states_mu_posterior[:, 0])),
            #                 states_mu_posterior[:, 0].flatten() - states_var_posterior[:, 0, 0]**0.5,
            #                 states_mu_posterior[:, 0].flatten() + states_var_posterior[:, 0, 0]**0.5,
            #                 alpha=0.5)
            # ax0.axvline(x=anm_begin_list[k], color='r', linestyle='--')
            # ax0.plot(generated_ts[k])

            # ax1.plot(states_mu_posterior[:, 1].flatten(), label='local trend')
            # ax1.fill_between(np.arange(len(states_mu_posterior[:, 1])),
            #                 states_mu_posterior[:, 1].flatten() - states_var_posterior[:, 1, 1]**0.5,
            #                 states_mu_posterior[:, 1].flatten() + states_var_posterior[:, 1, 1]**0.5,
            #                 alpha=0.5)
            
            # ax2.plot(states_mu_posterior[:, 2].flatten(), label='lstm')
            # ax2.fill_between(np.arange(len(states_mu_posterior[:, 2])),
            #                 states_mu_posterior[:, 2].flatten() - states_var_posterior[:, 2, 2]**0.5,
            #                 states_mu_posterior[:, 2].flatten() + states_var_posterior[:, 2, 2]**0.5,
            #                 alpha=0.5)
            
            # ax3.plot(states_mu_posterior[:, 3].flatten(), label='autoregression')
            # ax3.fill_between(np.arange(len(states_mu_posterior[:, 3])),
            #                 states_mu_posterior[:, 3].flatten() - states_var_posterior[:, 3, 3]**0.5,
            #                 states_mu_posterior[:, 3].flatten() + states_var_posterior[:, 3, 3]**0.5,
            #                 alpha=0.5)
            # ax4.plot(np.array(mu_ar_preds).flatten(), label='obs')
            # ax4.fill_between(np.arange(len(mu_ar_preds)),
            #                 np.array(mu_ar_preds).flatten() - np.array(std_ar_preds).flatten(),
            #                 np.array(mu_ar_preds).flatten() + np.array(std_ar_preds).flatten(),
            #                 alpha=0.5)
            # ax4.plot(states_drift_mu_posterior[:, 0].flatten())
            # ax4.fill_between(np.arange(len(states_drift_mu_posterior[:, 0])),
            #                 states_drift_mu_posterior[:, 0].flatten() - states_drift_var_posterior[:, 0, 0]**0.5,
            #                 states_drift_mu_posterior[:, 0].flatten() + states_drift_var_posterior[:, 0, 0]**0.5,
            #                 alpha=0.5)
            # ax4.set_ylabel('LLd')
            # ax5.plot(states_drift_mu_posterior[:, 1].flatten(), color='r')
            # ax5.fill_between(np.arange(len(states_drift_mu_posterior[:, 1])),
            #                 states_drift_mu_posterior[:, 1].flatten() - states_drift_var_posterior[:, 1, 1]**0.5,
            #                 states_drift_mu_posterior[:, 1].flatten() + states_drift_var_posterior[:, 1, 1]**0.5,
            #                 alpha=0.2, color='r')
            # ax5.set_ylabel('LTd')
            # ax6.plot(states_drift_mu_posterior[:, 2].flatten())
            # ax6.fill_between(np.arange(len(states_drift_mu_posterior[:, 2])),
            #                 states_drift_mu_posterior[:, 2].flatten() - states_drift_var_posterior[:, 2, 2]**0.5,
            #                 states_drift_mu_posterior[:, 2].flatten() + states_drift_var_posterior[:, 2, 2]**0.5,
            #                 alpha=0.5)
            # ax6.set_ylabel('ARd')
            # ax8.plot(p_anm_one_syn_ts)
            # ax8.axvline(x=anm_begin_list[k], color='r', linestyle='--')
            # ax8.set_ylim(-0.05, 1.05)
            # ax8.set_ylabel('p_anm')
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

    def _hidden_states_collector(self, current_step, hidden_states_all_step, step_look_back = 64):
        hidden_states_all_step_numpy = np.array(np.copy(hidden_states_all_step))
        look_back_steps_list = self._get_look_back_time_steps(current_step, step_look_back)
        hidden_states_collected = hidden_states_all_step_numpy[look_back_steps_list]
        return hidden_states_collected

    def learn_intervention(self, training_samples_path, save_lt_model_path=None, save_ll_model_path=None, load_lt_model_path=None, load_ll_model_path=None, max_training_epoch=10):
        samples = pd.read_csv(training_samples_path)
        samples['LTd_history'] = samples['LTd_history'].apply(lambda x: list(map(float, x[1:-1].split(','))))
        # Convert samples['anm_develop_time'] to float
        samples['anm_develop_time'] = samples['anm_develop_time'].apply(lambda x: float(x))
        # Create a new column that times itv_LT by anm_develop_time
        samples['itv_LLcLT'] = samples.apply(lambda row: row['itv_LT'] * row['anm_develop_time'], axis=1)

        # Shuffle samples
        samples = samples.sample(frac=1).reset_index(drop=True)

        n_samples = len(samples)
        n_train = int(n_samples * 0.8)

        # Get the moments of samples['LTd_history'] for normalization
        train_LTd = np.array(samples['LTd_history'].values.tolist()[:n_train], dtype=np.float32)
        if self.mean_LTd_class is None or self.std_LTd_class is None:
            self.mean_LTd_class = np.nanmean(train_LTd)
            self.std_LTd_class = np.nanstd(train_LTd)
        print('mean and std of training input', self.mean_LTd_class, self.std_LTd_class)
        # Normalize the two columns
        samples['LTd_history'] = samples['LTd_history'].apply(lambda x: [(val - self.mean_LTd_class) / self.std_LTd_class for val in x])
        # Combine the two columns for input feature
        samples['input_feature'] = samples.apply(lambda row: row['LTd_history'], axis=1)

        # Remove the samples with samples['anm_type'] == 1
        samples_lt = samples[samples['anm_type'] != 2].reset_index(drop=True)
        samples_ll = samples[samples['anm_type'] != 1].reset_index(drop=True)

        # Target list
        self.target_lt_model_list = ['itv_LT', 'itv_LLcLT']
        self.target_ll_model_list = ['itv_LL']

        samples_input_lt = np.array(samples_lt['input_feature'].values.tolist(), dtype=np.float32)
        samples_input_ll = np.array(samples_ll['input_feature'].values.tolist(), dtype=np.float32)
        samples_target_lt_model = np.array(samples_lt[self.target_lt_model_list].values, dtype=np.float32)
        samples_target_ll_model = np.array(samples_ll[self.target_ll_model_list].values, dtype=np.float32)

        n_samples_lt = len(samples_lt)
        n_train_lt = int(n_samples_lt * 0.8)
        n_samples_ll = len(samples_ll)
        n_train_ll = int(n_samples_ll * 0.8)

        train_X_lt = samples_input_lt[:n_train_lt]
        train_X_ll = samples_input_ll[:n_train_ll]
        train_y_lt_model = samples_target_lt_model[:n_train_lt]
        train_y_ll_model = samples_target_ll_model[:n_train_ll]
        # Get the moments of training set, and use them to normalize the validation set and test set
        if self.mean_target_lt_model is None or self.std_target_lt_model is None:
            self.mean_target_lt_model = train_y_lt_model.mean(axis=0)
            self.std_target_lt_model = train_y_lt_model.std(axis=0)
        if self.mean_target_ll_model is None or self.std_target_ll_model is None:
            self.mean_target_ll_model = train_y_ll_model.mean(axis=0)
            self.std_target_ll_model = train_y_ll_model.std(axis=0)
        print('mean and std of training target (lt model)', self.mean_target_lt_model, self.std_target_lt_model)
        print('mean and std of training target (ll model)', self.mean_target_ll_model, self.std_target_ll_model)

        train_y_lt_model = (train_y_lt_model - self.mean_target_lt_model) / self.std_target_lt_model
        train_y_ll_model = (train_y_ll_model - self.mean_target_ll_model) / self.std_target_ll_model

        # Validation set 10% of the samples
        n_val_lt = int(n_samples_lt * 0.1)
        n_val_ll = int(n_samples_ll * 0.1)
        val_X_lt = samples_input_lt[n_train_lt:n_train_lt+n_val_lt]
        val_X_ll = samples_input_ll[n_train_ll:n_train_ll+n_val_ll]
        val_y_lt_model = samples_target_lt_model[n_train_lt:n_train_lt+n_val_lt]
        val_y_lt_model = (val_y_lt_model - self.mean_target_lt_model) / self.std_target_lt_model
        val_y_ll_model = samples_target_ll_model[n_train_ll:n_train_ll+n_val_ll]
        val_y_ll_model = (val_y_ll_model - self.mean_target_ll_model) / self.std_target_ll_model

        # Test the model using 10% of the samples
        n_test_lt = int(n_samples_lt * 0.1)
        n_test_ll = int(n_samples_ll * 0.1)
        test_X_lt = samples_input_lt[n_train_lt+n_val_lt:n_train_lt+n_val_lt+n_test_lt]
        test_X_ll = samples_input_ll[n_train_ll+n_val_ll:n_train_ll+n_val_ll+n_test_ll]
        test_y_lt_model = samples_target_lt_model[n_train_lt+n_val_lt:n_train_lt+n_val_lt+n_test_lt]
        test_y_lt_model = (test_y_lt_model - self.mean_target_lt_model) / self.std_target_lt_model
        test_y_ll_model = samples_target_ll_model[n_train_ll+n_val_ll:n_train_ll+n_val_ll+n_test_ll]
        test_y_ll_model = (test_y_ll_model - self.mean_target_ll_model) / self.std_target_ll_model

        if self.nn_train_with == 'tagiv':
            self.lt_itv_model = TAGI_Net(len(samples_lt['input_feature'][0]), len(self.target_lt_model_list))
            self.ll_itv_model = TAGI_Net(len(samples_ll['input_feature'][0]), len(self.target_ll_model_list))

        self.batch_size = 20

        if load_lt_model_path is not None:
            if self.nn_train_with == 'tagiv':
                with open(load_lt_model_path, 'rb') as f:
                    param_dict = pickle.load(f)
                self.lt_itv_model.net.load_state_dict(param_dict)
        else:
            n_batch_train = n_train_lt // self.batch_size
            n_batch_val = n_val_lt // self.batch_size
            patience = 10
            best_loss = float('inf')
            for epoch in range(max_training_epoch):
                for i in range(n_batch_train):
                    if self.nn_train_with == 'tagiv':
                        prediction_mu, _ = self.lt_itv_model.net(train_X_lt[i*self.batch_size:(i+1)*self.batch_size])
                        prediction_mu = prediction_mu.reshape(self.batch_size, len(self.target_lt_model_list)*2)

                        # Update model
                        out_updater = OutputUpdater(self.lt_itv_model.net.device)
                        out_updater.update_heteros(
                            output_states = self.lt_itv_model.net.output_z_buffer,
                            mu_obs = train_y_lt_model[i*self.batch_size:(i+1)*self.batch_size].flatten(),
                            delta_states = self.lt_itv_model.net.input_delta_z_buffer,
                        )
                        self.lt_itv_model.net.backward()
                        self.lt_itv_model.net.step()

                loss_val = 0
                if self.nn_train_with == 'tagiv':
                    for j in range(n_batch_val):
                        val_pred_mu, _ = self.lt_itv_model.net(val_X_lt[j*self.batch_size:(j+1)*self.batch_size])
                        val_pred_mu = val_pred_mu.reshape(self.batch_size, len(self.target_lt_model_list)*2)
                        val_pred_y_mu = val_pred_mu[:, ::2]
                        val_y_batch = val_y_lt_model[j*self.batch_size:(j+1)*self.batch_size]
                        # Compute the mse between val_pred_y_mu and val_y_batch
                        loss_val += ((val_pred_y_mu - val_y_batch)**2).mean()
                    loss_val /= n_batch_val

                loss_val = round(loss_val, 3)

                print(f'Epoch {epoch}: {loss_val}')
                # Early stopping with patience 10
                if loss_val < best_loss:
                    best_loss = loss_val
                    patience = 10
                else:
                    patience -= 1
                    if patience == 0:
                        break

            loss_test = 0
            n_batch_test = n_test_lt // self.batch_size
            for j in range(n_batch_test):
                test_pred_mu, test_pred_var = self.lt_itv_model.net(test_X_lt[j*self.batch_size:(j+1)*self.batch_size])
                test_pred_mu = test_pred_mu.reshape(self.batch_size, len(self.target_lt_model_list)*2)
                test_pred_y_mu = test_pred_mu[:, ::2]
                test_pred_y_var = test_pred_mu[:, 1::2]
                test_y_batch = test_y_lt_model[j*self.batch_size:(j+1)*self.batch_size]
                # Compute the mse between test_pred_y_mu and test_y_batch
                loss_test += ((test_pred_y_mu - test_y_batch)**2).mean()
            loss_test_lt = loss_test/n_batch_test
            loss_test_lt = round(loss_test_lt, 3)
            print(f'Test loss of lt model: {loss_test_lt}')

        # ====================================== Train LL intervention model ==============================================

        if load_ll_model_path is not None:
            if self.nn_train_with == 'tagiv':
                with open(load_ll_model_path, 'rb') as f:
                    param_dict = pickle.load(f)
                self.ll_itv_model.net.load_state_dict(param_dict)
        else:
            n_batch_train = n_train_ll // self.batch_size
            n_batch_val = n_val_ll // self.batch_size
            patience = 10
            best_loss = float('inf')
            for epoch in range(max_training_epoch):
                for i in range(n_batch_train):
                    if self.nn_train_with == 'tagiv':
                        prediction_mu, _ = self.ll_itv_model.net(train_X_ll[i*self.batch_size:(i+1)*self.batch_size])
                        prediction_mu = prediction_mu.reshape(self.batch_size, len(self.target_ll_model_list)*2)

                        # Update model
                        out_updater = OutputUpdater(self.ll_itv_model.net.device)
                        out_updater.update_heteros(
                            output_states = self.ll_itv_model.net.output_z_buffer,
                            mu_obs = train_y_ll_model[i*self.batch_size:(i+1)*self.batch_size].flatten(),
                            delta_states = self.ll_itv_model.net.input_delta_z_buffer,
                        )
                        self.ll_itv_model.net.backward()
                        self.ll_itv_model.net.step()

                loss_val = 0
                if self.nn_train_with == 'tagiv':
                    for j in range(n_batch_val):
                        val_pred_mu, _ = self.ll_itv_model.net(val_X_ll[j*self.batch_size:(j+1)*self.batch_size])
                        val_pred_mu = val_pred_mu.reshape(self.batch_size, len(self.target_ll_model_list)*2)
                        val_pred_y_mu = val_pred_mu[:, ::2]
                        val_y_batch = val_y_ll_model[j*self.batch_size:(j+1)*self.batch_size]
                        # Compute the mse between val_pred_y_mu and val_y_batch
                        loss_val += ((val_pred_y_mu - val_y_batch)**2).mean()
                    loss_val /= n_batch_val

                loss_val = round(loss_val, 3)

                print(f'Epoch {epoch}: {loss_val}')
                # Early stopping with patience 10
                if loss_val < best_loss:
                    best_loss = loss_val
                    patience = 10
                else:
                    patience -= 1
                    if patience == 0:
                        break

            loss_test = 0
            n_batch_test = n_test_ll // self.batch_size
            for j in range(n_batch_test):
                test_pred_mu, test_pred_var = self.ll_itv_model.net(test_X_ll[j*self.batch_size:(j+1)*self.batch_size])
                test_pred_mu = test_pred_mu.reshape(self.batch_size, len(self.target_ll_model_list)*2)
                test_pred_y_mu = test_pred_mu[:, ::2]
                test_pred_y_var = test_pred_mu[:, 1::2]
                test_y_batch = test_y_ll_model[j*self.batch_size:(j+1)*self.batch_size]
                # Compute the mse between test_pred_y_mu and test_y_batch
                loss_test += ((test_pred_y_mu - test_y_batch)**2).mean()
            loss_test_ll = loss_test/n_batch_test
            loss_test_ll = round(loss_test_ll, 3)
            print(f'Test loss of ll model: {loss_test_ll}')

        if save_lt_model_path is not None:
            if self.nn_train_with == 'tagiv':
                param_dict = self.lt_itv_model.net.state_dict()
                # Save dictionary to file
                with open(save_lt_model_path, 'wb') as f:
                    pickle.dump(param_dict, f)
        if save_ll_model_path is not None:
            if self.nn_train_with == 'tagiv':
                param_dict = self.ll_itv_model.net.state_dict()
                # Save dictionary to file
                with open(save_ll_model_path, 'wb') as f:
                    pickle.dump(param_dict, f)

    def _save_lstm_input(self):
        self.lstm_history.append(copy.deepcopy(self.base_model.lstm_output_history))
        self.lstm_cell_states.append(self.base_model.lstm_net.get_lstm_states())
        pass


    def _erase_history(self, num_steps_to_erase):
        # Erase the last num_steps_to_erase steps of the lstm history
        remove_until_index = -(num_steps_to_erase)

        # Keep the states that are removed
        self.posterior_mu_states_no_itv[remove_until_index:]  = self.base_model.states.mu_posterior[remove_until_index:]
        self.posterior_var_states_no_itv[remove_until_index:] = self.base_model.states.var_posterior[remove_until_index:]

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
        self.mu_obs_preds = self.mu_obs_preds[:remove_until_index]
        self.std_obs_preds = self.std_obs_preds[:remove_until_index]

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
                    MixtureReLU(),
                    Linear(64, 32),
                    # Linear(64, 32, gain_weight=0.1, gain_bias=0.1),
                    MixtureReLU(),
                    Linear(32, n_actions * 2),
                    # Linear(32, n_actions * 2, gain_weight=0.1, gain_bias=0.1),
                    EvenExp()
                    )
        self.n_actions = n_actions
        self.n_observations = n_observations
    def forward(self, mu_x, var_x):
        return self.net.forward(mu_x, var_x)
    
def get_data_dist_coeff(n, target=0, n_converge=52):
    """
    f(1) = 1
    f(n) → target
    Reaches ~target around n ≈ n_converge
    """
    alpha = np.log(2) / (n_converge - 1)  # ensures f(52) ≈ 0.5
    return target + (1 - target) * np.exp(-alpha * (n - 1))
    

def choose_by_credible_interval(alpha, beta_, level=0.95, threshold=0.5):
    lo = beta.ppf((1-level)/2, alpha, beta_)
    hi = beta.ppf(1-(1-level)/2, alpha, beta_)
    if lo > threshold:
        return 1
    if hi < threshold:
        return 2
    return 0

def choose_by_certainty(alpha, beta_, tau=0.95, threshold=0.5):
    p1_wins = beta.sf(threshold, alpha, beta_)  # P(p1 > threshold)
    if p1_wins > tau:
        return 1
    if (1 - p1_wins) > tau:  # P(p1 < threshold)
        return 2
    return 0  # abstain

def reverse_lt_states(mu_t, var_t, transition_matrix, steps_back):
    """
    Reverse the local trend states back by steps_back steps, to get the local level states at that time step.
    """
    import sympy as sp
    # Use sympy inverse for higher accuracy
    transition_matrix_steps_back = np.linalg.matrix_power(transition_matrix, steps_back)

    mu_t_sym = sp.Matrix(mu_t)
    var_t_sym = sp.Matrix(var_t)
    transition_matrix_sym = sp.Matrix(transition_matrix_steps_back)

    reversed_mu = transition_matrix_sym.inv() * mu_t_sym
    reversed_var = transition_matrix_sym.inv() * var_t_sym * transition_matrix_sym.inv().T

    # Convert back to numpy arrays
    reversed_mu = np.array(reversed_mu).astype(np.float64)
    reversed_var = np.array(reversed_var).astype(np.float64)
    return reversed_mu, reversed_var

def transition_lt_states(mu_t, var_t, transition_matrix, steps_forward):
    """
    Reverse the local trend states back by steps_back steps, to get the local level states at that time step.
    """
    transition_matrix_steps_forward = np.linalg.matrix_power(transition_matrix, steps_forward)
    reversed_mu = transition_matrix_steps_forward @ mu_t
    reversed_var = transition_matrix_steps_forward @ var_t @ transition_matrix_steps_forward.T
    return reversed_mu, reversed_var