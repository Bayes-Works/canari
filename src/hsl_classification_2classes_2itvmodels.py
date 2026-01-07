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
from canari.common import likelihood
import pandas as pd
from tqdm import tqdm
from pytagi.nn import Linear, OutputUpdater, Sequential, ReLU, EvenExp, Remax
import pickle
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import stumpy
from src.convert_to_class import hierachical_softmax
from typing import List


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
        self.drift_model2.initialize_states_history()
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
        self.mean_target_lt_model, self.std_target_lt_model, self.mean_target_ll_model, self.std_target_ll_model = None, None, None, None
        self.mean_LTd_class, self.std_LTd_class, self.mean_LTd2_class, self.std_LTd2_class, self.mean_MP_class, self.std_MP_class = None, None, None, None, None, None
        self.pred_class_probs = []
        self.pred_class_probs_var = []
        self.data_loglikelihoods = []
        self.ll_itv_all, self.lt_itv_all = [], []
        self.y_std_scale = y_std_scale
        self._copy_initial_models()
        self.start_idx_mp = start_idx_mp
        self.m_mp = m_mp
        self.mp_all = []

    def _copy_initial_models(self):
        """
        Create copies of the base and generate models to avoid modifying the original models.
        """
        self.init_base_model = copy.deepcopy(self.base_model)
        self.init_drift_model = copy.deepcopy(self.drift_model)
        self.init_drift_model2 = copy.deepcopy(self.drift_model2)
        if "lstm" in self.base_model.states_name:
            self.init_base_model.lstm_net = self.base_model.lstm_net
            self.init_base_model.lstm_output_history = copy.deepcopy(self.base_model.lstm_output_history)
            self.init_base_model.lstm_net.set_lstm_states(copy.deepcopy(self.base_model.lstm_net.get_lstm_states()))
        self.init_base_model.initialize_states_history()
        self.init_drift_model.initialize_states_history()
        self.init_drift_model2.initialize_states_history()

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

        self.drift_model2 = Model(
            LocalTrend(
                mu_states=[0, 0],
                var_states=[baseline_process_error_std**2, baseline_process_error_std**2],
                std_error=1e-3,
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
            mu_ar_pred2, var_ar_pred2, mu_drift_states_prior2, _ = self.drift_model2.forward()
            _, _, mu_drift_states_posterior2, var_drift_states_posterior2 = self.drift_model2.backward(
                obs=self.base_model.mu_states_posterior[self.AR_index], 
                obs_var=self.base_model.var_states_posterior[self.AR_index, self.AR_index])
            self.drift_model2._save_states_history()
            self.drift_model2.set_states(mu_drift_states_posterior2, var_drift_states_posterior2)
            mu_ar_preds2.append(mu_ar_pred2)
            std_ar_preds2.append(var_ar_pred2**0.5)

            # Compute MP at the current time step
            mu_lstm = self.base_model.states.get_mean(states_type="posterior", states_name="lstm", standardization=True)
            std_lstm = self.base_model.states.get_std(states_type="posterior", states_name="lstm", standardization=True)
            mu_ar = self.base_model.states.get_mean(states_type="posterior", states_name="autoregression", standardization=True)
            mp_input = mu_lstm + mu_ar  # Combine LSTM and AR components for MP calculation
            if self.current_time_step >= self.start_idx_mp:
                T = np.array(mp_input).flatten().astype("float64")
                Q = T[self.current_time_step - self.m_mp:self.current_time_step]
                D = stumpy.mass(Q, T[:self.current_time_step - self.m_mp], normalize=False)
                min_idx = np.argmin(D)
                mp_value = D[min_idx]
                if mp_value == np.inf or np.isnan(mp_value):
                    mp_value = np.nan
            else:
                mp_value = 0.0  # No anomaly score before enough history
            self.mp_all.append(mp_value)

            if buffer_LTd:
                self.LTd_buffer.append(mu_drift_states_prior[1].item())
            
            # Dummy values for p_anm when it is not in detect mode
            self.p_anm_all.append(0)
            self.mu_itv_all.append([np.nan, np.nan, np.nan])
            self.std_itv_all.append([np.nan, np.nan, np.nan])
            self.pred_class_probs.append([0])
            self.pred_class_probs_var.append([0])
            self.data_loglikelihoods.append([None, None, None, None])
            self.ll_itv_all.append(0)
            self.lt_itv_all.append(0)

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
            anm_begin: Optional[int] = 52*7,
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

                # # Track what classifier learns
                LTd_mu_prior = np.array(self.drift_model.states.mu_prior)[:, 1].flatten()
                LTd2_mu_prior = np.array(self.drift_model2.states.mu_prior)[:, 1].flatten()
                # LTd_history = self._hidden_states_collector(i - 1, LTd_mu_prior)
                LTd_history = self._hidden_states_collector(self.current_time_step - 1, LTd_mu_prior, step_look_back=128)
                LTd2_history = self._hidden_states_collector(self.current_time_step - 1, LTd2_mu_prior, step_look_back=128)
                mp_history = self._hidden_states_collector(self.current_time_step - 1, self.mp_all, step_look_back=128)

                # Normalize the histories
                LTd_history = (LTd_history - self.mean_LTd_class) / self.std_LTd_class
                LTd2_history = (LTd2_history - self.mean_LTd2_class) / self.std_LTd2_class
                mp_history = (mp_history - self.mean_MP_class) / self.std_MP_class

                # input_history = torch.tensor(LTd_history.tolist()+mp_history.tolist())
                input_history = np.array(LTd_history.tolist()+LTd2_history.tolist()+mp_history.tolist())
                # input_history = np.repeat(input_history, self.batch_size, axis=0).flatten()
                input_history = input_history.astype(np.float32)
                m_pred_logits, v_pred_logits = self.model_class.forward(input_history)
                
                # Convert the logits to probabilities
                # pred_probs = torch.nn.functional.softmax(pred_logits, dim=0).detach().numpy()
                self.pred_class_probs.append(m_pred_logits[::2].tolist())
                self.pred_class_probs_var.append(m_pred_logits[1::2].tolist())
                

            if "lstm" in self.base_model.states_name:
                self._save_lstm_input()

            # Get interventions predicted by the model
            self.lt_itv_model.net.eval()
            self.ll_itv_model.net.eval()

            itv_input_history = np.array(LTd_history.tolist()+LTd2_history.tolist())
            # input_history = np.repeat(input_history, self.batch_size, axis=0).flatten()
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

            num_steps_retract_lt = max(int(itv_pred_lt_mu_denorm[1]), 1)
            # num_steps_retract_ll = max(int(itv_pred_ll_mu_denorm[1]), 1)

            # Apply intervention to estimate data likelihood
            # Get the true values of anomaly
            if p_a_I_Yt > self.detection_threshold:
                trend_itv = itv_pred_lt_mu_denorm[0]
                var_trend_itv = itv_pred_lt_var_denorm[0]
                level_itv = itv_pred_ll_mu_denorm[0]
                var_trend_itv = itv_pred_ll_var_denorm[0]

                # if anm_type == "LL":
                    # level_itv = (anm_magnitude-self.data_processor.scale_const_mean[self.data_processor.output_col]) / self.data_processor.scale_const_std[self.data_processor.output_col]
                    # num_steps_retract = self.current_time_step - anm_begin if self.current_time_step - anm_begin > 0 else 0
                    # trend_itv = level_itv / max(num_steps_retract, 1)

                    # var_level_itv = 0
                    # var_trend_itv = 0
                    
                # if anm_type == "LT":
                    # trend_itv = anm_magnitude / self.data_processor.scale_const_std[self.data_processor.output_col]
                    # num_steps_retract = self.current_time_step - anm_begin if self.current_time_step - anm_begin > 0 else 0
                    # level_itv = trend_itv * max(num_steps_retract, 1) / 2

                    # var_level_itv = 0
                    # var_trend_itv = 0

                # if first_time_trigger is False:
                #     trigger_time = self.current_time_step
                #     first_time_trigger = True

                # num_steps_retract = self.current_time_step - trigger_time + 1

                self.likelihoods_log_mask = []
                data_likelihoods_ll, itv_LL, _ = self._estimate_likelihoods_with_intervention(
                    ssm=self.base_model,
                    level_intervention = [level_itv, 0],
                    # level_intervention = [0, 1],
                    trend_intervention = [0, 0],
                    num_steps_retract = num_steps_retract_lt,
                    data = data,
                    make_mask=True
                )
                self.ll_itv_all.append(itv_LL)
                # gamma = 0.95
                gamma = 1
                decay_weights = np.array([gamma**i for i in range(len(data_likelihoods_ll)-1, -1, -1)])
                data_likelihoods_lt, _, itv_LT = self._estimate_likelihoods_with_intervention(
                    ssm=self.base_model,
                    level_intervention = [0, 0],
                    trend_intervention = [trend_itv, 0],
                    # trend_intervention = [0, 1/52/2],
                    num_steps_retract = num_steps_retract_lt,
                    data = data,
                    make_mask=False
                )
                self.lt_itv_all.append(itv_LT * num_steps_retract_lt)
                # self.lt_itv_all.append(itv_LT)
                # plt.show()

                gen_ar_phi = self.generate_model.components["autoregression 2"].phi
                gen_ar_sigma = self.generate_model.components["autoregression 2"].std_error

                stationary_ar_std = np.sqrt(gen_ar_sigma**2 / (1 - gen_ar_phi**2))

                # if abs(itv_LL) < 2 * stationary_ar_std and abs(itv_LT * num_steps_retract_lt) < 2 * stationary_ar_std:
                #     data_likelihoods_ll = []
                #     data_likelihoods_lt = []

                # Decay from the first value to the last value
                decay_weights_op = np.array([gamma**i for i in range(len(data_likelihoods_ll))])

                # # Take the logsum of each list data_likelihoods_ll and data_likelihoods_lt
                # log_likelihood_ll = np.sum(np.log(data_likelihoods_ll))
                # log_likelihood_lt = np.sum(np.log(data_likelihoods_lt))

                # Compute the average of data_likelihoods_ll and data_likelihoods_lt
                if len(data_likelihoods_ll) > 0 and len(data_likelihoods_lt) > 0:
                    log_likelihood_ll = np.sum(np.log(data_likelihoods_ll))
                    log_likelihood_lt = np.sum(np.log(data_likelihoods_lt))
                else:
                    log_likelihood_ll = 1
                    log_likelihood_lt = 1

                # # # Take the average of each list data_likelihoods_ll and data_likelihoods_lt
                # log_likelihood_ll = np.sum(decay_weights * data_likelihoods_ll)
                # log_likelihood_lt = np.sum(decay_weights * data_likelihoods_lt)
                # log_likelihood_ll_op = np.sum(decay_weights_op * data_likelihoods_ll)
                # log_likelihood_lt_op = np.sum(decay_weights_op * data_likelihoods_lt)
                log_likelihood_ll_op = None
                log_likelihood_lt_op = None

                # # Multivariate likelihood
                # log_likelihood_ll = data_likelihoods_ll
                # log_likelihood_lt = data_likelihoods_lt

                # Store the log-likelihoods
                self.data_loglikelihoods.append([log_likelihood_lt, log_likelihood_ll, log_likelihood_lt_op, log_likelihood_ll_op])
            else:
                self.data_loglikelihoods.append([None, None, None, None])
                self.ll_itv_all.append(0)
                self.lt_itv_all.append(0)

            # if apply_intervention:
            #     if rerun_kf is False:
            #         if p_a_I_Yt > self.detection_threshold:
            #             rerun_kf = True
            #             # To control that during the rerun from the past, the agent cannnot trigger again
            #             i_before_retract = copy.copy(i)
            #             # Retract agent
            #             step_back = max(int(itv_pred_lt_mu_denorm[2]), 1) if max(int(itv_pred_lt_mu_denorm[2]), 1) < i else i - 2
            #             self._retract_agent(time_step_back=step_back)
            #             i = i - step_back
            #             self.current_time_step = self.current_time_step - step_back

            #             # Apply intervention on base_model hidden states
            #             LL_index = self.base_model.states_name.index("level")
            #             LT_index = self.base_model.states_name.index("trend")
            #             # self.base_model.mu_states[LL_index] += itv_pred_lt_mu_denorm[1]
            #             self.base_model.mu_states[LT_index] += itv_pred_lt_mu_denorm[0]
            #             self.base_model.var_states[LL_index, LL_index] += itv_pred_lt_var_denorm[1]
            #             self.base_model.var_states[LT_index, LT_index] += itv_pred_lt_var_denorm[0]

            #             self.drift_model.mu_states[0] = 0
            #             self.drift_model.mu_states[1] = self.mu_LTd
            #             trigger = True

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
                obs=self.base_model.mu_states_posterior[self.AR_index], 
                obs_var=self.base_model.var_states_posterior[self.AR_index, self.AR_index])
            self.drift_model._save_states_history()
            self.drift_model.set_states(mu_drift_states_posterior, var_drift_states_posterior)
            self.mu_ar_preds.append(mu_ar_pred)
            self.std_ar_preds.append(var_ar_pred**0.5)

            mu_ar_pred2, var_ar_pred2, mu_drift_states_prior2, _ = self.drift_model2.forward()
            _, _, mu_drift_states_posterior2, var_drift_states_posterior2 = self.drift_model2.backward(
                obs=self.base_model.mu_states_posterior[self.AR_index], 
                obs_var=self.base_model.var_states_posterior[self.AR_index, self.AR_index])
            self.drift_model2._save_states_history()
            self.drift_model2.set_states(mu_drift_states_posterior2, var_drift_states_posterior2)

            # Compute MP at the current time step
            mu_lstm = self.base_model.states.get_mean(states_type="posterior", states_name="lstm", standardization=True)
            std_lstm = self.base_model.states.get_std(states_type="posterior", states_name="lstm", standardization=True)
            mu_ar = self.base_model.states.get_mean(states_type="posterior", states_name="autoregression", standardization=True)
            mp_input = mu_ar + mu_lstm
            if p_a_I_Yt > self.detection_threshold:
                anm_start_global = self.current_time_step - num_steps_retract_lt
            else:
                anm_start_global = np.inf
            if self.current_time_step >= self.start_idx_mp:
                T = np.array(mp_input).flatten().astype("float64")
                Q = T[self.current_time_step - self.m_mp:self.current_time_step]
                stationary_space_stop = self.current_time_step - self.m_mp if self.current_time_step - self.m_mp < anm_start_global else anm_start_global
                D = stumpy.mass(Q, T[:stationary_space_stop], normalize=False)
                min_idx = np.argmin(D)
                mp_value = D[min_idx]
                if mp_value == np.inf or np.isnan(mp_value):
                    mp_value = np.nan
            else:
                mp_value = 0.0  # No anomaly score before enough history
            self.mp_all.append(mp_value)

            self.current_time_step += 1
            i += 1

        return np.array(self.mu_obs_preds).flatten(), np.array(self.std_obs_preds).flatten(), np.array(self.mu_ar_preds).flatten(), np.array(self.std_ar_preds).flatten()

    def _estimate_likelihoods_with_intervention(self, ssm: Model, level_intervention: List[float], trend_intervention: List[float], num_steps_retract: int, data, make_mask=False):
        """
        Compute the likelihood of observation and hidden states given action
        """

        # Do the intervention again with the smoothed states
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
        num_steps_retract = min(num_steps_retract, current_preds_num - 1)
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
        LL_intervened_value = ssm_copy.mu_states[LL_index]
        LT_intervened_value = ssm_copy.mu_states[LT_index]
        y_likelihood_all = []
        for i in range(num_steps_retract):

            mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = ssm_copy.forward(data_all["x"][i])
            (_, _, mu_states_posterior, var_states_posterior) = ssm_copy.backward(data_all["y"][i])
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
            # ssm_copy.set_states(mu_states_prior, var_states_prior)

            mu_y_preds.append(mu_obs_pred)
            std_y_preds.append(var_obs_pred**0.5)

            # Regular likelihood
            y_likelihood = likelihood(mu_obs_pred, 
                                    np.sqrt(var_obs_pred), 
                                    data_all["y"][i])
            y_likelihood_all.append(y_likelihood.item())

        # # Plot retracted states after intervention
        # mu_level_plot = ssm_copy.states.get_mean(states_type="posterior", states_name="level", standardization=True)
        # std_level_plot = ssm_copy.states.get_std(states_type="posterior", states_name="level", standardization=True)
        # mu_trend_plot = ssm_copy.states.get_mean(states_type="posterior", states_name="trend", standardization=True)
        # std_trend_plot = ssm_copy.states.get_std(states_type="posterior", states_name="trend", standardization=True)
        # mu_lstm_plot = ssm_copy.states.get_mean(states_type="posterior", states_name="lstm", standardization=True)
        # std_lstm_plot = ssm_copy.states.get_std(states_type="posterior", states_name="lstm", standardization=True)
        # mu_ar_plot = ssm_copy.states.get_mean(states_type="posterior", states_name="autoregression", standardization=True)
        # std_ar_plot = ssm_copy.states.get_std(states_type="posterior", states_name="autoregression", standardization=True)
        # # Remove the first self.num_before_detect points to align with mu_y_preds
        # mu_level_plot = mu_level_plot[self.num_before_detect:]
        # std_level_plot = std_level_plot[self.num_before_detect:]
        # mu_trend_plot = mu_trend_plot[self.num_before_detect:]
        # std_trend_plot = std_trend_plot[self.num_before_detect:]
        # mu_lstm_plot = mu_lstm_plot[self.num_before_detect:]
        # std_lstm_plot = std_lstm_plot[self.num_before_detect:]
        # mu_ar_plot = mu_ar_plot[self.num_before_detect:]
        # std_ar_plot = std_ar_plot[self.num_before_detect:]

        # fig = plt.figure(figsize=(10, 5))
        # gs = gridspec.GridSpec(4, 1)
        # ax0 = plt.subplot(gs[0])
        # ax1 = plt.subplot(gs[1])
        # ax2 = plt.subplot(gs[2])
        # ax3 = plt.subplot(gs[3])
        # ax0.plot(range(len(mu_y_preds)), np.array(mu_y_preds).flatten(), label='After itv')
        # ax0.fill_between(range(len(mu_y_preds)),
        #                  np.array(mu_y_preds).flatten() - np.array(std_y_preds).flatten(),
        #                  np.array(mu_y_preds).flatten() + np.array(std_y_preds).flatten(),
        #                  color='gray', alpha=0.5)
        # ax0.plot(range(len(mu_y_preds)), np.array(original_mu_obs_preds).flatten(), label='No itv')
        # ax0.fill_between(range(len(original_mu_obs_preds)),
        #                  np.array(original_mu_obs_preds).flatten() - np.array(original_std_obs_preds).flatten(),
        #                  np.array(original_mu_obs_preds).flatten() + np.array(original_std_obs_preds).flatten(),
        #                  color='gray', alpha=0.5)
        # ax0.plot(range(len(data["y"])), data["y"], color='k')
        # ax0.axvline(x=len(mu_y_preds)-1, color='green', linestyle='--', label='Detection Point')
        # ax0.axvline(x=len(mu_y_preds)-num_steps_retract-1, color='red', linestyle='--', label='Intervention Point')
        # # Plot hidden states at index 0
        # ax0.plot(range(len(mu_level_plot)), np.array(mu_level_plot).flatten(),color='tab:blue')
        # ax0.fill_between(range(len(mu_level_plot)),
        #                  mu_level_plot.flatten() - std_level_plot.flatten(),
        #                  mu_level_plot.flatten() + std_level_plot.flatten(),
        #                  color='tab:blue', alpha=0.2)
        # ax0.set_title('Level State with and without Intervention')
        # ax0.legend(ncol=2, loc='upper left')
        # if level_intervention[0] > 0:
        #     ax0.set_title('LL itv')
        # if trend_intervention[0] > 0:
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
        # ax3.set_xlim(ax0.get_xlim())
        # ax3.set_ylabel('AR')
        # plt.show()

        if "lstm" in ssm.states_name:
            # Set the base_model back to the original state
            ssm.lstm_output_history = copy.deepcopy(output_history_temp)
            ssm.lstm_net.set_lstm_states(cell_states_temp)

        return y_likelihood_all, LL_intervened_value.item(), LT_intervened_value.item()
    
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
        samples = {'LTd_history': [], 'LTd2_history': [], 'MP_history': [], 'anm_type': [], 'itv_LT': [], 'itv_LL': [], 'anm_develop_time': [], 'p_anm': []}
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
            drift_model2_copy = copy.deepcopy(self.drift_model2)

            mu_obs_preds, std_obs_preds = [], []
            mu_ar_preds, std_ar_preds = [], []
            p_anm_one_syn_ts = []
            y_likelihood_a_one_ts, y_likelihood_na_one_ts = [], []
            x_likelihood_a_one_ts, x_likelihood_na_one_ts = [], []
            syn_mp_all = []
            base_model_copy.initialize_states_history()
            drift_model_copy.initialize_states_history()
            drift_model2_copy.initialize_states_history()

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
                    LTd_mu_prior2 = np.array(drift_model2_copy.states.mu_prior)[:, 1].flatten()
                    mu_LTd_history = self._hidden_states_collector(i - 1, LTd_mu_prior, step_look_back=128)
                    mu_LTd_history2 = self._hidden_states_collector(i - 1, LTd_mu_prior2, step_look_back=128)
                    mp_history = self._hidden_states_collector(i - 1, syn_mp_all, step_look_back=128)
                    samples['LTd_history'].append(mu_LTd_history.tolist())
                    samples['LTd2_history'].append(mu_LTd_history2.tolist())
                    samples['MP_history'].append(mp_history.tolist())
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
                mu_ar_pred2, var_ar_pred2, _, _ = drift_model2_copy.forward()
                _, _, mu_drift_states_posterior, var_drift_states_posterior = drift_model_copy.backward(
                    obs=base_model_copy.mu_states_posterior[self.AR_index], 
                    obs_var=base_model_copy.var_states_posterior[self.AR_index, self.AR_index])
                _, _, mu_drift_states_posterior2, var_drift_states_posterior2 = drift_model2_copy.backward(
                    obs=base_model_copy.mu_states_posterior[self.AR_index], 
                    obs_var=base_model_copy.var_states_posterior[self.AR_index, self.AR_index])
                drift_model_copy._save_states_history()
                drift_model_copy.set_states(mu_drift_states_posterior, var_drift_states_posterior)
                drift_model2_copy._save_states_history()
                drift_model2_copy.set_states(mu_drift_states_posterior2, var_drift_states_posterior2)
                mu_ar_preds.append(mu_ar_pred)
                std_ar_preds.append(var_ar_pred**0.5)

                # Compute MP at the current time step
                mu_lstm = base_model_copy.states.get_mean(states_type="posterior", states_name="lstm", standardization=True)
                std_lstm = base_model_copy.states.get_std(states_type="posterior", states_name="lstm", standardization=True)
                mu_ar = base_model_copy.states.get_mean(states_type="posterior", states_name="autoregression", standardization=True)
                mp_input = mu_lstm + mu_ar  # Combine LSTM and AR components for MP calculation
                if i >= self.start_idx_mp:
                    T = np.array(mp_input).flatten().astype("float64")
                    Q = T[i - self.m_mp:i]
                    stationary_space_stop = i - self.m_mp if i - self.m_mp < anm_begin_list[k] else anm_begin_list[k]
                    D = stumpy.mass(Q, T[:stationary_space_stop], normalize=False)
                    min_idx = np.argmin(D)
                    mp_value = D[min_idx]
                    if mp_value == np.inf or np.isnan(mp_value):
                        mp_value = np.nan
                else:
                    mp_value = 0.0  # No anomaly score before enough history
                syn_mp_all.append(mp_value)

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
            # ax7.plot(np.arange(len(syn_mp_all)), syn_mp_all, color='r')
            # ax7.set_ylabel('MP')
            # ax8.plot(p_anm_one_syn_ts)
            # ax8.axvline(x=anm_begin_list[k], color='r', linestyle='--')
            # ax8.set_ylim(-0.05, 1.05)
            # ax8.set_ylabel('p_anm')
            # plt.show()
        
        samples_df = pd.DataFrame(samples)
        samples_df.to_csv(save_to_path, index=False)
    
    def collect_synthetic_samples(self, num_time_series: int = 10, save_to_path: Optional[str] = 'data/hsl_tsad_training_samples/hsl_tsad_train_samples.csv'):
        # Collect samples from synthetic time series
        samples = {'LTd_history': [], 'itv_LT': [], 'itv_LL': [], 'anm_develop_time': [], 'p_anm': []}

        # Anomly feature range define
        ts_len = 52*6
        anm_mag_range = [-1/52, 1/52]       # LT anm mag
        anm_begin_range = [int(ts_len/4), int(ts_len*3/8)]

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
        generated_ts, time_covariate, anm_mag_list, anm_begin_list = gen_model_copy.generate_time_series(num_time_series=num_time_series, num_time_steps=ts_len, 
                                                                time_covariates=self.data_processor.time_covariates, 
                                                                time_covariate_info=time_covariate_info,
                                                                add_anomaly=True, anomaly_mag_range=anm_mag_range, 
                                                                anomaly_begin_range=anm_begin_range, sample_from_lstm_pred=False)
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
            drift_model2_copy = copy.deepcopy(self.drift_model2)

            mu_obs_preds, std_obs_preds = [], []
            mu_ar_preds, std_ar_preds = [], []
            p_anm_one_syn_ts = []
            y_likelihood_a_one_ts, y_likelihood_na_one_ts = [], []
            x_likelihood_a_one_ts, x_likelihood_na_one_ts = [], []
            base_model_copy.initialize_states_history()
            drift_model_copy.initialize_states_history()
            drift_model2_copy.initialize_states_history()

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
                if i > 65:
                    LTd_mu_prior = np.array(drift_model_copy.states.mu_prior)[:, 1].flatten()
                    mu_LTd_history = self._hidden_states_collector(i - 1, LTd_mu_prior)
                    samples['LTd_history'].append(mu_LTd_history.tolist())
                    LTd_mu_prior2 = np.array(drift_model2_copy.states.mu_prior)[:, 1].flatten()
                    mu_LTd_history2 = self._hidden_states_collector(i - 1, LTd_mu_prior2)
                    samples['LTd2_history'].append(mu_LTd_history2.tolist())
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
                    obs=base_model_copy.mu_states_posterior[self.AR_index], 
                    obs_var=base_model_copy.var_states_posterior[self.AR_index, self.AR_index])
                drift_model_copy._save_states_history()
                drift_model_copy.set_states(mu_drift_states_posterior, var_drift_states_posterior)
                mu_ar_preds.append(mu_ar_pred)
                std_ar_preds.append(var_ar_pred**0.5)

                _, _, _, _ = drift_model2_copy.forward()
                _, _, mu_drift_states_posterior2, var_drift_states_posterior2 = drift_model2_copy.backward(
                    obs=base_model_copy.mu_states_posterior[self.AR_index], 
                    obs_var=base_model_copy.var_states_posterior[self.AR_index, self.AR_index])
                drift_model2_copy._save_states_history()
                drift_model2_copy.set_states(mu_drift_states_posterior2, var_drift_states_posterior2)

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

    def _hidden_states_collector(self, current_step, hidden_states_all_step, step_look_back = 64):
        hidden_states_all_step_numpy = np.array(np.copy(hidden_states_all_step))
        look_back_steps_list = self._get_look_back_time_steps(current_step, step_look_back)
        hidden_states_collected = hidden_states_all_step_numpy[look_back_steps_list]
        return hidden_states_collected
    
    def soft_target_encode(self, target):
        samples_target = np.zeros((len(target), 1), dtype=np.float32)
        for i, c in enumerate(target):
            if c == 0:
                samples_target[i, 0] = 0
            else:
                if c == 1:
                    samples_target[i, 0] = 3
                elif c == 2:
                    samples_target[i, 0] = -3
        return samples_target

    def learn_classification(self, training_samples_path, save_model_path=None, load_model_path=None, max_training_epoch=10):
        samples = pd.read_csv(training_samples_path)
        samples['LTd_history'] = samples['LTd_history'].apply(lambda x: list(map(float, x[1:-1].split(','))))
        samples['LTd2_history'] = samples['LTd2_history'].apply(lambda x: list(map(float, x[1:-1].split(','))))
        samples['MP_history'] = samples['MP_history'].apply(lambda x: list(map(float, x[1:-1].split(','))))
        # Shuffle samples
        samples = samples.sample(frac=1).reset_index(drop=True)

        n_samples = len(samples)
        n_train = int(n_samples * 0.8)

        # Get the moments of samples['LTd_history'] and samples['MP_history'] for normalization
        train_LTd = np.array(samples['LTd_history'].values.tolist()[:n_train], dtype=np.float32)
        train_LTd2 = np.array(samples['LTd2_history'].values.tolist()[:n_train], dtype=np.float32)
        train_MP =  np.array(samples['MP_history'].values.tolist()[:n_train], dtype=np.float32)
        if self.mean_LTd_class is None or self.std_LTd_class is None or self.mean_LTd2_class is None or self.std_LTd2_class is None or self.mean_MP_class is None or self.std_MP_class is None:
            self.mean_LTd_class = np.mean(train_LTd)
            self.std_LTd_class = np.std(train_LTd)
            self.mean_LTd2_class = np.mean(train_LTd2)
            self.std_LTd2_class = np.std(train_LTd2)
            self.mean_MP_class = np.mean(train_MP)
            self.std_MP_class = np.std(train_MP)
        print('mean and std of training input', self.mean_LTd_class, self.std_LTd_class, self.mean_LTd2_class, self.std_LTd2_class, self.mean_MP_class, self.std_MP_class)
        # Normalize the two columns
        samples['LTd_history'] = samples['LTd_history'].apply(lambda x: [(val - self.mean_LTd_class) / self.std_LTd_class for val in x])
        samples['LTd2_history'] = samples['LTd2_history'].apply(lambda x: [(val - self.mean_LTd2_class) / self.std_LTd2_class for val in x])
        samples['MP_history'] = samples['MP_history'].apply(lambda x: [(val - self.mean_MP_class) / self.std_MP_class for val in x])
        # Combine the two columns for input feature
        samples['input_feature'] = samples.apply(lambda row: row['LTd_history'] + row['LTd2_history'] + row['MP_history'], axis=1)

        samples_input = np.array(samples['input_feature'].values.tolist(), dtype=np.float32)
        samples_target = np.array(samples['anm_type'].values, dtype=np.int64)
        samples_p_anm = np.array(samples['p_anm'].values.tolist(), dtype=np.float32)

        # Train the model using 80% of the samples
        train_X = samples_input[:n_train]
        train_y = samples_target[:n_train]

        # Validation set 10% of the samples
        n_val = int(n_samples * 0.1)
        val_X = samples_input[n_train:n_train+n_val]
        val_y = samples_target[n_train:n_train+n_val]

        # Test the model using 10% of the samples
        n_test = int(n_samples * 0.1)
        test_X = samples_input[n_train+n_val:n_train+n_val+n_test]
        test_y = samples_target[n_train+n_val:n_train+n_val+n_test]

        self.model_class = Sequential(
                                    Linear(len(samples['input_feature'][0]), 64, gain_weight=0.2, gain_bias=0.2),
                                    ReLU(),
                                    Linear(64, 32, gain_weight=0.2, gain_bias=0.2),
                                    ReLU(),
                                    Linear(32, 1*2, gain_weight=0.2, gain_bias=0.2),
                                    # Remax(),
                                    EvenExp(),
                                    )
        # Compute class weights to handle class imbalance
        # class_sample_counts = np.bincount(train_y)
        # class_weights = 1. / class_sample_counts
        # class_weights = torch.tensor(class_weights, dtype=torch.float32)
        # loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        # optimizer = torch.optim.Adam(self.model_class.parameters(), lr=0.001)
        train_X = torch.tensor(train_X)
        train_y = torch.tensor(train_y)
        val_X = torch.tensor(val_X)
        val_y = torch.tensor(val_y)
        test_X = torch.tensor(test_X)
        test_y = torch.tensor(test_y)

        self.batch_size = 20

        # Prepare dataloaders
        train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        if load_model_path is not None:
            with open(load_model_path, 'rb') as f:
                param_dict = pickle.load(f)
            self.model_class.load_state_dict(param_dict)
        else:
            # sigma_v = 0.1
            out_updater = OutputUpdater(self.model_class.device)
            # var_y = np.full((self.batch_size * 3,), sigma_v**2, dtype=np.float32)
            patience = 10
            best_loss = -float('inf')

            for epoch in range(max_training_epoch):
                self.model_class.train()
                train_likelihood = 0
                num_train_samples = 0

                all_m_pred = []
                all_v_pred = []

                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_training_epoch}")
                for batch_idx, (data, target) in enumerate(pbar):
                    # prepare data
                    x = data.numpy().flatten()
                    y = self.soft_target_encode(target).flatten()

                    m_pred, v_pred = self.model_class.forward(x)
                    all_m_pred.append(m_pred)
                    all_v_pred.append(v_pred)

                    # Update output layers
                    out_updater.update_heteros(
                        output_states=self.model_class.output_z_buffer,
                        mu_obs=y,
                        # var_obs=var_y,
                        delta_states=self.model_class.input_delta_z_buffer,
                    )
                    # Update parameters
                    self.model_class.backward()
                    self.model_class.step()

                    # Calculate error rate
                    # pred = m_pred[::2] - np.sqrt(m_pred[1::2])
                    pred = m_pred[::2]
                    s_pred = np.sqrt(m_pred[1::2])
                    pred = np.reshape(pred, (self.batch_size, 1))
                    s_pred = np.reshape(s_pred, (self.batch_size, 1))
                    m_probs = []
                    true_target = target.numpy()
                    num_no_anm = 0
                    for t in range(pred.shape[0]):
                        pr_classes = hierachical_softmax(pred[t], s_pred[t])
                        m_probs.append(pr_classes.tolist())
                        if true_target[t] == 0:
                            num_no_anm += 1
                        else:
                            train_likelihood += pr_classes[true_target[t]-1]
                    num_train_samples += len(target) - num_no_anm

                    # Update progress bar
                    pbar.set_postfix(
                        {"train_avg_likelihood": f"{train_likelihood/num_train_samples:.2f}"}
                    )

                average_m_pred = np.mean(np.concatenate(all_m_pred))
                average_v_pred = np.mean(np.concatenate(all_v_pred))
                std_v_m_pred = np.std(np.concatenate(all_m_pred))
                std_v_pred = np.std(np.concatenate(all_v_pred))

                average_v_pred_positive_m_pred = np.mean(
                    np.concatenate(all_v_pred)[np.concatenate(all_m_pred) > 0]
                )
                average_v_pred_negative_m_pred = np.mean(
                    np.concatenate(all_v_pred)[np.concatenate(all_m_pred) < 0]
                )
                print("Average mu prediction: ", average_m_pred)
                print("Average var prediction: ", average_v_pred)
                print("Std mu prediction: ", std_v_m_pred)
                print("Std var prediction: ", std_v_pred)
                print(
                    "Average var prediction (positive mu): ",
                    average_v_pred_positive_m_pred,
                )
                print(
                    "Average var prediction (negative mu): ",
                    average_v_pred_negative_m_pred,
                )

                # validation
                self.model_class.eval()
                val_likelihood = 0
                num_val_samples = 0

                for data, target in val_loader:
                    x = data.numpy().flatten()
                    m_pred, v_pred = self.model_class(x)
                    # m_pred = m_pred[::2]

                    # Calculate validation error
                    # pred = m_pred[::2] - np.sqrt(m_pred[1::2])
                    pred = m_pred[::2]
                    s_pred = np.sqrt(m_pred[1::2])
                    pred = np.reshape(pred, (self.batch_size, 1))
                    s_pred = np.reshape(s_pred, (self.batch_size, 1))
                    m_probs = []
                    true_target = target.numpy()
                    num_no_anm = 0
                    for t in range(pred.shape[0]):
                        pr_classes = hierachical_softmax(pred[t], s_pred[t])
                        m_probs.append(pr_classes.tolist())
                        if true_target[t] == 0:
                            num_no_anm += 1
                        else:
                            val_likelihood += pr_classes[true_target[t]-1]
                    num_val_samples += len(target) - num_no_anm

                print(
                    f"\nEpoch {epoch+1}/{max_training_epoch}: "
                    f"Train likelihood: {train_likelihood/num_train_samples:.2f} | "
                    f"Validation likelihood: {val_likelihood / num_val_samples:.2f}"
                )

                # Early stopping
                if val_likelihood > best_loss:
                    best_loss = val_likelihood
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print('Early stopping')
                        break
            
            # Testing
            self.model_class.eval()
            test_likelihood = 0
            num_test_samples = 0

            for data, target in test_loader:
                x = data.numpy().flatten()
                m_pred, v_pred = self.model_class(x)

                # Calculate test error
                pred = m_pred[::2]
                s_pred = np.sqrt(m_pred[1::2])
                pred = np.reshape(pred, (self.batch_size, 1))
                s_pred = np.reshape(s_pred, (self.batch_size, 1))
                m_probs = []
                true_target = target.numpy()
                num_no_anm = 0
                for t in range(pred.shape[0]):
                    pr_classes = hierachical_softmax(pred[t], s_pred[t])
                    m_probs.append(pr_classes.tolist())
                    if true_target[t] == 0:
                        num_no_anm += 1
                    else:
                        test_likelihood += pr_classes[true_target[t]-1]
                num_test_samples += len(target) - num_no_anm

            print(
                f"\nEpoch {epoch+1}/{max_training_epoch}: "
                f"Train likelihood: {train_likelihood/num_train_samples:.2f} | "
                f"Test likelihood: {test_likelihood/ num_test_samples:.2f}"
            )

            if save_model_path is not None:
                param_dict = self.model_class.state_dict()
                # Save dictionary to file
                with open(save_model_path, 'wb') as f:
                    pickle.dump(param_dict, f)

### ====================================================================================
        #     # Train the model with batch size 20
        #     n_batch_train = n_train // self.batch_size
        #     n_batch_val = n_val // self.batch_size
        #     patience = 10
        #     best_loss = float('inf')
        #     for epoch in range(max_training_epoch):
        #         for i in range(n_batch_train):
        #             batch_X = train_X[i*self.batch_size:(i+1)*self.batch_size]
        #             batch_y = train_y[i*self.batch_size:(i+1)*self.batch_size]
        #             optimizer.zero_grad()
        #             outputs = self.model_class(batch_X)
        #             loss = loss_fn(outputs, batch_y)
        #             loss.backward()
        #             optimizer.step()

        #         # Validate the model
        #         val_loss = 0.0
        #         with torch.no_grad():
        #             for i in range(n_batch_val):
        #                 batch_X = val_X[i*self.batch_size:(i+1)*self.batch_size]
        #                 batch_y = val_y[i*self.batch_size:(i+1)*self.batch_size]
        #                 outputs = self.model_class(batch_X)
        #                 loss = loss_fn(outputs, batch_y)
        #                 val_loss += loss.item()
        #         val_loss /= n_batch_val
        #         print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
        #         # Early stopping
        #         if val_loss < best_loss:
        #             best_loss = val_loss
        #             patience_counter = 0
        #         else:
        #             patience_counter += 1
        #             if patience_counter >= patience:
        #                 print('Early stopping')
        #                 break
            
        #     # Test the model
        #     test_loss = 0.0
        #     correct = 0
        #     total = 0
        #     n_batch_test = n_test // self.batch_size
        #     with torch.no_grad():
        #         for i in range(n_batch_test):
        #             batch_X = test_X[i*self.batch_size:(i+1)*self.batch_size]
        #             batch_y = test_y[i*self.batch_size:(i+1)*self.batch_size]
        #             outputs = self.model_class(batch_X)
        #             loss = loss_fn(outputs, batch_y)
        #             test_loss += loss.item()
        #             _, predicted = torch.max(outputs.data, 1)
        #             total += batch_y.size(0)
        #             correct += (predicted == batch_y).sum().item()
        #     test_loss /= n_batch_test
        #     accuracy = 100 * correct / total
        #     print(f'Test Loss: {test_loss}, Test Accuracy: {accuracy}%')

        # if save_model_path is not None:
        #     # Save the local pytorch model
        #     torch.save(self.model_class.state_dict(), save_model_path)

    def learn_intervention(self, training_samples_path, save_lt_model_path=None, save_ll_model_path=None, load_lt_model_path=None, load_ll_model_path=None, max_training_epoch=10):
        samples = pd.read_csv(training_samples_path)
        samples['LTd_history'] = samples['LTd_history'].apply(lambda x: list(map(float, x[1:-1].split(','))))
        samples['LTd2_history'] = samples['LTd2_history'].apply(lambda x: list(map(float, x[1:-1].split(','))))
        samples['MP_history'] = samples['MP_history'].apply(lambda x: list(map(float, x[1:-1].split(','))))
        # Convert samples['anm_develop_time'] to float
        samples['anm_develop_time'] = samples['anm_develop_time'].apply(lambda x: float(x))

        # Shuffle samples
        samples = samples.sample(frac=1).reset_index(drop=True)

        n_samples = len(samples)
        n_train = int(n_samples * 0.8)

        # Get the moments of samples['LTd_history'] and samples['MP_history'] for normalization
        train_LTd = np.array(samples['LTd_history'].values.tolist()[:n_train], dtype=np.float32)
        train_LTd2 = np.array(samples['LTd2_history'].values.tolist()[:n_train], dtype=np.float32)
        train_MP =  np.array(samples['MP_history'].values.tolist()[:n_train], dtype=np.float32)
        if self.mean_LTd_class is None or self.std_LTd_class is None or self.mean_LTd2_class is None or self.std_LTd2_class is None or self.mean_MP_class is None or self.std_MP_class is None:
            self.mean_LTd_class = np.mean(train_LTd)
            self.std_LTd_class = np.std(train_LTd)
            self.mean_LTd2_class = np.mean(train_LTd2)
            self.std_LTd2_class = np.std(train_LTd2)
            self.mean_MP_class = np.mean(train_MP)
            self.std_MP_class = np.std(train_MP)
        print('mean and std of training input', self.mean_LTd_class, self.std_LTd_class, self.mean_LTd2_class, self.std_LTd2_class, self.mean_MP_class, self.std_MP_class)
        # Normalize the two columns
        samples['LTd_history'] = samples['LTd_history'].apply(lambda x: [(val - self.mean_LTd_class) / self.std_LTd_class for val in x])
        samples['LTd2_history'] = samples['LTd2_history'].apply(lambda x: [(val - self.mean_LTd2_class) / self.std_LTd2_class for val in x])
        samples['MP_history'] = samples['MP_history'].apply(lambda x: [(val - self.mean_MP_class) / self.std_MP_class for val in x])
        # Combine the two columns for input feature
        samples['input_feature'] = samples.apply(lambda row: row['LTd_history'] + row['LTd2_history'], axis=1)

        # Remove the samples with samples['anm_type'] == 1
        samples_lt = samples[samples['anm_type'] != 2].reset_index(drop=True)
        samples_ll = samples[samples['anm_type'] != 1].reset_index(drop=True)

        # Target list
        self.target_lt_model_list = ['itv_LT', 'anm_develop_time']
        self.target_ll_model_list = ['itv_LL', 'anm_develop_time']

        samples_input_lt = np.array(samples_lt['input_feature'].values.tolist(), dtype=np.float32)
        samples_input_ll = np.array(samples_ll['input_feature'].values.tolist(), dtype=np.float32)
        samples_target_lt_model = np.array(samples_lt[self.target_lt_model_list].values, dtype=np.float32)
        samples_target_ll_model = np.array(samples_ll[self.target_ll_model_list].values, dtype=np.float32)
        samples_p_anm = np.array(samples_lt['p_anm'].values.tolist(), dtype=np.float32)

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
                        val_pred_y_mu = val_pred_mu[:, [0, 2]]
                        val_y_batch = val_y_lt_model[j*self.batch_size:(j+1)*self.batch_size]
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
                        val_pred_y_mu = val_pred_mu[:, [0, 2]]
                        val_y_batch = val_y_ll_model[j*self.batch_size:(j+1)*self.batch_size]
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

            

            if self.nn_train_with == 'tagiv':
                loss_test = 0
                n_batch_test = n_test_lt // self.batch_size
                for j in range(n_batch_test):
                    test_pred_mu, test_pred_var = self.lt_itv_model.net(test_X_lt[j*self.batch_size:(j+1)*self.batch_size])
                    test_pred_mu = test_pred_mu.reshape(self.batch_size, len(self.target_lt_model_list)*2)
                    test_pred_y_mu = test_pred_mu[:, [0, 2,]]
                    test_pred_y_var = test_pred_mu[:, [1, 3]]
                    test_y_batch = test_y_lt_model[j*self.batch_size:(j+1)*self.batch_size]
                    # Compute the mse between test_pred_y_mu and test_y_batch
                    loss_test += ((test_pred_y_mu - test_y_batch)**2).mean()
                loss_test_lt = loss_test/n_batch_test

                loss_test = 0
                n_batch_test = n_test_ll // self.batch_size
                for j in range(n_batch_test):
                    test_pred_mu, test_pred_var = self.ll_itv_model.net(test_X_ll[j*self.batch_size:(j+1)*self.batch_size])
                    test_pred_mu = test_pred_mu.reshape(self.batch_size, len(self.target_ll_model_list)*2)
                    test_pred_y_mu = test_pred_mu[:, [0, 2]]
                    test_pred_y_var = test_pred_mu[:, [1, 3]]
                    test_y_batch = test_y_ll_model[j*self.batch_size:(j+1)*self.batch_size]
                    # Compute the mse between test_pred_y_mu and test_y_batch
                    loss_test += ((test_pred_y_mu - test_y_batch)**2).mean()
                loss_test_ll = loss_test/n_batch_test

            print(f'Test loss of lt model: {loss_test_lt}')
            print(f'Test loss of ll model: {loss_test_ll}')

        # # Denormalize the prediction
        # y_pred = y_pred.detach().numpy()
        # y_pred_denorm = y_pred * self.std_target + self.mean_target
        # # y_pred_var_denorm = test_pred_y_var * self.std_target ** 2
        # y_test_denorm = test_y_lt_model * self.std_target + self.mean_target
        # print(y_test_denorm.tolist()[:20])
        # print(y_pred_denorm.tolist()[:20])
        # print(np.sqrt(y_pred_var_denorm))

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

        self.drift_model2.states.mu_prior = self.drift_model2.states.mu_prior[:remove_until_index]
        self.drift_model2.states.var_prior = self.drift_model2.states.var_prior[:remove_until_index]
        self.drift_model2.states.mu_posterior = self.drift_model2.states.mu_posterior[:remove_until_index]
        self.drift_model2.states.var_posterior = self.drift_model2.states.var_posterior[:remove_until_index]
        self.drift_model2.states.cov_states = self.drift_model2.states.cov_states[:remove_until_index]
        self.drift_model2.states.mu_smooth = self.drift_model2.states.mu_smooth[:remove_until_index]
        self.drift_model2.states.var_smooth = self.drift_model2.states.var_smooth[:remove_until_index]

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
        new_drift_mu_states2 = self.drift_model2.states.mu_posterior[-1]
        new_drift_var_states2 = self.drift_model2.states.var_posterior[-1]
        self.base_model.set_states(new_base_mu_states, new_base_var_states)
        self.drift_model.set_states(new_drift_mu_states, new_drift_var_states)
        self.drift_model2.set_states(new_drift_mu_states2, new_drift_var_states2)
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
    
class NN_Classification(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NN_Classification, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# class tagi_classification():
#     def __init__(self, input_size, output_size):
#         super(tagi_classification, self).__init__()
#         self.tagi_class_net = Sequential(
#                                     Linear(input_size, 64),
#                                     ReLU(),
#                                     Linear(64, 32),
#                                     ReLU(),
#                                     Linear(32, output_size),
#                                     Remax(),
#                                     )
#     def forward(self, mu_x, var_x):
#         return self.tagi_class_net.forward(mu_x, var_x)