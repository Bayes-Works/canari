import copy
import pandas as pd
from pytagi import Normalizer as normalizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytagi.metric as metric
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_data,
    plot_prediction,
    plot_skf_states,
)
import pickle
from canari.component import LocalTrend, LocalAcceleration,Autoregression, LstmNetwork, WhiteNoise

# # Read data
data_file = "./data/toy_time_series/synthetic_simple_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/synthetic_simple_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Add synthetic anomaly to data
anm_start_index = 52*10
anm_mag = 0.3/52
anm_baseline = np.arange(len(df_raw)) * anm_mag
# Set the first 52*12 values in anm_baseline to be 0
anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
anm_baseline[:anm_start_index] = 0
df_raw = df_raw.add(anm_baseline, axis=0)

# Data pre-processing
output_col = [0]
train_split=0.289
validation_split=0.0693*2
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=train_split,
    validation_split=validation_split,
    output_col=output_col,
)
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
with open("saved_params/toy_simple_model_rebased.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=19,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

print("phi_AR =", model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item())
print("sigma_AR =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()))


norm_model = Model(
    LocalTrend(mu_states=model_dict['early_stop_init_mu_states'][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(), 
                   mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()], 
                   var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
)
norm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

abnorm_model = Model(
    LocalAcceleration(mu_states=[model_dict['early_stop_init_mu_states'][0].item(), model_dict['early_stop_init_mu_states'][1].item(), 0], var_states=[1e-12, 1e-12, 1e-4]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()),
                   phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(),
                   mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()],
                   var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
)
abnorm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

skf = SKF(
    norm_model=norm_model,
    abnorm_model=abnorm_model,
    std_transition_error=0.0008005024870053932,
    norm_to_abnorm_prob=1.6648750624135252e-06,
    abnorm_to_norm_prob=1e-1,
    norm_model_prior_prob=0.99,
)

skf.filter_marginal_prob_history = skf._prob_history()
skf._set_same_states_transition_models()
skf.initialize_states_history()

filter_marginal_abnorm_prob, states = skf.filter(data=normalized_data)
smooth_marginal_abnorm_prob, states = skf.smoother()


fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=states,
    states_type="smooth",
    states_to_plot=["level", "trend", "lstm", "autoregression"],
    model_prob=smooth_marginal_abnorm_prob,
    # standardization=True,
    color="b",
    legend_location="upper left",
)
fig.suptitle("SKF hidden states", fontsize=10, y=1)
# plt.show()

df_detrend = df_raw.copy()
trend = states.get_mean(
                states_type="smooth", states_name="level", standardization=False,
                scale_const_mean=data_processor.scale_const_mean[
                    data_processor.output_col
                ],
                scale_const_std=data_processor.scale_const_std[
                    data_processor.output_col
                ],
            )
seasonal = states.get_mean(
                states_type="smooth", states_name="lstm", standardization=False,
                scale_const_mean=data_processor.scale_const_mean[
                    data_processor.output_col
                ],
                scale_const_std=data_processor.scale_const_std[
                    data_processor.output_col
                ],
            )
residual = states.get_mean(
                states_type="smooth", states_name="autoregression", standardization=False,
                scale_const_mean=data_processor.scale_const_mean[
                    data_processor.output_col
                ],
                scale_const_std=data_processor.scale_const_std[
                    data_processor.output_col
                ],
            )
df_detrend.values = df_detrend.values.flatten().tolist() - trend
max_value = df_detrend["values"].max()
min_value = df_detrend["values"].min()

fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
axs[0].plot(df_raw["values"], label="Raw Data")
axs[0].plot(df_detrend["values"], label="Detrended Data")
axs[0].axhline(0, color='red', linestyle='--')
axs[0].fill_between(df_detrend.index, min_value, max_value, color='red', alpha=0.1)
axs[0].legend()
axs[0].set_title("SKF & smoother detrending")
axs[1].plot(trend)
axs[1].set_ylabel("Trend")
axs[2].plot(seasonal)
axs[2].set_ylabel("Seasonal")
axs[3].plot(residual)
axs[3].set_ylabel("Residual")
plt.show()