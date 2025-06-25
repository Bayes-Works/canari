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
    plot_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, Autoregression
import pickle

# # # Read data
data_file = "./data/benchmark_data/detrended_data/test_5_data_detrended.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

# LT anomaly
# anm_mag = 0.010416667/10
anm_start_index = 52*8
anm_mag = 0.2/52
# anm_baseline = np.linspace(0, 3, num=len(df_raw))
anm_baseline = np.arange(len(df_raw)) * anm_mag
# Set the first 52*12 values in anm_baseline to be 0
anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
anm_baseline[:anm_start_index] = 0
df_raw = df_raw.add(anm_baseline, axis=0)

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=0.3,
    validation_split=0.1,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Load model_dict from local
with open("saved_params/real_ts5_detrend_tsmodel.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=16,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

phi_index = model_dict["states_name"].index("phi")
W2bar_index = model_dict["states_name"].index("W2bar")
autoregression_index = model_dict["states_name"].index("autoregression")

print("phi_AR =", model_dict['states_optimal'].mu_prior[-1][phi_index].item())
print("sigma_AR =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()))
model = Model(
    # LocalTrend(mu_states=model_dict["mu_states"][0:2].reshape(-1), var_states=np.diag(model_dict["var_states"][0:2, 0:2])),
    LocalTrend(mu_states=model_dict["mu_states"][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                   mu_states=[model_dict["mu_states"][autoregression_index].item()], 
                   var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
)

model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

#  Abnormal model
ab_model = Model(
            LocalAcceleration(mu_states=[model_dict['early_stop_init_mu_states'][0].item(), model_dict['early_stop_init_mu_states'][1].item(), 0], var_states=[1e-12, 1e-12, 1e-4]),
            LSTM,
            Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                        phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                        mu_states=[model_dict["mu_states"][autoregression_index].item()], 
                        var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
        )

# Switching Kalman filter
skf = SKF(
    norm_model=model,
    abnorm_model=ab_model,
    std_transition_error=0.00017265262887754653,
    norm_to_abnorm_prob=7.171842166745157e-06,
    # std_transition_error=1e-4,
    # norm_to_abnorm_prob=1e-4,
)

# # Anomaly Detection
filter_marginal_abnorm_prob, _ = skf.filter(data=all_data)
smooth_marginal_abnorm_prob, states = skf.smoother()
marginal_abnorm_prob_plot = smooth_marginal_abnorm_prob
p_anm_all = filter_marginal_abnorm_prob

#  Plot
from matplotlib import gridspec
state_type = "prior"
time = data_processor.get_time(split="all")
#  Plot states from pretrained model
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(5, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
ax4 = plt.subplot(gs[4])

plot_data(
    data_processor=data_processor,
    standardization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax0,
)
plot_states(
    data_processor=data_processor,
    standardization=True,
    # states=pretrained_model.states,
    states=states,
    states_type=state_type,
    states_to_plot=['level'],
    sub_plot=ax0,
)
ax0.axvline(x=time[anm_start_index], color='r', linestyle='--')
ax0.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=states,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax1,
)
ax1.set_xticklabels([])
# ax1.set_ylim(-0.002, 0.005)

plot_states(
    data_processor=data_processor,
    standardization=True,
    states=states,
    states_type=state_type,
    states_to_plot=['lstm'],
    sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax3,
)
ax3.set_xticklabels([])

ax4.plot(time, p_anm_all, color='b')
ax4.set_ylabel(r'$p_{\mathrm{anm}}$')
ax4.set_xlim(ax0.get_xlim())
# ax4.axvline(x=time[anm_start_index], color='r', linestyle='--')
ax4.set_ylim(-0.05, 1.05)
ax4.set_yticks([0, 1])
# ax4.set_xticks([time[int(len(time)*1/9)-1], time[int(len(time)*3/9)-1],time[int(len(time)*5/9)-1],time[int(len(time)*7/9)-1],time[int(-1)]])
# ax4.set_xticklabels(['2016', '2018', '2020', '2022', '2024'])
ax4.set_xlim(ax0.get_xlim())

plt.show()

# fig, ax = plot_skf_states(
#     data_processor=data_processor,
#     states=states,
#     states_type="smooth",
#     states_to_plot=["level", "trend", "lstm", "white noise"],
#     model_prob=marginal_abnorm_prob_plot,
#     # standardization=True,
#     color="b",
#     legend_location="upper left",
# )
# fig.suptitle("SKF hidden states", fontsize=10, y=1)
# plt.show()
