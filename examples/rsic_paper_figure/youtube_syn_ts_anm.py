import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from canari.component import LocalTrend, LstmNetwork, Autoregression
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
import pickle
from pytagi import Normalizer

from matplotlib import ticker
import matplotlib.dates as mdates

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          'lines.linewidth' : 1,
          }
plt.rcParams.update(params)


# # # Read data
data_file = "./data/toy_time_series/paper_data_decompose_delay_example.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

# LT anomaly
# anm_mag = 0.010416667/10
time_anomaly = 52*4
anm_mag = 5/52
# anm_baseline = np.linspace(0, 3, num=len(df_raw))
anm_baseline = np.arange(len(df_raw)) * anm_mag
# Set the first 52*12 values in anm_baseline to be 0
anm_baseline[time_anomaly:] -= anm_baseline[time_anomaly]
anm_baseline[:time_anomaly] = 0
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

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Define AR model
AR_process_error_var_prior = 1e2
var_W2bar_prior = 1e2
AR = Autoregression(mu_states=[0, 0, 0, 0, 0, AR_process_error_var_prior],var_states=[1e-06, 0.01, 0, AR_process_error_var_prior, 0, var_W2bar_prior])
LSTM = LstmNetwork(
        look_back_len=52,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

model = Model(
    LocalTrend(),
    LSTM,
    AR,
)
# model._mu_local_level = 0
model.auto_initialize_baseline_states(train_data["y"][0 : 52 * 3])



# Load model_dict to local
import pickle
with open("saved_params/paper_example.pkl", "rb") as f:
    model_dict = pickle.load(f)

####################################################################
######################### Pretrained model #########################
####################################################################
phi_index = model_dict["states_name"].index("phi")
W2bar_index = model_dict["states_name"].index("W2bar")
autoregression_index = model_dict["states_name"].index("autoregression")

print("phi_AR for base_model =", model_dict['states_optimal'].mu_prior[-1][phi_index].item())
print("sigma_AR for base_model =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()))
print("phi_AR_gen =", model_dict['gen_phi_ar'])
print("sigma_AR_gen =", model_dict['gen_sigma_ar'])
pretrained_model = Model(
    # LocalTrend(mu_states=model_dict["mu_states"][0:2].reshape(-1), var_states=np.diag(model_dict["var_states"][0:2, 0:2])),
    LocalTrend(mu_states=[0, 0], var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                   mu_states=[model_dict["mu_states"][autoregression_index].item()], 
                   var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
)

pretrained_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

mu_y_preds, std_y_preds,_ = pretrained_model.filter(normalized_data,train_lstm=False)
pretrained_model.smoother()

time = data_processor.get_time(split="all")
states_type = "prior"
# Get the mean and std of the hidden states
level_mean = pretrained_model.states.get_mean(states_type=states_type, states_name="level",
            standardization=False,scale_const_mean=data_processor.scale_const_mean[data_processor.output_col],scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
level_std = pretrained_model.states.get_std(states_type=states_type, states_name="level",
            standardization=False,scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
trend_mean = pretrained_model.states.get_mean(states_type=states_type, states_name="trend",
            standardization=False,scale_const_mean=data_processor.scale_const_mean[data_processor.output_col],scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
trend_std = pretrained_model.states.get_std(states_type=states_type, states_name="trend",
            standardization=False,scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
ar_mean = pretrained_model.states.get_mean(states_type=states_type, states_name="autoregression",
            standardization=False,scale_const_mean=data_processor.scale_const_mean[data_processor.output_col],scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
ar_std = pretrained_model.states.get_std(states_type=states_type, states_name="autoregression",
            standardization=False,scale_const_std=data_processor.scale_const_std[data_processor.output_col],)

# #  Plot
plt.figure(figsize=(4, 3))
gs = gridspec.GridSpec(4, 1)
# ax0 = plt.subplot(gs[0])
# ax1 = plt.subplot(gs[1])
# ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])

# ax0: plot data
ax3.plot(time, df_raw["obs"], 'k')
# ax3.plot(time, level_mean, color="tab:blue")
# ax3.fill_between(time, level_mean - level_std, level_mean + level_std, color="tab:blue", alpha=0.2)
# ax0.set_xticklabels([])
ax3.set_ylabel("Obs.")
ax3.axvline(x=time[time_anomaly], color='r', linestyle='--')
ax3.plot(time, anm_baseline, color='gray', alpha=0.5)

# # ax1: plot trend
# ax1.plot(time, trend_mean, color="tab:blue")
# ax1.fill_between(time, trend_mean - trend_std, trend_mean + trend_std, color="tab:blue", alpha=0.2)
# ax1.set_xticklabels([])
# ax1.set_ylabel("Trend")
# ax1.set_yticks([-0.02, -0.05])

# # ax3: plot autoregression
# ax2.plot(time, ar_mean, color="tab:blue")
# ax2.fill_between(time, ar_mean - ar_std, ar_mean + ar_std, color="tab:blue", alpha=0.2)
# ax2.set_xticklabels([])
# ax2.set_ylabel("Residual")

# # add grid in all subplots
for ax in [ax3]:
    # ax.grid(axis='x')  # vertical lines only
    # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # Set major ticks every 4 years
    ax.xaxis.set_major_locator(mdates.YearLocator(2))

# plt.gcf().align_ylabels([ax0, ax1, ax2, ax3])
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.3)
plt.savefig('yt_syn_ts_anm.png', dpi=300)
plt.show()