import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from canari.component import LocalTrend, LstmNetwork, Autoregression
import pickle


###########################
###########################
#  Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Add synthetic anomaly to data
# anm_mag = 0
anm_index = 200

# LT anomaly
anm_mag = -15/52
anm_baseline = np.arange(0, len(df_raw)-anm_index, dtype='float')
anm_baseline *= anm_mag

# # LL anomaly
# anm_mag = -50
# anm_baseline = np.zeros(len(df_raw)-anm_index, dtype='float')
# anm_baseline += anm_mag

# # Recurrent anomaly
# anm_mag = -30
# anm_baseline = np.zeros(len(df_raw)-anm_index, dtype='float')
# for i in range(len(df_raw) - anm_index):
#     anm_baseline[i] = anm_mag * np.sin(i / 10)
df_raw.values[anm_index:] = (df_raw.values[anm_index:].squeeze() + anm_baseline).reshape(-1, 1)


# Add trend
trend_true = 0.1
df_raw["values"] += np.arange(len(df_raw)) * trend_true

# Data processor initialization
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=0.6,
    validation_split=0.2,
    output_col=output_col,
)

data_processor.scale_const_mean = np.array([35.571228, 26.068333])
data_processor.scale_const_std = np.array([28.92418, 15.090957])

train_data, val_data, test_data, standardized_data = data_processor.get_splits()

# Standardization constants
scale_const_mean = data_processor.scale_const_mean[output_col].item()
scale_const_std = data_processor.scale_const_std[output_col].item()

with open("saved_params/toy_simple_model_smaller_phi_AR.pkl", "rb") as f:
# with open("saved_params/toy_simple_model_bigger_LSTM_uncertainty.pkl", "rb") as f:
# with open("saved_params/toy_simple_model.pkl", "rb") as f:
    model_dict = pickle.load(f)

lstm = LstmNetwork(
    look_back_len=52,
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
    # manual_seed=1,
)

###########################
###########################
# Reload pretrained model

# Load learned parameters from the saved trained model
phi_index = model_dict["states_name"].index("phi")
W2bar_index = model_dict["states_name"].index("W2bar")
autoregression_index = model_dict["states_name"].index("autoregression")
mu_W2bar_learn = model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()
phi_AR_learn = model_dict['states_optimal'].mu_prior[-1][phi_index].item()
mu_AR = model_dict["mu_states"][autoregression_index].item()
var_AR = model_dict["var_states"][autoregression_index, autoregression_index].item()

print("Learned phi_AR =", phi_AR_learn)
print("Learned sigma_AR =", np.sqrt(mu_W2bar_learn))

# # # # # # #
# Define pretrained model:
pretrained_model = Model(
    LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=[model_dict['states_optimal'].var_prior[0][0,0].item(), model_dict['states_optimal'].var_prior[0][1,1].item()]),
    lstm,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                   mu_states=[model_dict['states_optimal'].mu_prior[0][autoregression_index].item()], 
                   var_states=[model_dict['states_optimal'].var_prior[0][autoregression_index, autoregression_index].item()]),
)

# load lstm's component
pretrained_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

# filter and smoother
mu_obs_preds, var_obs_preds, _ = pretrained_model.filter(standardized_data, train_lstm=False)
pretrained_model.smoother()


state_type = "prior"
# # Plotting results from pre-trained model
fig, ax = plot_states(
    data_processor=data_processor,
    states=pretrained_model.states,
    states_type=state_type,
    standardization=True,
    states_to_plot=[
        "level",
        "trend",
        "lstm",
        "autoregression",
    ],
)
plot_data(
    data_processor=data_processor,
    standardization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax[0],
    plot_test_data=True,
)
time = data_processor.get_time(split="all")
ax[0].plot(time, mu_obs_preds)
ax[0].fill_between(
    time,
    mu_obs_preds - np.sqrt(var_obs_preds),
    mu_obs_preds + np.sqrt(var_obs_preds),
    alpha=0.2,
    color="C0",
)
# Plot the location when the anomaly starts
if anm_mag != 0:
    ax[0].axvline(
        x=df_raw.index[anm_index],
        color="red",
        linestyle="--",
        label="Anomaly start",
    )
fig.suptitle("Hidden states estimated by the pre-trained model", fontsize=10, y=1)
plt.show()
