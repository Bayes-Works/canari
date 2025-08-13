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

data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
# linear_space = np.linspace(0, 2, num=len(df_raw))
# df_raw = df_raw.add(linear_space, axis=0)

data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

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


####################################################################
######################### Pretrained model #########################
####################################################################

# Load model_dict from local
with open("saved_params/syn_complex_ts_tsmodel.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=29,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

phi_index = model_dict["states_name"].index("phi")
W2bar_index = model_dict["states_name"].index("W2bar")
autoregression_index = model_dict["states_name"].index("autoregression")

print("phi_AR for base_model =", model_dict['states_optimal'].mu_prior[-1][phi_index].item())
print("sigma_AR for base_model =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()))
pretrained_model = Model(
    LocalTrend(mu_states=model_dict["mu_states"][0:2].reshape(-1), var_states=np.diag(model_dict["var_states"][0:2, 0:2])),
    # LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][W2bar_index].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][phi_index].item(), 
                   mu_states=[model_dict["mu_states"][autoregression_index].item()], 
                   var_states=[model_dict["var_states"][autoregression_index, autoregression_index].item()]),
)

pretrained_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

mu_y_preds, std_y_preds,_ = pretrained_model.filter(all_data,train_lstm=False)
pretrained_model.smoother()

#  Plot
state_type = "prior"
#  Plot states from pretrained model
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(4, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
plot_data(
    data_processor=data_processor,
    standardization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax0,
)
time = data_processor.get_time(split="all")
ax0.plot(time, mu_y_preds, color='tab:blue', label='Predicted mean')
ax0.fill_between(time, 
                 mu_y_preds - std_y_preds, 
                 mu_y_preds + std_y_preds, 
                 color='tab:blue', alpha=0.2, label='Predicted std')
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=pretrained_model.states,
    states_type=state_type,
    states_to_plot=['level'],
    sub_plot=ax0,
)
ax0.set_xticklabels([])
ax0.set_title("Hidden states estimated by the pre-trained model")
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=pretrained_model.states,
    states_type=state_type,
    states_to_plot=['trend'],
    sub_plot=ax1,
)
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=pretrained_model.states,
    states_type=state_type,
    states_to_plot=['lstm'],
    sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    standardization=True,
    states=pretrained_model.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax3,
)
ax3.set_xticklabels([])
plt.show()