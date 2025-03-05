import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from src import (
    LocalTrend,
    LstmNetwork,
    Autoregression,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from examples import DataProcess
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
import src.common as common


LSTM = LstmNetwork(
        look_back_len=52,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

import pickle
with open("saved_params/toy_model_dict.pkl", "rb") as f:
    pretrained_model_dict = pickle.load(f)
print("phi_AR =", pretrained_model_dict['states_optimal'].mu_prior[-1][pretrained_model_dict['phi_index']].item())
print("sigma_AR =", np.sqrt(pretrained_model_dict['states_optimal'].mu_prior[-1][pretrained_model_dict['W2bar_index']].item()))
model = Model(
    LocalTrend(mu_states=[0, 0], var_states=[1e-12, 1e-12], std_error=0), # True baseline values
    # LocalTrend(mu_states=pretrained_model_dict["mu_states"][0:2].reshape(-1), var_states=np.diag(pretrained_model_dict["var_states"][0:2, 0:2])),
    # LocalTrend(mu_states=pretrained_model_dict["mu_states"][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(pretrained_model_dict['states_optimal'].mu_prior[-1][pretrained_model_dict['W2bar_index']].item()), 
                   phi=pretrained_model_dict['states_optimal'].mu_prior[-1][pretrained_model_dict['phi_index']].item(), 
                   mu_states=[pretrained_model_dict["mu_states"][pretrained_model_dict['autoregression_index']].item()], 
                   var_states=[pretrained_model_dict["var_states"][pretrained_model_dict['autoregression_index'], pretrained_model_dict['autoregression_index']].item()]),
)

model.lstm_net.load_state_dict(pretrained_model_dict["lstm_network_params"])

model.initialize_lstm_output_history()
input_covariates = [0]

for i in range(3):
    mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
        model.lstm_output_history, input_covariates
    )
    print("Input:")
    print(mu_lstm_input)
    print(var_lstm_input)
    mu_lstm_pred, var_lstm_pred = model.lstm_net.forward(
        mu_x=np.float32(mu_lstm_input), var_x=np.float32(var_lstm_input)
    )
    model.lstm_net.reset_lstm_states()

    print("Output:")
    print(mu_lstm_pred)
    print(var_lstm_pred)
    print('=====================')