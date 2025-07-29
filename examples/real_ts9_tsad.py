import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari.component import LocalTrend, LstmNetwork, Autoregression, WhiteNoise
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_states,
    common,
)
from src.hsl_detection import hsl_detection
from src.hsl_generator import hsl_generator
import pytagi.metric as metric
from matplotlib import gridspec
import pickle
from pytagi import Normalizer


# # # Read data
data_file = "./data/benchmark_data/test_9_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values", "water_level", "temp_min", "temp_max"]
df_raw = df_raw.iloc[:, :-3]

# Data pre-processing
output_col = [0]
# train_split=0.28
# validation_split=0.07
train_split=0.35354
validation_split=0.087542
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
with open("saved_params/real_ts9_wn_ssm_dist_update.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=33,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
    )
phi_ar = model_dict['phi_ar']
sigma_ar = model_dict['sigma_ar']

print("phi_AR =", phi_ar)
print("sigma_AR =", sigma_ar)

wn_model = Model(
    LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=np.diag(model_dict['states_optimal'].var_prior[0][0:2, 0:2])),
    LSTM,
    WhiteNoise(std_error=model_dict["sigma_v_optimal"]),
)

ar_model = Model(
    LocalTrend(mu_states=model_dict['states_optimal'].mu_prior[0][0:2].reshape(-1), var_states=np.diag(model_dict['states_optimal'].var_prior[0][0:2, 0:2])),
    LSTM,
    Autoregression(phi=phi_ar, std_error=sigma_ar),
)

wn_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])
ar_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

ltd_error = 1e-5

hsl_tsad_agent = hsl_detection(base_model=wn_model, data_processor=data_processor, drift_model_process_error_std=ltd_error, y_std_scale = 1,
                               phi_ar=phi_ar, sigma_ar=sigma_ar)
hsl_tsad_generator = hsl_generator(base_model=ar_model, data_processor=data_processor, drift_model_process_error_std=ltd_error, y_std_scale = 1,
                               phi_ar=phi_ar, sigma_ar=sigma_ar)
# hsl_tsad_generator = hsl_detection(base_model=wn_model, data_processor=data_processor, drift_model_process_error_std=ltd_error, y_std_scale = 1,
#                                phi_ar=phi_ar, sigma_ar=sigma_ar)

# # Get flexible drift model from the beginning
# hsl_tsad_agent_pre = hsl_detection(base_model=wn_model.load_dict(wn_model.get_dict()), data_processor=data_processor, drift_model_process_error_std=ltd_error)
# hsl_tsad_agent_pre.filter(train_data)
# hsl_tsad_agent_pre.filter(validation_data)
# hsl_tsad_agent.drift_model.var_states = hsl_tsad_agent_pre.drift_model.var_states


mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_generator.filter(train_data, buffer_LTd=True)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_generator.filter(validation_data, buffer_LTd=True)
# hsl_tsad_agent.estimate_LTd_dist()
# # hsl_tsad_agent.mu_LTd = -1.6632523544953974e-06
# # hsl_tsad_agent.LTd_std = 2.9068328673424882e-05
# hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = hsl_tsad_agent.LTd_std * 1)

hsl_tsad_generator.collect_synthetic_samples(num_time_series=1, save_to_path='data/hsl_tsad_training_samples/itv_learn_samples_real_ts9_wn.csv')