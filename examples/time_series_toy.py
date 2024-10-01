import os
import sys

# # # Add the src directory to the sys.path
# try:
#     # Try using __file__ in a script
#     current_dir = os.path.dirname(__file__)
# except NameError:
#     # Use current working directory in Jupyter notebooks
#     current_dir = os.path.join(os.path.dirname(os.getcwd()), 'src')
# sys.path.append(os.path.join(current_dir, 'src'))

# import libraries
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from pytagi import exponential_scheduler
from pytagi.nn import LSTM, Linear, Sequential

from dataProcess import DataProcess
from baseline_component import Local_level, Local_trend, Local_acceleration
from periodic_component import Periodic
from residual_component import Autoregression_deterministic
from assemble_component import Assemble_component
from hybrid import LSTM_SSM, process_input_ssm


# Define parametersx
num_epochs = 30
batch_size = 1
sigma_v = 1
output_col = [0]
num_features = 5
input_seq_len = 26
output_seq_len = 1
seq_stride = 1

# Read csv file
# Dataset
data_file = "data/HQ/LGA007PIAP-E010_Y_train.csv"
df = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

# Time
time_file = "data/HQ/LGA007PIAP-E010_Y_train_datetime.csv"
time = pd.read_csv(time_file, skiprows=1, delimiter=",", header=None)


# # Read Hydro Quebec's .DAT file
# data_file = "data/HQ/DAT/LGA007PIAP-E010.DAT"
# df = DataProcess.read_dat_file(data_file)
# df = df[['Date','Niveau Reservoir (m)','Deplacements cumulatif X (mm)']]

# # water level data
# data_file = "data/HQ/DAT/BASSIN-LGA.DAT"
# df_wl = DataProcess.read_dat_file(data_file)
# df_wl = df_wl[['Date','Temperature min (C)','Temperature max (C)']]

# # Combine data
# df_combined = DataProcess.combine_data(df, df_wl, time_ref_from='df', interpolate_method='linear')


# Resampling data
df = DataProcess.resample_data(df, freq='D', agg_func='mean')

data = DataProcess(
    data = df,
    train_time_start = '2010-02-07 00:00:00',
    val_time_start = '2012-03-23 00:00:00',
    test_time_start = '2013-03-23 00:00:00',
    test_time_end = '2014-03-23 00:00:00',
    time_covariates = ['week_of_year']  # 'hour_of_day','day_of_week', 'week_of_year', 'month_of_year','quarter_of_year'
)

LL = Local_level (sigma_w = 0.1, mu_x = 0.1)
LT = Local_trend(sigma_w = 0.1, mu_x = [0.1, 0.1])
LA = Local_acceleration(sigma_w = 0.1, mu_x = [0.1, 0.1, 0.1])
AR = Autoregression_deterministic(phi_AR=0.9,sigma_AR = 0.1, mu_x=0.1, var_x=0.2)
P = Periodic(sigma_w=0.1, period=52)

model = Assemble_component([LA,P,AR])

check = 1 


# # Network
# net = Sequential(
#     LSTM(1, 30, input_seq_len),
#     LSTM(30, 30, input_seq_len),
#     Linear(30 * input_seq_len, 1),
# )
# net.set_threads(8)

# # State-space models: for baseline hidden states
# hybrid = LSTM_SSM(
#     neural_network = net,           # LSTM
#     baseline = 'trend', # 'level', 'trend', 'acceleration', 'ETS'
#     zB  = np.array([0.1, 1E-4]),    # initial mu for baseline hidden states
#     SzB = np.array([1E-5, 1E-5])    # var
# )

# # -------------------------------------------------------------------------#
# # Training
# mses = []

# pbar = tqdm(range(num_epochs), desc="Training Progress")
# for epoch in pbar:
#     batch_iter = train_dtl.create_data_loader(batch_size, shuffle=False)

#     # Decaying observation's variance
#     sigma_v = exponential_scheduler(
#         curr_v=sigma_v, min_v=0.3, decaying_factor=0.99, curr_iter=epoch
#     )
#     var_y = np.full((batch_size * len(output_col),), sigma_v**2, dtype=np.float32)

#     # Initialize list to save
#     hybrid.init_ssm_hs()
#     mu_preds_lstm = []
#     var_preds_lstm = []
#     mu_preds_unnorm = []
#     obs_unnorm = []
#     #

#     for x, y in batch_iter:
#         mu_x, var_x = process_input_ssm(
#             mu_x = x, mu_preds_lstm = mu_preds_lstm, var_preds_lstm = var_preds_lstm,
#             input_seq_len = input_seq_len,num_features = num_features,
#             )

#         # Feed forward
#         y_pred, _,m_pred, v_pred = hybrid(mu_x, var_x)
#         # Backward
#         hybrid.backward(mu_obs = y, var_obs = var_y)

#         # Training metric
#         pred = normalizer.unstandardize(
#             y_pred.flatten(), train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
#         )
#         obs = normalizer.unstandardize(
#             y, train_dtl.x_mean[output_col], train_dtl.x_std[output_col]
#         )
#         mse = metric.mse(pred, obs)
#         mses.append(mse)
#         mu_preds_lstm.extend(m_pred)
#         var_preds_lstm.extend(v_pred)
#         obs_unnorm.extend(y)
#         mu_preds_unnorm.extend(y_pred)

#     # Smoother
#     hybrid.smoother()

#     # Progress bar
#     pbar.set_description(
#         f"Epoch {epoch + 1}/{num_epochs}| mse: {np.nanmean(mses):>7.2f}",
#         refresh=True,
#     )

