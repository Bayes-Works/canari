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

from baseline_component import Local_level, Local_trend, Local_acceleration
from periodic_component import Periodic
from residual_component import Autoregression_deterministic

# initialize with no param, no initial hiddens states
LL = Local_level ()
LT = Local_trend()
LA = Local_acceleration()
AR = Autoregression_deterministic()
P  = Periodic(period=52)

# initialize with param
LL_param = Local_level (sigma_w=0.1)
LT_param = Local_trend(sigma_w=0.1)
LA_param = Local_acceleration(sigma_w=0.1)
AR_param = Autoregression_deterministic(sigma_AR=0.1, phi_AR=0.9)
P_param  = Periodic(sigma_w=0.1, period=52)

# initialize with param and initial hiddens states
LL_param_hs = Local_level (sigma_w=0.1, mu_x = [0.1], var_x=[0.1])
LT_param_hs = Local_trend(sigma_w=0.1, mu_x = [0.1, 0.2], var_x=[0.1, 0.2])
LA_param_hs = Local_acceleration(sigma_w=0.1, mu_x = [0.1, 0.2, 0.3], var_x=[0.1, 0.2, 0.3])
AR_param_hs = Autoregression_deterministic(sigma_AR=0.1, phi_AR=0.9, mu_x = [0.1], var_x=[0.1])
P_param_hs  = Periodic(sigma_w=0.1, period=52, mu_x = [0.1, 0.2], var_x=[0.1, 0.2])

check = 1


