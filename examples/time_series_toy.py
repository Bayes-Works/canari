import copy
import os
import sys


sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src")))

from baseline_component import LocalLevel, LocalTrend, LocalAcceleration
from periodic_component import Periodic
from autoregression_component import Autoregression
from lstm_component import Lstm
from model import Model


model = Model(
    LocalTrend(std_error=0.1, mu_states=[0.1, 0.1], var_states=[0.1, 0.1]),
    Lstm(
        look_back_len=10,
        num_features=1,
        num_layer=2,
        num_hidden_unit=50,
    ),
    Autoregression(std_error=0.1, mu_states=[0.1], var_states=[0.1]),
)

check = 1
