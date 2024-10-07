from baseline_component import LocalLevel, LocalTrend, LocalAcceleration
from periodic_component import Periodic
from residual_component import Autoregression
from lstm_component import Lstm
from lstm_ssm_model import LstmSsm

# # initialize with no param, no initial hiddens states
local_level = LocalLevel()
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
periodic = Periodic(period=52)
autoregression = Autoregression(phi=1)
lstm = Lstm(look_back_len=52, num_features=1, num_layer=2, num_hidden_unit=[10, 11])

model = LstmSsm(local_acceleration, lstm, autoregression)

check = 1
