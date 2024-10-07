from baseline_component import LocalLevel, LocalTrend, LocalAcceleration
from periodic_component import Periodic
from residual_component import Autoregression
from lstm_component import Lstm
from lstm_ssm_model import LstmSsm

# # initialize with no param, no initial hiddens states
local_level = LocalLevel()
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
periodic = Periodic(period=1)
autoregression = Autoregression(phi=1)
lstm = Lstm(look_back_len=52, num_features=1, num_layer=2, num_hidden_unit=50)

model = LstmSsm(local_level, periodic, autoregression)

check = 1

# [[np.cos(w), np.sin(w)], [-np.sin(w), np.cos(w)]]
