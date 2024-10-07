from baseline_component import LocalLevel, LocalTrend, LocalAcceleration
from periodic_component import Periodic
from residual_component import Autoregression
from lstm_component import Lstm

# initialize with no param, no initial hiddens states
local_level = LocalLevel()
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
autoregression = Autoregression()

# initialize with param, no initial hiddens states
local_level_1 = LocalLevel(std_error=0.1)
local_trend_1 = LocalTrend(std_error=0.1)
local_acceleration_1 = LocalAcceleration(std_error=0.1)
periodic_1 = Periodic(period=52)
autoregression_1 = Autoregression(std_error=0.1)

# initialize with param, initial hiddens states
local_level_2 = LocalLevel(std_error=0.1, mu_states=[0.1], var_states=[0.1])
local_trend_2 = LocalTrend(std_error=0.1, mu_states=[0.1, 0.1], var_states=[0.1, 0.1])
local_acceleration_2 = LocalAcceleration(
    std_error=0.1, mu_states=[0.1, 0.1, 0.1], var_states=[0.1, 0.1, 0.1]
)
periodic_2 = Periodic(period=52, mu_states=[0.1, 0.1], var_states=[0.1, 0.1])
autoregression_2 = Autoregression(std_error=0.1, mu_states=[0.1], var_states=[0.1])

#  lstm
lstm = Lstm(
    look_back_len=10,
    num_features=1,
    num_layer=2,
    num_hidden_unit=50,
)

check = 1
