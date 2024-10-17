from src import (
    LocalTrend,
    LstmNetwork,
    Autoregression,
    WhiteNoise,
    Model,
)


model = Model(
    LocalTrend(std_error=0.1, mu_states=[0.1, 0.1], var_states=[0.1, 0.1]),
    LstmNetwork(
        look_back_len=10,
        num_features=1,
        num_layer=2,
        num_hidden_unit=50,
    ),
    Autoregression(std_error=0.1, mu_states=[0.1], var_states=[0.1]),
    WhiteNoise(std_error=0.1),
)

check = 1
