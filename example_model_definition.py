from canari.components import (
    LocalLevel,
    LocalTrend,
    LSTM,
    Autoregression,
)

LL_block = LocalLevel()
LT_block = LocalTrend()
AR_block_explanatory = Autoregression(phi=0)
AR_block_dependent = Autoregression()
LSTM_explanatory = LSTM(nb_layer=1, nb_states=100, lookback_window=52)
LSTM_dependent= LSTM(nb_layer=1, nb_states=100, lookback_window=52, input='model_explanatory')

model_explanatory = definition(LL_block, AR_block_explanatory, LSTM_explanatory)
model_dependant = definition(LT_block, AR_block_dependent, LSTM_dependent)

model_example = join(model_explanatory, model_dependant)