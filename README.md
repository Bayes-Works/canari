# canari
Open-source library for probabilistic anomaly detection in time series 

```
canari
|
|----src
|    |----dataProcess (class) 
|    |      (v0) read data
|    |      (v0) resampling 
|    |      (v0) split data into train/validation/test sets 
|    |      (v0) normalization/unnormalization (from pytagi)
|    |      (v1) automated outliers removal 
|    |      (v1) save data 
|    |
|    |----common (class)
|    |      (v0) forward(1 sample)
|    |      (v0) backward(1 sample)
|    |      (v0) RTS(1 sample)
|    |      (v0) step()
|    |
|    |----base_component (class) (v0)
|    |----baseline_component (inherent class)
|    |      (v0) LL(param_value, initial hidden states),LT, LA, LcT, LcA, TcA 
|    |      (v2) exponential smoothing
|    |----intervention_component (inherent class)
|    |      (v1) LL, LT, LA
|    |      (v2) \theta for LSTM
|    |----LSTM_component (inherent class) (v0)
|    |----periodic_component (inherent class) (v0)
|    |----residual_component (inherent class)
|    |      (v0) AR
|    |      (v2) BAR
|    |
|    |----base_model (class) (v0)
|    |----TAGI-LSTM/SSM model (inherent class)
|    |      (v0) training: model.train(ts_data) (v0)
|    |                     goal: obtain TAGI-LSTM's weights and biases.
|    |      (v0) filter: model.filter(ts_data), recursively apply forward(1sample) and backward(1sample),
|    |                   goal: only update SSM's hidden states
|    |      (v0) forecast: model.forecast(ts_data), recursively apply forward(1sample)
|    |      (v0) smoother: model.smoother(ts_data), recursively apply forward(1sample), backward() and RTS()
|    |                     goal: smoothed estimates for SSM's hidden states, and LSTM's hidden and cell states
|    |      (v1) hyper-paramters grid-search: model.gridSearch()
|    |      (v1) save/load model/parameters: component wise
|    |      (v2) parallel computing with multiple seeds: model.seeds([1]) or model.seeds([1 2 3]).
|    |      (v2) online learning (David): model.onlineTrain()
|    |----SKF model (inherent class)
|    |      (v0) filter: model.filter
|    |      (v0) forecast: model.forecast
|    |      (v0) smoother: model.smoother
|    |      (v1) hyper-paramters grid-search: model.gridSearch()
|    |      (v1) save/load model/parameters: component wise
|    |      (v2) parallel computing with multiple seeds: model.seeds([1]) or model.seeds([1 2 3]).
|    |----RL model (inherent class) (v2)
|    |
|    |
|    |----metrics(MSE/RMSE, MAE, p50, p90, Likelihood, log-likelihood) (v0)
|    |----dataVisualization (plot data, predictions, hidden states) (v0)
|    |----task
|           (v1) synthetic data generation
|           (v1) unit tests
|           (v2) benmarking on some datasetes
|
|----data
|----examples
|----unit_test
|----saved_results
|----saved_models

```
