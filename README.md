# canari
Open-source library for probabilistic anomaly detection in time series 

```
canari
|
|----src
|    |----dataProcess
|    |    |  read data   
|    |    |  resampling
|    |    |  split data into train/validation/test sets
|    |    |  normalization/unnormalization (from pytagi)
|    |    |  automated outliers removal
|    |    |  save data
|    |
|    |----common: base operations
|    |    |  forward(1 sample)
|    |    |  backward(1 sample)
|    |    |  RTS(1 sample)
|    |    |  step()
|    |    
|    |----hybridmodel TAGI-LSTM/SSM (model)
|    |    |----model constructions
|    |    |    |----model components and their parameters
|    |    |    |    |  baselines: LL(param_value),LT(),LA(),LcT(), LcA(), TcA(), exponential smoothing()
|    |    |    |    |  interventions: LL(),LT(),LA(),\theta() for LSTM
|    |    |    |    |  TAGI-LSTM: LSTM() (from pyTAGI)
|    |    |    |    |  periodic
|    |    |    |    |  residual: AR(), BAR()
|    |    |    |    |  NOTE: user-define or grid-search for parameters
|    |    |    |              e.g. user-define: AR(sigma_AR = 0.9)  vs grid-search: AR(sigma_AR = [0.7, 0.8, 0.9])
|    |    |    |    
|    |    |    |----initial hidden states for components
|    |    |    |    |  baselines: LL(hidden_state_value),LT(),LA(),LcT(), LcA(), TcA(), exponential smoothing()
|    |    |    |    |  interventions: LL(),LT(),LA(),\theta() for LSTM
|    |    |    |    |  TAGI-LSTM: LSTM() (from pyTAGI)
|    |    |    |    |  periodic
|    |    |    |    |  residual: AR(), BAR()
|    |    |    |
|    |    |    |----build: model.build()
|    |    |
|    |    |----task: on time series data (ts_data)
|    |    |     |  hyper-paramters grid-search: model.gridSearch()
|    |    |     |  training: model.train(ts_data), recursively apply forward(1sample) and backward(1sample),
|    |    |     |            goal: obtain TAGI-LSTM's weights and biases.
|    |    |     |  filter: model.filter(ts_data), recursively apply forward(1sample) and backward(1sample),
|    |    |     |            goal: only update SSM's hidden states
|    |    |     |  forecast: model.forecast(ts_data), recursively apply forward(1sample)
|    |    |     |  smoother: model.smoother(ts_data), recursively apply forward(1sample), backward() and RTS(),
|    |    |     |            goal: smoothed estimates for SSM's hidden states, and LSTM's hidden and cell states
|    |    |     |  local/global models: local(one TAGI-LSTM model for each time series) vs global (one TAGI-LSTM shared for all time series)
|    |    |     |                      (NOTE: No idea how to do)
|    |    |     |  parallel computing with multiple seeds: model.seeds([1]) or model.seeds([1 2 3]). (NOTE: No idea how to do)
|    |    |     |  online learning (David): model.onlineTrain()
|    |    |
|    |    |---- save/load model/parameters: component wise
|    |                         
|    |----anomaly detection model (AD_model)
|    |    |----SKF: AD_model = SKF(data='', model=[model1# model#2], hyper-parameter=[])
|    |    |    |  filter: AD_model.filterts_data
|    |    |    |  smoother: AD_model.smoother(ts_data)
|    |    |    |  forecast: AD_model.forecast(ts_data)
|    |    |
|    |    |----Reinforement learning: AD_model = RL(data='',model='', hyper-parameter=[])
|    |    |    |....
|    |    |
|    |    |----hyper-paramters grid-search: AD_model.gridSearch(),
|    |    |----parallel computing with multiple seeds: AD_model.seeds([1]) or AD_model.seeds([1 2 3]). (NOTE: No idea how to do)
|    |    |----save/load model/parameters
|    |
|    |----other tasks
|    |    |  synthetic data generation
|    |    |  benmarking on some datasetes
|    |
|    |----metrics(MSE/RMSE, MAE, p50, p90, Likelihood, log-likelihood)
|    |----dataVisualization (plot data, predictions, hidden states)
|
|----data
|----examples
|----unit_test
|----saved_results
|----saved_models



```
