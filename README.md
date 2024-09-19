# canari
Open-source library for probabilistic anomaly detection in time series 

```
canari
|
|----dataProcess
|    |  read data   
|    |  resampling
|    |  split data into train/validation/test sets
|    |  normalization/unnormalization (from pytagi)
|    |  automated outliers removal
|    |  save data
|
|----common
|    |  forward(1 sample)
|    |  backward(1 sample)
|    |  RTS(1 sample)
|    |  step()
|    
|----hybridmodel(TAGI-LSTM/SSM)
|    |----model constructions
|    |    |----model components and their parameters
|    |    |    |  baselines: LL(),LT(),LA(),LcT(), LcA(), TcA(), exponential smoothing()
|    |    |    |  interventions: LL(),LT(),LA(),\theta() for LSTM
|    |    |    |  TAGI-LSTM: LSTM() (from pyTAGI)
|    |    |    |  periodic
|    |    |    |  residual: AR(), BAR()
|    |    |    
|    |    |----initial hidden states for components
|    |    |    |  baselines: LL(),LT(),LA(),LcT(), LcA(), TcA(), exponential smoothing()
|    |    |    |  interventions: LL(),LT(),LA(),\theta() for LSTM
|    |    |    |  TAGI-LSTM: LSTM() (from pyTAGI)
|    |    |    |  periodic
|    |    |    |  residual: AR(), BAR()
|    |    |
|    |    |----build: model.build()
```
