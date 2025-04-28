# pycanari
Open-source library for probabilistic anomaly detection in time series 

## Installation

#### Create Miniconda Environment

1. Install Miniconda by following these [instructions](https://docs.conda.io/en/latest/miniconda.html)
2. Create a conda environment named `canari`:

    ```sh
    conda create --name canari python=3.10
    ```

3. Activate conda environment:

    ```sh
    conda activate canari
    ```

#### Install pycanari
1. Install pycanari

    ```sh
    pip install pycanari
    ```

2. [Search pycanari and download pycanari-0.0.2.tar.gz file from the lastest version](https://pypi.org)

3. Copy the downloaded pycanari-0.0.2.tar file to the your working folder

4. Extract the pycanari-0.0.2.tar file using:

    ```sh
    tar -xvf pycanari-0.0.2.tar
    ```
5. Set directory
    ```sh
    cd pycanari-0.0.2
    ```

6. Run:

    ```sh
    conda install -c conda-forge libstdcxx-ng
    ```
    
7. Install requirements:

    ```sh
    pip install -r requirements.txt
    ```

8. Test **pycanari** package:

    ```sh
    python -m examples.toy_forecast
    ```

NOTE: Replace the name `pycanari-0.0.2` with the corresponding version, e.g. pycanari-0.0.3


## Code organization
for development purposes
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
|    |----baseline_component (derived class)
|    |      (v0) LL(param_value, initial hidden states),LT, LA, LcT, LcA, TcA 
|    |      (v2) exponential smoothing
|    |----intervention_component (derived class)
|    |      (v1) LL, LT, LA
|    |      (v2) \theta for LSTM
|    |----LSTM_component (derived class) (v0)
|    |----periodic_component (derived class) (v0)
|    |----residual_component (derived class)
|    |      (v0) AR: deterministic
|    |      (v1) AR: online
|    |      (v2) BAR
|    |
|    |----base_model (class) (v0)
|    |----TAGI-LSTM/SSM model (derived class)
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
|    |----SKF model (derived class)
|    |      (v0) filter: model.filter
|    |      (v0) forecast: model.forecast
|    |      (v0) smoother: model.smoother
|    |      (v1) hyper-paramters grid-search: model.gridSearch()
|    |      (v1) save/load model/parameters: component wise
|    |      (v2) parallel computing with multiple seeds: model.seeds([1]) or model.seeds([1 2 3]).
|    |----RL model (derived class) (v2)
|    |
|    |
|    |----metrics(Likelihood, log-likelihood (v0), MSE/RMSE, MAE, p50, p90 (v1))
|    |----dataVisualization (plot data, predictions, hidden states) (v0:basic, v1:advanced)
|    |----task
|           (v0) unit tests
|           (v1) synthetic data generation
|           (v2) benmarking on some datasetes
|
|----data
|----examples
|----unit_test
|----saved_results
|----saved_models

```
