## What is Canari?
Canari is an open-source library for online change point detection in univariate time series. It has been developed in the context of long-term online monitoring where we aim for the early detection of subtle change points in the baseline response, while limiting to a minimum the number of false alarms.

Because Canari’s probabilistic change-point detection relies on 1-step ahead predictions, it is inherently capable of performing short- and long-term forecasting, as well as online data imputation; One-step ahead predictions can either rely solely on a target time series, or on explanatory time series that can be non-linearly related to the target.

In addition to change-point detection and forecasting capabilities, it returns an interpretable decomposition of time series into baseline components representing the irreversible responses, recurrent pattern components caused by external effects, and residual components allowing the characterization of both the model prediction and the observation uncertainties.

## How does it work?

<img src="./docs/_static/Canari_SSM_LSTM.png" alt="Canari SSM LSTM" width="23%" align="right">

The methodological core behind the canary library consists of a seamless integration between [state-space models (SSM)](http://profs.polymtl.ca/jagoulet/Site/PMLCE/CH12.html) and Bayesian neural networks. On the one hand, the Gaussian SSM theory enables modelling baseline responses and residuals, while on the other, [Tractable Approximate Gaussian Inference (TAGI)](https://github.com/lhnguyen102/cuTAGI/tree/main) also enables treating all parameters and hidden states in neural networks as Gaussians.

Canari uses LSTM neural networks to model recurrent patterns as well as non-linear dependencies with respect to explanatory variables. Because both the [SSM and LSTM rely on the same Gaussian conditional](https://www.sciencedirect.com/science/article/pii/S0169207024000335) inference mechanism, their hidden states can be inferred analytically in a same unified probabilistic framework.

<img src="./docs/_static/Canari_example.png" alt="Canari example" width="40%" align="right">

The figure above presents an example where the raw data in red is decomposed in a baseline that is characterized by a baseline “level” component where its rate of change is described by the “trend”. The recurrent pattern is modelled by a LSTM Bayesian neural network where the training and validation set consists only in 4 years of data. The residual characterizing the model errors is itself modelled by a white “noise” component. The change point detection can be performed either online or offline after the training and validation period; The presence of change points is indicated by the probability of regime switches that rise toward 1 on several occasions.

## Getting started
You can get started with Canari by going through our [tutorials](https://bayes-works.github.io/canari/tutorials.html) covering all the main features of the library.

## Installation
Canari available on PyPI, for installation, execute the following command in Terminal:
```
pip install pycanari
```
For more info. on how to install, visit our [installation guide](https://bayes-works.github.io/canari/installation_guide.html).
## Contributors
The principal developer of canary is Van Dai Vuong with the mentoring of Luong Ha Nguyen and oversight by James-A. Goulet. The major contributors to the library are:
- Zhanwen Xin (Online AR, Bounded AR & several+++ bug fixes through PRs)
- David Wardan (SLSTM component & several+++ bug fixes through PRs)

# Acknowledgements
We acknowledge the financial support from Hydro-Québec and the Natural Sciences and Engineering Research Council of Canada in the development of the Canari library.

## License
Canari is released under the MIT license.
**THIS IS AN OPEN SOURCE SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT. NO WARRANTY EXPRESSED OR IMPLIED.**

## Related references
### Papers
* [Coupling LSTM Neural Networks and State-Space Models through Analytically Tractable Inference](https://www.sciencedirect.com/science/article/pii/S0169207024000335) (Van Dai Vuong, Luong-Ha Nguyen and James-A. Goulet. International Journal of Forecasting, 2024)
* [Enhancing structural anomaly detection using a bounded autoregressive component](https://profs.polymtl.ca/jagoulet/Site/Papers/Xin_Goulet_BAR_2024.pdf)(Zhanwen Xin and James-A. Goulet, 2024)
* [Analytically tractable heteroscedastic uncertainty quantification in Bayesian neural networks for regression tasks](http://profs.polymtl.ca/jagoulet/Site/Papers/Deka_TAGIV_2024_preprint.pdf) (Bhargob Deka, Luong-Ha Nguyen and James-A. Goulet. Neurocomputing, 2024)
* [Approximate Gaussian Variance Inference for State-Space Models](https://profs.polymtl.ca/jagoulet/Site/Papers/Deka_Goulet_AGVI_Preprint_2023.pdf) (Bhargob Deka and James-A. Goulet, 2023)
* [The Gaussian multiplicative approximation for state-space models](https://profs.polymtl.ca/jagoulet/Site/Papers/Deka_Ha_Amiri_Goulet_GMA_2022_preprint.pdf)(Bhargob Deka, Luong Ha Nguyen, Saeed Amiri, and James-A. Goulet, 2021)
* [Anomaly Detection with the Switching Kalman Filter for Structural Health Monitoring](https://profs.polymtl.ca/jagoulet/Site/Papers/2017_Nguyen_and_Goulet_AD-SKF.pdf) (Luong Ha Nguyen and James-A. Goulet, 2018)
* [Bayesian dynamic linear models for structural health monitoring](https://profs.polymtl.ca/jagoulet/Site/Papers/Goulet_BDLM_SHM_2017_preprint.pdf)(James-A. Goulet, 2017)
### Theses
*[Analytically Tractable Bayesian Recurrent Neural Networks with Structural Health Monitoring Applications](https://profs.polymtl.ca/jagoulet/Site/Papers/DV_Thesis_2024.pdf) (Van-Dai Vuong, 2024)
* [Analytical Bayesian Parameter Inference for Probabilistic Models with Engineering Applications](https://profs.polymtl.ca/jagoulet/Site/Papers/BhargobDekaThesis.pdf) (Bhargob Deka, 2022)
* [Real-time Anomaly Detection in the Behaviour of Structures](https://profs.polymtl.ca/jagoulet/Site/Papers/LHNguyen_these_2019.pdf) (Luong Ha Nguyen, 2019)
### Book chapter
* [Probabilistic machine learning for civil engineers (chapter 12 - SSM)](http://profs.polymtl.ca/jagoulet/Site/PMLCE/CH12.html) (James-A. Goulet. 2020)
