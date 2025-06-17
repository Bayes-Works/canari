Theory
======

The Canary library relies on the integration of state-space models (SSM) and Bayesian neural networks with recurrent architectures. This section contains resources to understand the theory and methods behind the library.

Background
----------

Both SSMs and Bayesian neural networks rely extensively on linear algebra and probability theory.

- Probabilistic Machine Learning for Civil Engineers [`Chapter 2 - Linear Algebra <http://profs.polymtl.ca/jagoulet/Site/PMLCE/CH2.html>`_] [`YouTube <https://youtu.be/ORDbWuYzuRE?si=rsb1XMG8ENW0GFiy>`_]
- Probabilistic Machine Learning for Civil Engineers [`Chapter 3 - Probability Theory <http://profs.polymtl.ca/jagoulet/Site/PMLCE/CH3.html>`_] [`YouTube <https://youtu.be/Ndu3z4uUREs?si=JcA4FqjNua0crJ9i>`_]
- Probabilistic Machine Learning for Civil Engineers [`Chapter 4 - Distributions <http://profs.polymtl.ca/jagoulet/Site/PMLCE/CH4.html>`_] [`YouTube <https://youtu.be/BKs_2q1hnTk?si=IKO4sLmU4Yzxw6Hh>`_]

State-space Models
------------------

State-space models (SSM), or more specifically linear-Gaussian SSMs used by Canary, model dynamic systems in a recurrent manner using a vector of hidden (i.e., not directly observed) states. The dynamic evolution between two consecutive time steps is governed by a first system of linear equations. A second set of linear equations defines the observation model as a function of the same hidden state vector.

In the case of linear-Gaussian SSMs, exact tractability of all calculations is maintained. This enables closed-form exact Bayesian inference and marginalization using the Kalman Filter and Rauch–Tung–Striebel (RTS) smoother.

**Kalman Filtering, Smoothing, and Regime Switching**

- Probabilistic Machine Learning for Civil Engineers [`Chapter 12 - State-space Models <http://profs.polymtl.ca/jagoulet/Site/PMLCE/CH12.html>`_] [`YouTube <https://youtu.be/8lPBkkbtNW8?si=CuPIZObGkpZTsjX7>`_]

**Basic Components: Level, Trend, Acceleration, Periodic, Autoregressive, and White Noise**

- Probabilistic Machine Learning for Civil Engineers [`Chapter 12 - State-space Models <http://profs.polymtl.ca/jagoulet/Site/PMLCE/CH12.html>`_] [`YouTube <https://youtu.be/2vf-d_fRCXs?si=pLsuMwG6N3PQ4tFo>`_]

**Advanced Components**

- Online Autoregressive [`Paper <https://profs.polymtl.ca/jagoulet/Site/Papers/Deka_Goulet_AGVI_Preprint_2023.pdf>`_] [`Thesis <https://profs.polymtl.ca/jagoulet/Site/Papers/BhargobDekaThesis.pdf>`_] [`YouTube <https://youtu.be/Jzkiof8X244>`_]
- Approximate Gaussian Variance Inference for SSM [`Paper <https://profs.polymtl.ca/jagoulet/Site/Papers/Deka_Goulet_AGVI_Preprint_2023.pdf>`_] [`Thesis <https://profs.polymtl.ca/jagoulet/Site/Papers/BhargobDekaThesis.pdf>`_] [`YouTube <https://youtu.be/ho2wvuq2H68>`_]
- Bounded Autoregressive Component [`Paper <https://profs.polymtl.ca/jagoulet/Site/Papers/Xin_Goulet_BAR_2024.pdf>`_] [`YouTube <https://youtu.be/8jqwKp97PoY>`_]

TAGI & LSTM Neural Networks
---------------------------

The TAGI method treats all the parameters and hidden units in a neural network as Gaussian random variables. This enables reliance on the same closed-form Gaussian conditional equations as the linear-Gaussian SSM.

- Tractable Approximate Gaussian Inference (TAGI) for Bayesian Neural Networks [`Paper <https://profs.polymtl.ca/jagoulet/Site/Papers/2021_Goulet_Nguyen_Amiri_TAGI_JMLR.pdf>`_] [`YouTube <https://youtu.be/jqd3Bj0q2Sc>`_]
- Coupling LSTM Neural Networks and SSM through Analytically Tractable Inference [`Paper <https://profs.polymtl.ca/jagoulet/Site/Papers/Vuong_el_al_TAGI_LSTM_2024.pdf>`_] [`Thesis <https://profs.polymtl.ca/jagoulet/Site/Papers/DV_Thesis_2024.pdf>`_] [`YouTube <https://youtu.be/urYuJXzMzrk>`_]
