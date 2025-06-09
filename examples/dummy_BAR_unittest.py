from canari import Model, common
from canari.component import (
    LocalTrend,
    LocalLevel,
    LocalAcceleration,
    LstmNetwork,
    Autoregression,
    WhiteNoise,
    Periodic,
    BaseComponent,
)

mu_W2bar_prior = 3
var_AR_prior = 10
var_W2bar_prior = 2.5

ar = Autoregression(
    mu_states=[0.2, 0.8, 0, 0, 0, mu_W2bar_prior],
    var_states=[1.2, 0.25, 0, var_AR_prior, 0, var_W2bar_prior],
)

model = Model(ar)

mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = model.forward()
(
    delta_mu_states,
    delta_var_states,
    mu_states_posterior,
    var_states_posterior,
) = model.backward(0.1)

print(mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior)
print('-----------------------------')
print(delta_mu_states, delta_var_states, mu_states_posterior, var_states_posterior)