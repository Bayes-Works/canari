import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import copy
from canari.component import LocalLevel, Autoregression
from canari import (
    DataProcess,
    Model,
    common,
    plot_data,
    plot_prediction,
    plot_states,
)

def kl_divergence(true_mu, true_sigma, cand_mu, cand_sigma):
    var_ratio = (true_sigma ** 2) / (cand_sigma ** 2)
    mean_diff = cand_mu - true_mu
    kl = 0.5 * (var_ratio + (mean_diff ** 2) / (cand_sigma ** 2) - 1 + np.log((cand_sigma ** 2) / (true_sigma ** 2)))
    return kl

# True distribution to generate data
np.random.seed(0)
true_mu, true_sigma = 1, 2.0
ys = np.random.normal(true_mu, true_sigma, size=200)

# Model selection for two candidate: 1 - Normal(1, 1) vs 2 - Normal(-1, 1)
can_sigma = 2
candidates = [(0, can_sigma), (2, can_sigma)]  # (mu, sigma)

# Distribution for the observations
AR_process_error_var_prior = 1e4
var_W2bar_prior = 1e8

model = Model(
    # LocalLevel(mu_states=[0]),
    LocalLevel(mu_states=[0], var_states=[1e4]),
    Autoregression(mu_states=[0, 0, 0, AR_process_error_var_prior],var_states=[1e-06, AR_process_error_var_prior, 0, var_W2bar_prior], phi = 0),
    # Autoregression(mu_states=[0],var_states=[1e-06], phi = 0, std_error=2),

)

mu_obs_preds = []
std_obs_preds = []
model.initialize_states_history()

convergence_epsilon = 1e-2
mean_converged = False
converge_steps = 0

ll_index = model.states_name.index("level")
ar_index = model.states_name.index("autoregression")
mu_obs_last_preds = np.inf

for i, y in enumerate(ys, 1):
    # if mean_converged:
    #     # model.var_states[LL_index] += 1e4
    #     mean_converged = False
    #     converge_steps = 0
    model.observation_matrix[0, ll_index] = 1
    model.observation_matrix[0, ar_index] = 1
    if mean_converged:
        model.var_states[ll_index, ll_index] = convergence_epsilon ** 2
    mu_obs_pred, var_obs_pred, _, var_states_prior = model.forward()
    if mean_converged is False:
        model.observation_matrix[0, ar_index] = 0
    else:
        model.observation_matrix[0, ll_index] = 0
    (
        delta_mu_states,
        delta_var_states,
        mu_states_posterior,
        var_states_posterior,
    ) = model.backward(y)

    if abs(mu_obs_pred.item() - mu_obs_last_preds) < convergence_epsilon:
        converge_steps += 1
        if converge_steps >= 5:
            mean_converged = True

    model._save_states_history()
    model.set_states(mu_states_posterior, var_states_posterior)

    mu_obs_preds.append(mu_obs_pred.item())
    std_obs_preds.append(var_obs_pred.item()**0.5)
    mu_obs_last_preds = copy.deepcopy(mu_obs_pred.item())

# Compare the predicted distribution with candidates
all_kl = []
for i, (cand_mu, cand_sigma) in enumerate(candidates, 1):
    kl_cand = []
    for n in range(len(mu_obs_preds)):
        kl = kl_divergence(cand_mu, cand_sigma, mu_obs_preds[n], std_obs_preds[n])
        # kl = kl_divergence_moment_approx(cand_mu, cand_sigma, mean_track[n], std_track[n])
        kl_cand.append(kl)
    all_kl.append(kl_cand)
# Convert the KL distance to probabilities
all_kl = np.array(all_kl)
prob_model = [all_kl[1] / (all_kl[0] + all_kl[1]), all_kl[0] / (all_kl[0] + all_kl[1])]


# Plot results
time = np.arange(len(ys))
fig = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(2, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax0.plot(time, ys, label="Observations", color="blue", alpha=0.5)
ax0.plot(time, mu_obs_preds, label="Predicted Mean", color="red")
ax0.fill_between(
    time,
    np.array(mu_obs_preds) - np.array(std_obs_preds),
    np.array(mu_obs_preds) + np.array(std_obs_preds),
    color="red",
    alpha=0.2,
)
ax0.set_ylim(-5, 5)
ax0.axhline(true_mu, color="black", linestyle="--", label="True Mean")
ax0.set_xlabel("Time")
ax0.set_ylabel("obs dist")
# ax0.set_title("Autoregression Model Predictions")
# ax0.legend()

ax1.plot(prob_model[0], label="Prob Model 1", color='C0')
ax1.plot(prob_model[1], label="Prob Model 2", color='C1')
ax1.set_ylabel("Prob")
ax1.set_ylim(0, 1)
plt.show()