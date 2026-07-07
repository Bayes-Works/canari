"""Plot AR(1) realizations together with their stationary bounds.

A set of time series is generated from a first-order autoregressive
component
    x_t = phi_AR * x_{t-1} + w_t,   w_t ~ N(0, sigma_AR^2),
whose stationary standard deviation is
    sigma_AR0 = sigma_AR / sqrt(1 - phi_AR^2).
The bounds +/- gamma * sigma_AR0 are drawn for gamma = 1, 2, 3.
"""

from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          'lines.linewidth' : 1,
          }
plt.rcParams.update(params)

# ------------------------------------------------------------------ setup
np.random.seed(2026)

PHI_AR = 0.8          # autoregression coefficient phi^AR
SIGMA_AR = 0.5        # process noise std sigma^AR
N_SERIES = 10         # number of realizations
N_STEPS = 365         # length of each realization
GAMMAS = (1, 2, 3)    # bound multipliers

# Stationary standard deviation of the AR(1) process (Eq. 1.10 / Eq. 2.4)
sigma_ar0 = SIGMA_AR / np.sqrt(1.0 - PHI_AR**2)

# LaTeX-like typography without requiring a TeX installation
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.linewidth": 0.8,
})

# ------------------------------------------------- simulate realizations
def simulate_ar1(n_steps, phi, sigma, x0=0.0):
    """Return one AR(1) realization of length n_steps."""
    x = np.empty(n_steps)
    x[0] = x0
    w = np.random.normal(0.0, sigma, size=n_steps)
    for t in range(1, n_steps):
        x[t] = phi * x[t - 1] + w[t]
    return x

t = np.arange(N_STEPS)
series = np.array([simulate_ar1(N_STEPS, PHI_AR, SIGMA_AR) for _ in range(N_SERIES)])

# ------------------------------------------------------------------ plot
fig, ax = plt.subplots(figsize=(5.0, 1.8))

# Nested stationary-bound regions, lightest for the largest gamma
greys = {1: "0.62", 2: "0.78", 3: "0.90"}
for gamma in sorted(GAMMAS, reverse=True):
    b = gamma * sigma_ar0
    ax.axhspan(-b, b, color=greys[gamma], zorder=0, lw=0)

# Realizations
cmap = plt.get_cmap("viridis")
for i, x in enumerate(series):
    ax.plot(t, x, lw=0.7, alpha=0.85, color=cmap(i / (N_SERIES - 1)), zorder=2)

# Bound edges and their gamma labels at the right edge
for gamma in GAMMAS:
    b = gamma * sigma_ar0
    for sign in (+1, -1):
        ax.axhline(sign * b, color="0.35", lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        rf"$\gamma={gamma}$",
        xy=(1.0, b), xycoords=("axes fraction", "data"),
        xytext=(4, 0), textcoords="offset points",
        va="center", ha="left", fontsize=10,
    )
    # Mark gamma=2 as the default value
    if gamma == 2:
        ax.annotate(
            r"$\gamma=2$", color="tab:red",
            xy=(1.0, b), xycoords=("axes fraction", "data"),
            xytext=(4, 0), textcoords="offset points",
            va="center", ha="left", fontsize=10, fontweight="bold",
        )

ax.set_xlim(0, N_STEPS - 1)
ax.set_ylim(-4.2 * sigma_ar0, 4.2 * sigma_ar0)
ax.set_xlabel(r"Time step $t$")
ax.set_ylabel(r"$x^{\mathtt{AR}}$")
# ax.set_title(
#     rf"AR realizations ($\phi^{{\mathtt{{AR}}}}={PHI_AR}$, "
#     rf"$\sigma^{{\mathtt{{AR}}}}={SIGMA_AR}$, "
#     rf"$\sigma^{{\mathtt{{AR}},0}}={sigma_ar0:.3f}$)",
#     fontsize=11,
# )
ax.tick_params(direction="out")

fig.tight_layout()
fig.savefig("ar_stationary_bounds.pdf", bbox_inches="tight")
fig.savefig("ar_stationary_bounds.png", dpi=300, bbox_inches="tight")
print(f"sigma_AR0 = {sigma_ar0:.4f}")
print("Saved ar_stationary_bounds.pdf / .png")
