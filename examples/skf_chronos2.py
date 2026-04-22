"""
Anomaly detection with a Switching Kalman Filter where the recurrent-pattern
predictor is Amazon's Chronos-2 (wrapped via the `Auxiliary` component).

Requires: `pip install chronos-forecasting torch`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from chronos import BaseChronosPipeline

from canari import DataProcess, Model, SKF, plot_skf_states
from canari.data_visualization import plot_with_uncertainty
from canari.component import (
    Auxiliary,
    LocalTrend,
    LocalAcceleration,
    WhiteNoise,
)


# ---- Data: synthetic AR + periodic with an injected trend anomaly ----
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.to_datetime(
    pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)[0]
)
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Shorten so Chronos-2 runs in reasonable time on CPU.
df_raw = df_raw.iloc[:500].copy()

# Inject a linear trend anomaly near the end.
time_anomaly = 220
AR_stationary_var = 5**2 / (1 - 0.9**2)
anomaly_magnitude = np.sqrt(AR_stationary_var) * 0.05
df_raw.iloc[time_anomaly:, 0] = df_raw.iloc[time_anomaly:, 0].values + (
    np.arange(len(df_raw) - time_anomaly) * anomaly_magnitude
)

output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    train_split=0.5,
    validation_split=0.1,
    output_col=output_col,
    standardization=False,
)
train_data, _, _, all_data = data_processor.get_splits()


# ---- Chronos-2 predictor ----
LOOK_BACK = 52  # matches the series' periodicity
NUM_FEATURES = 0

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
    dtype=torch.float32,
)


def chronos_predict(mu_input: np.ndarray, var_input: np.ndarray):
    context = torch.tensor(mu_input[:LOOK_BACK], dtype=torch.float32).reshape(1, 1, -1)
    
    # Request quantiles corresponding to -1 std, median, and +1 std
    quantiles, _ = pipeline.predict_quantiles(
        inputs=context,
        prediction_length=1,
        quantile_levels=[0.1587, 0.5, 0.8413], 
    )
    
    # q is now [q_minus_1std, q_median, q_plus_1std]
    q = quantiles[0][0, 0].cpu().numpy()  
    
    mu = np.array([q[1]])
    
    # The distance between +1 std and -1 std is exactly 2 std
    std = (q[2] - q[0]) / 2.0 
    var = np.array([max(std**2, 1e-6)])
    
    return mu, var


# ---- Models ----
sigma_v = 0.2  # data is unstandardized; AR residual std ~ sqrt(25/(1-0.81)) ~ 11
auxiliary = Auxiliary(
    predict_fn=chronos_predict,
    std_error=0.0,
    look_back_len=LOOK_BACK,
    num_features=NUM_FEATURES,
)
noise = WhiteNoise(std_error=sigma_v)

norm_model = Model(LocalTrend(), auxiliary, noise)
abnorm_model = Model(LocalAcceleration(), auxiliary, noise)

# ---- SKF ----
skf = SKF(
    norm_model=norm_model,
    abnorm_model=abnorm_model,
    std_transition_error=1.0e-4,
    norm_to_abnorm_prob=1.0e-3,
    abnorm_to_norm_prob=0.1,
)
skf.auto_initialize_baseline_states(train_data["y"][0:52])

# Warm the aux rolling history on the norm model with the first real targets.
warm = np.asarray(train_data["y"][:LOOK_BACK], dtype=float).reshape(-1)
skf.model["norm_norm"].lstm_output_history.mu = warm.copy()
skf.model["norm_norm"].lstm_output_history.var = np.zeros_like(warm)

# ---- Anomaly detection ----
filter_marginal_abnorm_prob, states = skf.filter(data=all_data)
smooth_marginal_abnorm_prob, states = skf.smoother()

# ---- Plot ----
fig, axes = plot_skf_states(
    data_processor=data_processor,
    states=states,
    states_type="posterior",
    model_prob=filter_marginal_abnorm_prob,
    color="b",
    legend_location="upper left",
)

# Overlay Chronos prior on the same subplot as the posterior.
# plot_skf_states lays out panels in states.states_name order, then the Pr(Abnormal) panel.
aux_panel = axes[states.states_name.index("auxiliary")]
t_axis = aux_panel.lines[0].get_xdata()  # reuse the same time axis used for the posterior
mu_prior = states.get_mean(states_name="auxiliary", states_type="prior", standardization=True)
std_prior = states.get_std(states_name="auxiliary", states_type="prior", standardization=True)
plot_with_uncertainty(
    time=t_axis,
    mu=mu_prior,
    std=std_prior,
    color="orange",
    linestyle="--",
    ax=aux_panel,
)
aux_panel.legend(
    [r"$\mu_{post}$", r"$\pm\sigma_{post}$", r"$\mu_{prior}$", r"$\pm\sigma_{prior}$"],
    loc="upper left",
    fontsize=8,
    ncol=2,
)

fig.suptitle("SKF + Chronos-2 — hidden states", fontsize=10, y=1)
plt.savefig("skf_chronos2.png", dpi=120)
plt.show()
