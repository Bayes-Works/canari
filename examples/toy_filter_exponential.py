import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari import (
    DataProcess,
    Model,
    plot_states,
)
from canari.component import Exponential, WhiteNoise, LocalTrend

# Read data
data_file = "./data/toy_time_series/synthetic_exponential_localtrend.csv"
df_raw = pd.read_csv(data_file, sep=";", parse_dates=["temps"], index_col="temps")
df = df_raw[["exponential"]]
X_latent_level = df_raw[["X_EL"]]
X_latent_trend = df_raw[["X_ET"]]
X_scale = df_raw[["X_A"]]
X_local_level = df_raw[["X_local_level"]]
X_local_trend = df_raw[["X_local_trend"]]

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=1,
    output_col=output_col,
    standardization=False,
)

train_data, _, _, all_data = data_processor.get_splits()

# Components
sigma_v = np.sqrt(0.01)
exponential = Exponential(
    mu_states=[0, 0.3, 10.5, 0, 0],
    var_states=[0.2**2, 0.1**2, 0.5**2, 0, 0],
)
noise = WhiteNoise(std_error=sigma_v)
localtrend = LocalTrend(
    mu_states=[1.95, -0.00], var_states=[0.1**2, 0.02**2], std_error=0
)

# Model
model = Model(exponential, noise, localtrend)
model.filter(data=all_data)
model.smoother(matrix_inversion_tol=1e-12)

# Plot Prior
fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="prior",
)

ax[model.get_states_index("latent level")].plot(
    df_raw.index, X_latent_level, color="black", linestyle="--"
)

ax[model.get_states_index("latent trend")].plot(
    df_raw.index, X_latent_trend, color="black", linestyle="--"
)
ax[model.get_states_index("scale")].plot(
    df_raw.index, X_scale, color="black", linestyle="--"
)

ax[model.get_states_index("level")].plot(
    df_raw.index, X_local_level, color="black", linestyle="--"
)

ax[model.get_states_index("trend")].plot(
    df_raw.index, X_local_trend, color="black", linestyle="--"
)

# Plot posterior
fig, ax = plot_states(
    data_processor=data_processor, states=model.states, states_type="posterior"
)

ax[model.get_states_index("latent level")].plot(
    df_raw.index, X_latent_level, color="black", linestyle="--"
)

ax[model.get_states_index("latent trend")].plot(
    df_raw.index, X_latent_trend, color="black", linestyle="--"
)
ax[model.get_states_index("scale")].plot(
    df_raw.index, X_scale, color="black", linestyle="--"
)

ax[model.get_states_index("level")].plot(
    df_raw.index, X_local_level, color="black", linestyle="--"
)

ax[model.get_states_index("trend")].plot(
    df_raw.index, X_local_trend, color="black", linestyle="--"
)

# Plot smooth
fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="smooth",
    states_to_plot=(
        "latent level",
        "latent trend",
        "scale",
        "exp",
        "scaled exp",
        "level",
        "trend",
    ),
)

ax[model.get_states_index("latent level")].plot(
    df_raw.index, X_latent_level, color="black", linestyle="--"
)

ax[model.get_states_index("latent trend")].plot(
    df_raw.index, X_latent_trend, color="black", linestyle="--"
)
ax[model.get_states_index("scale")].plot(
    df_raw.index, X_scale, color="black", linestyle="--"
)

ax[-2].plot(df_raw.index, X_local_level, color="black", linestyle="--")
ax[-1].plot(df_raw.index, X_local_trend, color="black", linestyle="--")


scaled_exp_index = model.get_states_index("scaled exp")
level_index = model.get_states_index("level")
cov_scaled_exp_level = []
for i in range(len(model.states.get_mean("level", "smooth"))):
    cov_scaled_exp_level.append(
        model.states.var_smooth[i][scaled_exp_index, level_index]
    )
cov_scaled_exp_level = np.array(cov_scaled_exp_level)

# Plot scaled exp add to local level
plt.figure(figsize=(10, 8))
plt.plot(
    df_raw.index,
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth"),
    color="purple",
)
plt.fill_between(
    data_processor.get_time("all"),
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth")
    + np.sqrt(
        model.states.get_std("scaled exp", "smooth") ** 2
        + model.states.get_std("level", "smooth") ** 2
        + 2 * (cov_scaled_exp_level)
    ),
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth")
    - np.sqrt(
        model.states.get_std("scaled exp", "smooth") ** 2
        + model.states.get_std("level", "smooth") ** 2
        + 2 * (cov_scaled_exp_level)
    ),
    color="purple",
    alpha=0.4,
)

plt.scatter(df_raw.index, data_processor.get_data("all"), color="red", s=2.5)

plt.show()
