import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari import (
    DataProcess,
    Model,
    plot_states,
)
from canari.component import Exponential, WhiteNoise, LocalTrend,LocalLevel, Periodic

# Read data
data_file = '/Users/michelwu/Desktop/Exponential component/donnes_synth_exponentiel_sinus_level.csv'
df_raw = pd.read_csv(data_file, sep=";", parse_dates=["temps"], index_col="temps")
df = df_raw[["exponential"]]
X_latent_level = df_raw[["X_EL"]]
X_latent_trend = df_raw[["X_ET"]]
X_exp_scale_factor = df_raw[["X_A"]]
X_local_level = df_raw[["X_local_level"]]
# X_local_trend = df_raw[["X_local_trend"]]

output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=1,
    output_col=output_col,
    standardization=False,
)

train_data, _, _, all_data = data_processor.get_splits()
df_train=pd.DataFrame(index=train_data["time"], data={'y':train_data["y"].flatten()})
df_train.index.name="date"

exponential = Exponential()
noise = WhiteNoise()
# localtrend = LocalTrend()
locallevel=LocalLevel()
periodic = Periodic(period=52)
print(all_data["x"])
# Model
model = Model(exponential, noise, periodic, locallevel)
model.auto_initialize_comp(data_training=df_train, ratio_training=0.6)
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
ax[model.get_states_index("exp scale factor")].plot(
    df_raw.index, X_exp_scale_factor, color="black", linestyle="--"
)

ax[model.get_states_index("level")].plot(
    df_raw.index, X_local_level, color="black", linestyle="--"
)

# ax[model.get_states_index("trend")].plot(
#     df_raw.index, X_local_trend, color="black", linestyle="--"
# )

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
ax[model.get_states_index("exp scale factor")].plot(
    df_raw.index, X_exp_scale_factor, color="black", linestyle="--"
)

ax[model.get_states_index("level")].plot(
    df_raw.index, X_local_level, color="black", linestyle="--"
)

# ax[model.get_states_index("trend")].plot(
#     df_raw.index, X_local_trend, color="black", linestyle="--"
# )

# Plot smooth
fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="smooth",
    # states_to_plot=(
    #     "latent level",
    #     "latent trend",
    #     "exp scale factor",
    #     "exp",
    #     "scaled exp",
    #     "level",
    #     "trend",
    # ),
)

ax[model.get_states_index("latent level")].plot(
    df_raw.index, X_latent_level, color="black", linestyle="--"
)

ax[model.get_states_index("latent trend")].plot(
    df_raw.index, X_latent_trend, color="black", linestyle="--"
)
ax[model.get_states_index("exp scale factor")].plot(
    df_raw.index, X_exp_scale_factor, color="black", linestyle="--"
)

ax[model.get_states_index("level")].plot(df_raw.index, X_local_level, color="black", linestyle="--")
# ax[-1].plot(df_raw.index, X_local_trend, color="black", linestyle="--")


scaled_exp_index = model.get_states_index("scaled exp")
level_index = model.get_states_index("level")
periodic_index = model.get_states_index("periodic 1")
cov_scaled_exp_level = []
cov_scaled_exp_periodic = []
cov_level_periodic = []
for i in range(len(model.states.get_mean(states_name="level", states_type="smooth"))):
    cov_scaled_exp_level.append(
        model.states.var_smooth[i][scaled_exp_index, level_index]
    )
    cov_scaled_exp_periodic.append(
        model.states.var_smooth[i][scaled_exp_index, periodic_index]
    )
    cov_level_periodic.append(model.states.var_smooth[i][level_index, periodic_index])
cov_scaled_exp_level = np.array(cov_scaled_exp_level)
cov_scaled_exp_periodic = np.array(cov_scaled_exp_periodic)
cov_level_periodic = np.array(cov_level_periodic)

# Plot scaled exp add to local level
plt.figure(figsize=(10, 8))
plt.plot(
    df_raw.index,
    model.states.get_mean(states_name="scaled exp", states_type="smooth")
    + model.states.get_mean(states_name="level", states_type="smooth")
    + model.states.get_mean(states_name="periodic 1", states_type="smooth"),
    color="purple",
)
plt.fill_between(
    data_processor.get_time("all"),
    model.states.get_mean(states_name="scaled exp", states_type="smooth")
    + model.states.get_mean(states_name="level", states_type="smooth")
    + model.states.get_mean(states_name="periodic 1", states_type="smooth")
    + np.sqrt(
        model.states.get_std(states_name="scaled exp", states_type="smooth") ** 2
        + model.states.get_std(states_name="level", states_type="smooth") ** 2
        + model.states.get_std(states_name="periodic 1", states_type="smooth") ** 2
        + 2 * (cov_scaled_exp_level + cov_scaled_exp_periodic + cov_level_periodic)
    ),
    model.states.get_mean(states_name="scaled exp", states_type="smooth")
    + model.states.get_mean(states_name="level", states_type="smooth")
    + model.states.get_mean(states_name="periodic 1", states_type="smooth")
    - np.sqrt(
        model.states.get_std(states_name="scaled exp", states_type="smooth") ** 2
        + model.states.get_std(states_name="level", states_type="smooth") ** 2
        + 2 * (cov_scaled_exp_level + cov_scaled_exp_periodic + cov_level_periodic)
    ),
    color="purple",
    alpha=0.3,
)

plt.scatter(df_raw.index, data_processor.get_data("all"), color="red", s=1)

plt.show()
