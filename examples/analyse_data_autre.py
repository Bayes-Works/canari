import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari import (
    DataProcess,
    Model,
    plot_states,
)
from canari.component import (
    Exponential,
    WhiteNoise,
    LocalTrend,
    Periodic,
    LocalLevel,
    Autoregression,
)

df_raw = pd.read_csv(
    "/Users/michelwu/Desktop/Exp DAT/LTU012PIMP-E010.DAT",
    sep=";",  # Semicolon as delimiter
    quotechar='"',  # Double quotes as text qualifier
    engine="python",  # Python engine for complex cases
    na_values=[""],  # Treat empty strings as NaN
    skipinitialspace=True,  # Skip spaces after delimiter
    encoding="ISO-8859-1",
    parse_dates=["Date"],
    index_col="Date",
)
df = df_raw[["Deplacements cumulatif X (mm)"]]
df = df.iloc[:]
print(df)
df = df.resample("M").mean()
# fig = plt.subplots(figsize=(12, 3))
# plt.scatter(df.index, df.values, color="r")
# plt.title("Orginal data")
# plt.show()

output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=1,
    output_col=output_col,
    standardization=False,
)

train_data, _, _, all_data = data_processor.get_splits()

sigma_v = np.sqrt(0.001)
exponential = Exponential(
    mu_states=[0, 0.006, 6, 0, 0],
    var_states=[0.00001**2, 0.005**2, 1**2, 0, 0],
)
noise = WhiteNoise(std_error=sigma_v)
localtrend = LocalTrend(mu_states=[2, -0.01], var_states=[0.5**2, 0.01**2], std_error=0)
locallevel = LocalLevel(mu_states=[0], var_states=[0.5**2], std_error=0)
periodic = Periodic(mu_states=[0, 0], var_states=[0.5**2, 0.5**2], period=12)

var_W2bar_prior = 0.05
AR_process_error_var_prior = 0.2
ar = Autoregression(
    phi=0,
    mu_states=[0, 0, 0, AR_process_error_var_prior],
    var_states=[0, 0, 1e-6, var_W2bar_prior],
)
model = Model(exponential, ar, periodic, locallevel)
model.filter(data=all_data)
model.smoother(matrix_inversion_tol=1e-12)

# Plot Prior
fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="prior",
)

# Plot posterior
fig, ax = plot_states(
    data_processor=data_processor, states=model.states, states_type="posterior"
)
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

scaled_exp_index = model.get_states_index("scaled exp")
level_index = model.get_states_index("level")
periodic_index = model.get_states_index("periodic 1")
cov_scaled_exp_level = []
cov_scaled_exp_periodic = []
cov_level_periodic = []
cov_level_auto = []
cov_scaled_exp_auto = []
cov_periodic_auto = []
for i in range(len(model.states.get_mean("level", "smooth"))):
    cov_scaled_exp_level.append(
        model.states.var_smooth[i][scaled_exp_index, level_index]
    )
    cov_scaled_exp_periodic.append(
        model.states.var_smooth[i][scaled_exp_index, periodic_index]
    )
    cov_level_periodic.append(model.states.var_smooth[i][level_index, periodic_index])
    cov_level_auto.append(
        model.states.var_smooth[i][
            level_index, model.get_states_index("autoregression")
        ]
    )
    cov_scaled_exp_auto.append(
        model.states.var_smooth[i][
            scaled_exp_index, model.get_states_index("autoregression")
        ]
    )
    cov_periodic_auto.append(
        model.states.var_smooth[i][
            periodic_index, model.get_states_index("autoregression")
        ]
    )
cov_scaled_exp_level = np.array(cov_scaled_exp_level)
cov_scaled_exp_periodic = np.array(cov_scaled_exp_periodic)
cov_level_periodic = np.array(cov_level_periodic)
cov_level_auto = np.array(cov_level_auto)
cov_scaled_exp_auto = np.array(cov_scaled_exp_auto)
cov_periodic_auto = np.array(cov_periodic_auto)

# Plot scaled exp add to local level
plt.figure(figsize=(10, 8))
plt.plot(
    df.index,
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth")
    + model.states.get_mean("periodic 1", "smooth")
    # + model.states.get_mean("autoregression", "smooth")
    ,
    color="purple",
)
plt.fill_between(
    data_processor.get_time("all"),
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth")
    + model.states.get_mean("periodic 1", "smooth")
    # + model.states.get_mean("autoregression", "smooth")
    + np.sqrt(
        model.states.get_std("scaled exp", "smooth") ** 2
        + model.states.get_std("level", "smooth") ** 2
        + model.states.get_std("periodic 1", "smooth") ** 2
        # + model.states.get_std("autoregression", "smooth") ** 2
        + 2
        * (
            cov_scaled_exp_level
            + cov_scaled_exp_periodic
            + cov_level_periodic
            # + cov_level_auto
            # + cov_scaled_exp_auto
            # + cov_periodic_auto
        )
    ),
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth")
    + model.states.get_mean("periodic 1", "smooth")
    # + model.states.get_mean("autoregression", "smooth")
    - np.sqrt(
        model.states.get_std("scaled exp", "smooth") ** 2
        + model.states.get_std("level", "smooth") ** 2
        + model.states.get_std("periodic 1", "smooth") ** 2
        # + model.states.get_std("autoregression", "smooth") ** 2
        + 2
        * (
            cov_scaled_exp_level
            + cov_scaled_exp_periodic
            + cov_level_periodic
            # + cov_level_auto
            # + cov_scaled_exp_auto
            # + cov_periodic_auto
        )
    ),
    color="purple",
    alpha=0.3,
)

plt.scatter(df.index, data_processor.get_data("all"), color="red", s=1)

plt.show()
