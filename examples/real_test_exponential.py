import copy
import pandas as pd
from pytagi import Normalizer as normalizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytagi.metric as metric
from statsmodels.tsa.seasonal import seasonal_decompose
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_states,
    plot_data,
    plot_prediction,
    plot_skf_states,
)
from canari.component import (
    Exponential,
    WhiteNoise,
    Periodic,
    LocalTrend,
    Autoregression,
    LocalLevel,
)

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
    }
)


df_raw = pd.read_csv(
    # "/Users/michelwu/Desktop/Exponential component/2650F162.CSV",
    # "/Users/michelwu/Desktop/Exponential component/1700B042.CSV",
    "/Users/michelwu/Desktop/Exponential component/0590P073.CSV",
    sep=";",  # Semicolon as delimiter
    quotechar='"',  # Double quotes as text qualifier
    engine="python",  # Python engine for complex cases
    na_values=[""],  # Treat empty strings as NaN
    skipinitialspace=True,  # Skip spaces after delimiter
    encoding="ISO-8859-1",
    parse_dates=["Date"],
    index_col="Date",
)

# Resample
df = df_raw[["Deplacements cumulatif X (mm)"]]
df = df
df = df.iloc[:]
df = df.resample("M").mean()

output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=1,
    validation_split=0,
    output_col=output_col,
    standardization=False,
)
train_data, validation_data, _, all_data = data_processor.get_splits()

sigma_v = np.sqrt(0.1)

locallevel = LocalLevel(mu_states=[-3], var_states=[0.5], std_error=0)
AR_process_error_var_prior = 0.5
var_W2bar_prior = 1
AR_process_error_var_prior = 1
var_W2bar_prior = 1
ar = Autoregression(
    mu_states=[-0.1, 0.7, 0, 0, 0, AR_process_error_var_prior],
    var_states=[
        6.36e-05,
        0.25,
        0,
        AR_process_error_var_prior,
        1e-6,
        var_W2bar_prior,
    ],
)

noise = WhiteNoise(std_error=sigma_v)
periodic = Periodic(period=12, mu_states=[0, 0], var_states=[2, 2], std_error=0)
periodic2 = Periodic(period=6, mu_states=[0, 0], var_states=[0.5, 0.5], std_error=0)
# exponential = Exponential(
#     # std_error=0.0,
#     mu_states=[0, 0.0028, 10.5, 0, 0],
#     var_states=[0.1**2, 0.0001**2, 0.5**2, 0, 0],
# )
# localtrend = LocalTrend(
#     mu_states=[1, -0.005], var_states=[0.2**2, 0.0005**2], std_error=0
# )

# exponential = Exponential(
#     # std_error=0.0,
#     mu_states=[0, 0.0025, 10.5, 0, 0],
#     var_states=[0.01**2, 0.0005**2, 1**2, 0, 0],
# )
# localtrend = LocalTrend(
#     mu_states=[2, -0.005], var_states=[0.2**2, 0.005**2], std_error=0
# )

exponential = Exponential(
    # std_error=0.0,
    mu_states=[0, 0.0025, 10.5, 0, 0],
    var_states=[0.000001**2, 0.0015**2, 2**2, 0, 0],
)
localtrend = LocalTrend(
    mu_states=[2, -0.005], var_states=[1**2, 0.0025**2], std_error=0
)

model = Model(exponential, ar, periodic, localtrend)

model.filter(data=all_data)
model.smoother(matrix_inversion_tol=1e-12)

# Plot
fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="prior",
)
plot_data(
    data_processor=data_processor,
    plot_column=output_col,
    standardization=True,
    plot_test_data=False,
    validation_label="y",
    sub_plot=ax[4],
)


fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="posterior",
    states_to_plot=(
        "latent level",
        "latent trend",
        "scale",
        "exp",
        "scaled exp",
        "periodic 1",
        "autoregression",
        "phi",
        "W2bar",
        "level",
        "trend",
    ),
)
# plot_data(
#     data_processor=data_processor,
#     plot_column=output_col,
#     standardization=True,
#     plot_test_data=False,
#     validation_label="y",
#     sub_plot=ax[4],
# )
# ax[model.get_states_index("exp with amplitude")].plot(
#     df_raw.index,
#     model.states.get_mean("exp with amplitude", "posterior")
#     + model.states.get_mean("level", "posterior"),
#     color="purple",
# )

ax[model.get_states_index("scaled exp")].scatter(
    data_processor.get_time("all"), data_processor.get_data("all"), color="red", s=2.5
)

fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_to_plot=(
        "latent level",
        "latent trend",
        "scale",
        # "exp",
        # "scaled exp",
        "periodic 1",
        "autoregression",
        # "phi",
        # "W2bar",
        "level",
        "trend",
    ),
    states_type="smooth",
)

# plot_data(
#     data_processor=data_processor,
#     plot_column=output_col,
#     standardization=True,
#     plot_test_data=False,
#     validation_label="y",
#     # sub_plot=ax[model.get_states_index("scaled exp")],
#     sub_plot=ax[4],
# )
# ax[model.get_states_index("scaled exp")].plot(
#     df.index,
#     model.states.get_mean("scaled exp", "smooth")
#     + model.states.get_mean("level", "smooth"),
#     color="purple",
# )

# ax[model.get_states_index("scaled exp")].scatter(
# ax[4].scatter(
#     data_processor.get_time("all"), data_processor.get_data("all"), color="red", s=2.5
# )
plt.savefig(
    f"real_exp_smoother.pgf", bbox_inches="tight", pad_inches=0, transparent=True
)
plt.show()

plt.figure(figsize=(7, 3))
plt.plot(
    data_processor.get_time("all"),
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth")
    + model.states.get_mean("periodic 1", "smooth"),
    color="purple",
)
scaled_exp_index = model.get_states_index("scaled exp")
level_index = model.get_states_index("level")
periodic_index = model.get_states_index("periodic 1")

cov_scaled_exp_level = []
cov_level_periodic = []
cov_scaled_exp_periodic = []

for i in range(len(model.states.get_mean("level", "smooth"))):
    cov_scaled_exp_level.append(
        model.states.var_smooth[i][scaled_exp_index, level_index]
    )
    cov_level_periodic.append(model.states.var_smooth[i][periodic_index, level_index])
    cov_scaled_exp_periodic.append(
        model.states.var_smooth[i][scaled_exp_index, periodic_index]
    )

cov_level_periodic = np.array(cov_level_periodic)
cov_scaled_exp_level = np.array(cov_scaled_exp_level)
cov_scaled_exp_periodic = np.array(cov_scaled_exp_periodic)


plt.fill_between(
    data_processor.get_time("all"),
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth")
    + model.states.get_mean("periodic 1", "smooth")
    + np.sqrt(
        model.states.get_std("scaled exp", "smooth") ** 2
        + model.states.get_std("level", "smooth") ** 2
        + model.states.get_std("periodic 1", "smooth") ** 2
        + 2 * (cov_scaled_exp_level + cov_level_periodic + cov_scaled_exp_periodic)
    ),
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth")
    + model.states.get_mean("periodic 1", "smooth")
    - np.sqrt(
        model.states.get_std("scaled exp", "smooth") ** 2
        + model.states.get_std("level", "smooth") ** 2
        + model.states.get_std("periodic 1", "smooth") ** 2
        + 2 * (cov_scaled_exp_level + cov_level_periodic + cov_scaled_exp_periodic)
    ),
    color="purple",
    alpha=0.2,
)


# print(model.states.var_smooth[2])
# print(model.states.get_mean("scaled exp", "smooth").dtype)

plt.scatter(
    data_processor.get_time("all"), data_processor.get_data("all"), color="red", s=2.5
)
plt.savefig(
    f"real_exp_comb_smooth.pgf", bbox_inches="tight", pad_inches=0, transparent=True
)
plt.show()

scale = model.states.get_mean("scale", "smooth")[-1]
xet = model.states.get_mean("latent trend", "smooth")[-1]
lvl = model.states.get_mean("level", "smooth")[0]
trd = model.states.get_mean("trend", "smooth")[-1]
exp = []

for i in range(len(data_processor.get_data("all"))):
    exp.append(scale * (np.exp(-xet * i) - 1) + lvl + trd * i)

plt.plot(data_processor.get_time("all"), exp, color="blue")
plt.scatter(
    data_processor.get_time("all"), data_processor.get_data("all"), color="red", s=2.5
)

plt.show()
