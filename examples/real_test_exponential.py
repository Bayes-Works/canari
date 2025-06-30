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

df_raw = pd.read_csv(
    "/Users/michelwu/Desktop/Exponential component/2650F162.CSV",
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

locallevel = LocalLevel(mu_states=[1], var_states=[0.5], std_error=0)
exponential = Exponential(
    # std_error=0.0,
    mu_states=[0, 0.0028, 10.5, 0, 0],
    var_states=[0.1**2, 0.0001**2, 0.5**2, 0, 0],
)
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
periodic2 = Periodic(period=6, mu_states=[0, 0], var_states=[0.1, 0.1], std_error=0)

localtrend = LocalTrend(
    mu_states=[1, -0.003], var_states=[0.1**2, 0.0005**2], std_error=0
)
model = Model(exponential, ar, periodic, localtrend)

model.filter(data=all_data)
model.smoother(matrix_inversion_tol=1e-5)

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
    data_processor=data_processor, states=model.states, states_type="posterior"
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
        # "latent level",
        # "latent trend",
        # "scale",
        # "exp",
        "scaled exp",
        "periodic 1",
        "autoregression",
        "phi",
        "W2bar",
        "level",
        # "trend",
    ),
    states_type="smooth",
)

plot_data(
    data_processor=data_processor,
    plot_column=output_col,
    standardization=True,
    plot_test_data=False,
    validation_label="y",
    # sub_plot=ax[model.get_states_index("scaled exp")],
    sub_plot=ax[0],
)
# ax[model.get_states_index("scaled exp")].plot(
#     df.index,
#     model.states.get_mean("scaled exp", "smooth")
#     + model.states.get_mean("level", "smooth"),
#     color="purple",
# )

# ax[model.get_states_index("scaled exp")].scatter(
ax[0].scatter(
    data_processor.get_time("all"), data_processor.get_data("all"), color="red", s=2.5
)
plt.show()


plt.plot(
    data_processor.get_time("all"),
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth")
    + model.states.get_mean("periodic 1", "smooth"),
    color="purple",
)
plt.scatter(
    data_processor.get_time("all"), data_processor.get_data("all"), color="red", s=2.5
)
plt.show()
