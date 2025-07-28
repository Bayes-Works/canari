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
from canari.component import Exponential, WhiteNoise, Periodic, LocalTrend

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
    }
)

df_raw = pd.read_csv(
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiquesavecobsettrend11.csv",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiques11.csv",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiques12.csv",
    "./data/toy_time_series/synthetic_exponential_localtrend.csv",
    sep=";",  # Semicolon as delimiter
    quotechar='"',  # Double quotes as text qualifier
    engine="python",  # Python engine for complex cases
    na_values=[""],  # Treat empty strings as NaN
    skipinitialspace=True,  # Skip spaces after delimiter
    encoding="ISO-8859-1",
    parse_dates=["temps"],
    index_col="temps",
)
df = df_raw[["exponential"]]
X_EL = df_raw[["X_EL"]]
X_ET = df_raw[["X_ET"]]
X_A = df_raw[["X_A"]]
X_local_level = df_raw[["X_local_level"]]
X_local_trend = df_raw[["X_local_trend"]]
df = df
df = df.iloc[:]

# Split into train and test
output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=1,
    validation_split=0,
    output_col=output_col,
    standardization=False,
)
train_data, validation_data, _, all_data = data_processor.get_splits()

sigma_v = np.sqrt(0.01)

exponential = Exponential(
    mu_states=[0, 0.3, 10.5, 0, 0],
    var_states=[0.2**2, 0.1**2, 0.5**2, 0, 0],
)
# exponential = Exponential(
#     mu_states=[0, 0.0125, 12.5, 0, 0],
#     var_states=[1e-8**2, 0.005**2, 2.5**2, 0, 0],
# )
noise = WhiteNoise(std_error=sigma_v)
periodic = Periodic(
    period=365.24, mu_states=[1.4, 0], var_states=[1e-1, 1e-3], std_error=0
)
# localtrend = LocalTrend(
#     mu_states=[1.95, -0.0], var_states=[0.1**2, 0.1**2], std_error=0
# )
localtrend = LocalTrend(
    mu_states=[1.95, -0.00], var_states=[0.1**2, 0.02**2], std_error=0
)
model = Model(exponential, noise, localtrend)

model.filter(data=all_data)
model.smoother(matrix_inversion_tol=1e-12)
# Plot
fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="prior",
)
# plot_data(
#     data_processor=data_processor,
#     plot_column=output_col,
#     standardization=True,
#     plot_test_data=False,
#     validation_label="y",
#     sub_plot=ax[4],
# )
ax[model.get_states_index("latent level")].plot(
    df_raw.index, X_EL, color="black", linestyle="--"
)

ax[model.get_states_index("latent trend")].plot(
    df_raw.index, X_ET, color="black", linestyle="--"
)
ax[model.get_states_index("scale")].plot(
    df_raw.index, X_A, color="black", linestyle="--"
)

# ax[model.get_states_index("level")].plot(
#     df_raw.index, X_local_level, color="black", linestyle="--"
# )

# ax[model.get_states_index("trend")].plot(
#     df_raw.index, X_local_trend, color="black", linestyle="--"
# )

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
ax[model.get_states_index("latent level")].plot(
    df_raw.index, X_EL, color="black", linestyle="--"
)

ax[model.get_states_index("latent trend")].plot(
    df_raw.index, X_ET, color="black", linestyle="--"
)
ax[model.get_states_index("scale")].plot(
    df_raw.index, X_A, color="black", linestyle="--"
)

# ax[model.get_states_index("scaled exp")].plot(
#     df_raw.index,
#     model.states.get_mean("scaled exp", "posterior")
#     # + model.states.get_mean("level", "posterior")
#     ,
#     color="purple",
# )

ax[model.get_states_index("level")].plot(
    df_raw.index, X_local_level, color="black", linestyle="--"
)

ax[model.get_states_index("trend")].plot(
    df_raw.index, X_local_trend, color="black", linestyle="--"
)


fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="smooth",
    states_to_plot=(
        "latent level",
        "latent trend",
        "scale",
        "level",
        "trend",
    ),
)
stored_ylims = [a.get_ylim() for a in ax.flatten()]

# plot_data(
#     data_processor=data_processor,
#     plot_column=output_col,
#     standardization=True,
#     plot_test_data=False,
#     validation_label="y",
#     sub_plot=ax[4],
# )
ax[model.get_states_index("latent level")].plot(
    df_raw.index, X_EL, color="black", linestyle="--"
)

ax[model.get_states_index("latent trend")].plot(
    df_raw.index, X_ET, color="black", linestyle="--"
)
ax[model.get_states_index("scale")].plot(
    df_raw.index, X_A, color="black", linestyle="--"
)

# ax[model.get_states_index("scaled exp")].plot(
#     df_raw.index,
#     model.states.get_mean("scaled exp", "smooth")
#     + model.states.get_mean("level", "smooth"),
#     color="purple",
# )

ax[-2].plot(df_raw.index, X_local_level, color="black", linestyle="--")

ax[-1].plot(df_raw.index, X_local_trend, color="black", linestyle="--")
# share x-ax

plt.savefig(
    f"toy_exp_smoother.pgf", bbox_inches="tight", pad_inches=0, transparent=True
)

scaled_exp_index = model.get_states_index("scaled exp")
level_index = model.get_states_index("level")
cov_scaled_exp_level = []
for i in range(len(model.states.get_mean("level", "smooth"))):
    cov_scaled_exp_level.append(
        model.states.var_smooth[i][scaled_exp_index, level_index]
    )

cov_scaled_exp_level = np.array(cov_scaled_exp_level)

plt.figure(figsize=(7, 3))
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
plt.savefig(
    f"toy_exp_comb_smooth.pgf", bbox_inches="tight", pad_inches=0, transparent=True
)
plt.show()
