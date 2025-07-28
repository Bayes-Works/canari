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

df_raw = pd.read_csv(
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiques9.CSV",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiquesavecobs7.CSV",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiquesavecobs9.CSV",
    "/Users/michelwu/Desktop/Exponential component/donnees_synthetiquesavecobsettrend9.CSV",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiquesavecobsettrend7.CSV",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiquesavecobsettrend8.CSV",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiquesavecobsetperiod7.CSV",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiques5.CSV",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiques_avecerreurobs2.CSV",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiques_avecperiodiqueeterreurobs.CSV",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiquesavecobsetperiod6.CSV",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiques_avectrendperiodiqueeterreurobs4.CSV",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiquesavecobsetperiodtrend6.CSV",
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
df = df.iloc[:10]

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

sigma_v = np.sqrt(0.3)
# sigma_v = np.sqrt(0.15)

exponential = Exponential(
    std_error=0.0,
    mu_states=[-0.2, 0.31, 11, 0, 0],
    var_states=[0.2**2, 0.1**2, 1**2, 0, 0],
)

# exponential = Exponential(
#     std_error=0.0,
#     mu_states=[0, 0.0010, 11.0, 0, 0],
#     var_states=[0.2**2, 0.0005**2, 1**2, 0, 0],
# )
noise = WhiteNoise(std_error=sigma_v)
periodic = Periodic(
    period=365.24, mu_states=[1.4, 0], var_states=[1e-1, 1e-3], std_error=0
)
# localtrend = LocalTrend(mu_states=[1.95, -0.0], var_states=[0.1, 1e-4], std_error=0)
# model = Model(exponential, noise)

localtrend = LocalTrend(
    mu_states=[1.95, -0.0], var_states=[0.1**2, 0.1**2], std_error=0
)
model = Model(exponential, noise, localtrend)

model.filter(data=all_data)
model.smoother()
print(model.states.get_mean("exp level", "prior"))
print(model.states.get_std("exp", "prior"))
print(model.states.get_mean("exp", "posterior"))
print(model.states.get_std("exp", "posterior"))
print(model.get_states_index("exp level"))

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
ax[model.get_states_index("exp level")].plot(
    df_raw.index, X_EL, color="black", linestyle="--"
)

ax[model.get_states_index("exp trend")].plot(
    df_raw.index, X_ET, color="black", linestyle="--"
)
ax[model.get_states_index("exp amplitude")].plot(
    df_raw.index, X_A, color="black", linestyle="--"
)

ax[model.get_states_index("level")].plot(
    df_raw.index, X_local_level, color="black", linestyle="--"
)

ax[model.get_states_index("trend")].plot(
    df_raw.index, X_local_trend, color="black", linestyle="--"
)

fig, ax = plot_states(
    data_processor=data_processor, states=model.states, states_type="posterior"
)
plot_data(
    data_processor=data_processor,
    plot_column=output_col,
    standardization=True,
    plot_test_data=False,
    validation_label="y",
    sub_plot=ax[4],
)
ax[model.get_states_index("exp level")].plot(
    df_raw.index, X_EL, color="black", linestyle="--"
)

ax[model.get_states_index("exp trend")].plot(
    df_raw.index, X_ET, color="black", linestyle="--"
)
ax[model.get_states_index("exp amplitude")].plot(
    df_raw.index, X_A, color="black", linestyle="--"
)

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
)
plot_data(
    data_processor=data_processor,
    plot_column=output_col,
    standardization=True,
    plot_test_data=False,
    validation_label="y",
    sub_plot=ax[4],
)
ax[model.get_states_index("exp level")].plot(
    df_raw.index, X_EL, color="black", linestyle="--"
)

ax[model.get_states_index("exp trend")].plot(
    df_raw.index, X_ET, color="black", linestyle="--"
)
ax[model.get_states_index("exp amplitude")].plot(
    df_raw.index, X_A, color="black", linestyle="--"
)

ax[model.get_states_index("level")].plot(
    df_raw.index, X_local_level, color="black", linestyle="--"
)

ax[model.get_states_index("trend")].plot(
    df_raw.index, X_local_trend, color="black", linestyle="--"
)

plt.show()
