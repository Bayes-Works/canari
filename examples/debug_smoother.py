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
from canari.component import ExpSimp, WhiteNoise, Periodic, LocalTrend

df_raw = pd.read_csv(
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiquesavecobsettrend11.csv",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiques11.csv",
    # "/Users/michelwu/Desktop/Exponential component/donnees_synthetiques12.csv",
    "/Users/michelwu/Desktop/Exponential component/donnees_synthetiquesfullexp.csv",
    # "./data/toy_time_series/synthetic_exponential_localtrend.csv",
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
# X_A = df_raw[["X_A"]]
# X_local_level = df_raw[["X_local_level"]]
# X_local_trend = df_raw[["X_local_trend"]]
df = df
df = df.iloc[:]

output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=1,
    validation_split=0,
    output_col=output_col,
    standardization=False,
)
train_data, validation_data, _, all_data = data_processor.get_splits()

sigma_v = np.sqrt(1e-3)
exponential = ExpSimp(
    mu_states=[0, 0.09, 0],
    var_states=[1e-8**2, 0.02**2, 0],
)
whitenoise = WhiteNoise(sigma_v)

model = Model(exponential, whitenoise)
model.filter(data=all_data)
model.smoother(matrix_inversion_tol=1e-12)

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
    sub_plot=ax[2],
)
ax[model.get_states_index("simp latent level")].plot(
    df_raw.index, X_EL, color="black", linestyle="--"
)

ax[model.get_states_index("simp latent trend")].plot(
    df_raw.index, X_ET, color="black", linestyle="--"
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
plot_data(
    data_processor=data_processor,
    plot_column=output_col,
    standardization=True,
    plot_test_data=False,
    validation_label="y",
    sub_plot=ax[2],
)
ax[model.get_states_index("simp latent level")].plot(
    df_raw.index, X_EL, color="black", linestyle="--"
)

ax[model.get_states_index("simp latent trend")].plot(
    df_raw.index, X_ET, color="black", linestyle="--"
)

fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="smooth",
)

ax[model.get_states_index("simp latent level")].plot(
    df_raw.index, X_EL, color="black", linestyle="--"
)

ax[model.get_states_index("simp latent trend")].plot(
    df_raw.index, X_ET, color="black", linestyle="--"
)

plt.show()
# ax[model.get_states_index("level")].plot(
#     df_raw.index, X_local_level, color="black", linestyle="--"
# )

# ax[model.get_states_index("trend")].plot(
#     df_raw.index, X_local_trend, color="black", linestyle="--"
# )
