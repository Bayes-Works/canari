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
    plot_data,
    plot_prediction,
    plot_states,
    plot_skf_states,
)
from canari.component import (
    Exponential,
    WhiteNoise,
    Periodic,
    LocalAcceleration,
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
    # "/Users/michelwu/Desktop/Exponential component/0590P073.CSV",
    "/Users/michelwu/Desktop/Exponential component/LTU014PIAEVA920.CSV",
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
df = df.resample("D").mean()

output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=1,
    validation_split=0,
    output_col=output_col,
    standardization=False,
)
train_data, validation_data, _, all_data = data_processor.get_splits()

sigma_v = np.sqrt(0.45)
whitenoise = WhiteNoise(sigma_v)

periodic = Periodic(period=365.25, mu_states=[0, 0], var_states=[1, 1], std_error=0)
localtrend = LocalTrend(
    mu_states=[-3, -0.002], var_states=[1**2, 0.0005**2], std_error=0
)
local_acceleration = LocalAcceleration()

# Normal model
model = Model(localtrend, periodic, whitenoise)

#  Abnormal model
ab_model = Model(local_acceleration, periodic, whitenoise)

skf = SKF(
    norm_model=model,
    abnorm_model=ab_model,
    std_transition_error=1e-4,
    norm_to_abnorm_prob=1e-4,
)

filter_marginal_abnorm_prob, states = skf.filter(data=all_data)
smooth_marginal_abnorm_prob, states = skf.smoother()

fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=states,
    model_prob=filter_marginal_abnorm_prob,
    legend_location="upper left",
)
fig.suptitle("SKF hidden states", fontsize=10, y=1)
ax[-1].set_xlabel("MM-DD")
plt.show()

df1 = df.copy()
df2 = df.copy()
df3 = df.copy()
df1 = df1.iloc[0:2741]
df1 = df1.resample("D").mean()
df2 = df2.iloc[2741:5115]
df2 = df2.resample("D").mean()
df3 = df3.iloc[5115:]
df3 = df3.resample("D").mean()


output_col = [0]
data_processor1 = DataProcess(
    data=df1,
    train_split=0.4,
    validation_split=0.6,
    output_col=output_col,
    standardization=False,
)
train_data1, validation_data1, _, all_data1 = data_processor1.get_splits()

# model localtrend seulement

localtrend11 = LocalTrend(
    mu_states=[-3, -0.002], var_states=[1**2, 0.0005**2], std_error=0
)

# model expo+localtrend

exponential1 = Exponential(
    # std_error=0.0,
    mu_states=[0, 0.003, 10.5, 0, 0],
    var_states=[0.0001**2, 0.001**2, 0.5**2, 0, 0],
)
localtrend12 = LocalTrend(
    mu_states=[-3, -0.00005], var_states=[0.2**2, 0.00005**2], std_error=0
)

train_data2, validation_data2, _, all_data2 = data_processor.get_splits()


model1 = Model(localtrend11, whitenoise, periodic)
mu_train_pred, std_train_pred, states = model1.filter(data=train_data1)
model1.smoother(matrix_inversion_tol=1e-12)
mu_val_pred, std_val_pred, states = model1.forecast(data=validation_data1)

fig, ax = plot_states(
    data_processor=data_processor1,
    states=states,
    states_to_plot=["level"],
    states_type="smooth",
)
plot_data(
    data_processor=data_processor1,
    plot_train_data=True,
    plot_test_data=False,
    plot_validation_data=True,
    sub_plot=ax[0],
)
plot_prediction(
    data_processor=data_processor1,
    mean_validation_pred=mu_val_pred,
    std_validation_pred=std_val_pred,
    sub_plot=ax[0],
    color="k",
)
ax[0].scatter(
    data_processor1.get_time("all"), data_processor1.get_data("all"), color="red", s=2.5
)
plt.savefig(f"LTU1lin.pgf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()
model2 = Model(localtrend12, whitenoise, periodic, exponential1)
mu_train_pred, std_train_pred, states = model2.filter(data=train_data1)
model2.smoother(matrix_inversion_tol=1e-12)
mu_val_pred, std_val_pred, states = model2.forecast(data=validation_data1)

fig, ax = plot_states(
    data_processor=data_processor1,
    states=states,
    states_to_plot=["level"],
    states_type="smooth",
)

ax[-1].plot(
    df1.index,
    model2.states.get_mean("scaled exp", "smooth")
    + model2.states.get_mean("level", "smooth"),
    color="purple",
)
scaled_exp_index = model2.get_states_index("scaled exp")
level_index = model2.get_states_index("level")

cov_scaled_exp_level = []

for i in range(len(model2.states.get_mean("level", "smooth"))):
    cov_scaled_exp_level.append(
        model2.states.var_smooth[i][scaled_exp_index, level_index]
    )

cov_scaled_exp_level = np.array(cov_scaled_exp_level)

plt.fill_between(
    df1.index,
    model2.states.get_mean("scaled exp", "smooth")
    + model2.states.get_mean("level", "smooth")
    + np.sqrt(
        model2.states.get_std("scaled exp", "smooth") ** 2
        + model2.states.get_std("level", "smooth") ** 2
        + 2 * (cov_scaled_exp_level)
    ),
    model2.states.get_mean("scaled exp", "smooth")
    + model2.states.get_mean("level", "smooth")
    - np.sqrt(
        model2.states.get_std("scaled exp", "smooth") ** 2
        + model2.states.get_std("level", "smooth") ** 2
        + 2 * (cov_scaled_exp_level)
    ),
    color="purple",
    alpha=0.2,
)


plot_data(
    data_processor=data_processor1,
    plot_train_data=True,
    plot_test_data=False,
    plot_validation_data=True,
    sub_plot=ax[-1],
)
plot_prediction(
    data_processor=data_processor1,
    mean_validation_pred=mu_val_pred,
    std_validation_pred=std_val_pred,
    sub_plot=ax[-1],
    color="k",
)
ax[-1].scatter(
    data_processor1.get_time("all"), data_processor1.get_data("all"), color="red", s=2.5
)

plt.savefig(f"LTU1explin.pgf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()

data_processor2 = DataProcess(
    data=df2,
    train_split=0.4,
    validation_split=0.6,
    output_col=output_col,
    standardization=False,
)

train_data2, validation_data2, _, all_data2 = data_processor2.get_splits()

# model localtrend seulement

localtrend21 = LocalTrend(
    mu_states=[-12, -0.003], var_states=[1**2, 0.00005**2], std_error=0
)

# model expo+localtrend

exponential2 = Exponential(
    # std_error=0.0,
    mu_states=[0, 0.004, 10, 0, 0],
    var_states=[0.0001**2, 0.002**2, 0.5**2, 0, 0],
)
localtrend22 = LocalTrend(
    mu_states=[-12, -0.0005], var_states=[1**2, 0.00005**2], std_error=0
)

train_data2, validation_data2, _, all_data2 = data_processor2.get_splits()


model21 = Model(localtrend21, whitenoise, periodic)
mu_train_pred, std_train_pred, states = model21.filter(data=train_data2)
model21.smoother(matrix_inversion_tol=1e-12)
mu_val_pred, std_val_pred, states = model21.forecast(data=validation_data2)

fig, ax = plot_states(
    data_processor=data_processor2,
    states=states,
    states_to_plot=["level"],
    states_type="smooth",
)
plot_data(
    data_processor=data_processor2,
    plot_train_data=True,
    plot_test_data=False,
    plot_validation_data=True,
    sub_plot=ax[0],
)
plot_prediction(
    data_processor=data_processor2,
    mean_validation_pred=mu_val_pred,
    std_validation_pred=std_val_pred,
    sub_plot=ax[0],
    color="k",
)
ax[0].scatter(
    data_processor2.get_time("all"), data_processor2.get_data("all"), color="red", s=2.5
)
plt.savefig(f"LTU2lin.pgf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()
model22 = Model(localtrend22, whitenoise, periodic, exponential2)
mu_train_pred, std_train_pred, states = model22.filter(data=train_data2)
model22.smoother(matrix_inversion_tol=1e-12)
mu_val_pred, std_val_pred, states = model22.forecast(data=validation_data2)

# fig, ax = plt.subplots(figsize=(10, 8))

fig, ax = plot_states(
    data_processor=data_processor2,
    states=states,
    states_to_plot=["level"],
    states_type="smooth",
)
ax[-1].plot(
    df2.index,
    model22.states.get_mean("scaled exp", "smooth")
    + model22.states.get_mean("level", "smooth"),
    color="purple",
)
scaled_exp_index = model22.get_states_index("scaled exp")
level_index = model22.get_states_index("level")

cov_scaled_exp_level = []

for i in range(len(model22.states.get_mean("level", "smooth"))):
    cov_scaled_exp_level.append(
        model22.states.var_smooth[i][scaled_exp_index, level_index]
    )

cov_scaled_exp_level = np.array(cov_scaled_exp_level)

plt.fill_between(
    df2.index,
    model22.states.get_mean("scaled exp", "smooth")
    + model22.states.get_mean("level", "smooth")
    + np.sqrt(
        model22.states.get_std("scaled exp", "smooth") ** 2
        + model22.states.get_std("level", "smooth") ** 2
        + 2 * (cov_scaled_exp_level)
    ),
    model22.states.get_mean("scaled exp", "smooth")
    + model22.states.get_mean("level", "smooth")
    - np.sqrt(
        model22.states.get_std("scaled exp", "smooth") ** 2
        + model22.states.get_std("level", "smooth") ** 2
        + 2 * (cov_scaled_exp_level)
    ),
    color="purple",
    alpha=0.2,
)

plot_data(
    data_processor=data_processor2,
    plot_train_data=True,
    plot_test_data=False,
    plot_validation_data=True,
    sub_plot=ax[-1],
)
plot_prediction(
    data_processor=data_processor2,
    mean_validation_pred=mu_val_pred,
    std_validation_pred=std_val_pred,
    sub_plot=ax[-1],
    color="k",
)
ax[-1].scatter(
    data_processor2.get_time("all"), data_processor2.get_data("all"), color="red", s=2.5
)
plt.savefig(f"LTU2explin.pgf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()
# plt.show()

data_processor3 = DataProcess(
    data=df3,
    train_split=0.4,
    validation_split=0.6,
    output_col=output_col,
    standardization=False,
)
train_data3, validation_data3, _, all_data3 = data_processor3.get_splits()

localtrend31 = LocalTrend(
    mu_states=[-21, -0.002], var_states=[1**2, 0.001**2], std_error=0
)

# model expo+localtrend

exponential3 = Exponential(
    # std_error=0.0,
    mu_states=[0, 0.0035, 11, 0, 0],
    var_states=[0.0001**2, 0.001**2, 0.5**2, 0, 0],
)
localtrend32 = LocalTrend(
    mu_states=[-21, -0.002], var_states=[1**2, 0.0001**2], std_error=0
)

train_data3, validation_data3, _, all_data3 = data_processor3.get_splits()


model31 = Model(localtrend31, whitenoise, periodic)
mu_train_pred, std_train_pred, states = model31.filter(data=train_data3)
model31.smoother(matrix_inversion_tol=1e-12)
mu_val_pred, std_val_pred, states = model31.forecast(data=validation_data3)

fig, ax = plot_states(
    data_processor=data_processor3,
    states=states,
    states_to_plot=["level"],
    states_type="smooth",
)
plot_data(
    data_processor=data_processor3,
    plot_train_data=True,
    plot_test_data=False,
    plot_validation_data=True,
    sub_plot=ax[0],
)
plot_prediction(
    data_processor=data_processor3,
    mean_validation_pred=mu_val_pred,
    std_validation_pred=std_val_pred,
    sub_plot=ax[0],
    color="k",
)
ax[0].scatter(
    data_processor3.get_time("all"), data_processor3.get_data("all"), color="red", s=2.5
)
plt.savefig(f"LTU3lin.pgf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()
model32 = Model(localtrend32, whitenoise, periodic, exponential3)
mu_train_pred, std_train_pred, states = model32.filter(data=train_data3)
model32.smoother(matrix_inversion_tol=1e-15)
mu_val_pred, std_val_pred, states = model32.forecast(data=validation_data3)

# fig, ax = plt.subplots(figsize=(12, 10))

fig, ax = plot_states(
    data_processor=data_processor3,
    states=states,
    states_to_plot=["level"],
    states_type="smooth",
)
ax[-1].plot(
    df3.index,
    model32.states.get_mean("scaled exp", "smooth")
    + model32.states.get_mean("level", "smooth"),
    color="purple",
)
scaled_exp_index = model32.get_states_index("scaled exp")
level_index = model32.get_states_index("level")

cov_scaled_exp_level = []

for i in range(len(model32.states.get_mean("level", "smooth"))):
    cov_scaled_exp_level.append(
        model32.states.var_smooth[i][scaled_exp_index, level_index]
    )

cov_scaled_exp_level = np.array(cov_scaled_exp_level)

plt.fill_between(
    df3.index,
    model32.states.get_mean("scaled exp", "smooth")
    + model32.states.get_mean("level", "smooth")
    + np.sqrt(
        model32.states.get_std("scaled exp", "smooth") ** 2
        + model32.states.get_std("level", "smooth") ** 2
        + 2 * (cov_scaled_exp_level)
    ),
    model32.states.get_mean("scaled exp", "smooth")
    + model32.states.get_mean("level", "smooth")
    - np.sqrt(
        model32.states.get_std("scaled exp", "smooth") ** 2
        + model32.states.get_std("level", "smooth") ** 2
        + 2 * (cov_scaled_exp_level)
    ),
    color="purple",
    alpha=0.2,
)

plot_data(
    data_processor=data_processor3,
    plot_train_data=True,
    plot_test_data=False,
    plot_validation_data=True,
    sub_plot=ax[-1],
)
plot_prediction(
    data_processor=data_processor3,
    mean_validation_pred=mu_val_pred,
    std_validation_pred=std_val_pred,
    sub_plot=ax[-1],
    color="k",
)
ax[-1].scatter(
    data_processor3.get_time("all"), data_processor3.get_data("all"), color="red", s=2.5
)
plt.savefig(f"LTU3explin.pgf", bbox_inches="tight", pad_inches=0, transparent=True)

plt.show()

# plt.scatter(
#     data_processor1.get_time("all"), data_processor1.get_data("all"), color="red", s=2.5
# )
# plt.show()

# plt.scatter(
#     data_processor2.get_time("all"), data_processor2.get_data("all"), color="red", s=2.5
# )
# plt.show()

# plt.scatter(
#     data_processor3.get_time("all"), data_processor3.get_data("all"), color="red", s=2.5
# )
# plt.show()
