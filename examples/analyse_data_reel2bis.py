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
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_states,
    plot_data,
    plot_prediction,
    plot_skf_states,
)

df_raw = pd.read_csv(
    "/Users/michelwu/Desktop/Exp DAT/GMR1975M106A.DAT",
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
print(df)
df = df.iloc[:7500]
df = df.resample("D").mean()
# fig = plt.subplots(figsize=(12, 3))
# plt.scatter(df.index, df.values, color="r")
# plt.title("Orginal data")
# plt.show()

output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=0.6,
    validation_split=0.4,
    output_col=output_col,
    standardization=False,
)

train_data, validation_data, _, all_data = data_processor.get_splits()

sigma_v = np.sqrt(0.001)
exponential = Exponential(
    mu_states=[0, 0.006, 22, 0, 0],
    var_states=[0.00001**2, 0.0025**2, 0.5**2, 0, 0],
)
noise = WhiteNoise(std_error=sigma_v)
localtrend = LocalTrend(
    mu_states=[-2, -0.005], var_states=[0.5**2, 0.005**2], std_error=0
)
locallevel = LocalLevel(mu_states=[-2], var_states=[0.5**2], std_error=0)
periodic = Periodic(mu_states=[0, 0], var_states=[1**2, 1**2], period=365.25)
var_W2bar_prior = 0.05
AR_process_error_var_prior = 0.05
ar = Autoregression(
    phi=0,
    mu_states=[0, 0, 0, AR_process_error_var_prior],
    var_states=[0, 0, 1e-6, var_W2bar_prior],
)
model = Model(exponential, ar, periodic, locallevel)
model.filter(data=train_data)
model.smoother(matrix_inversion_tol=1e-12)
mu_val_pred, std_val_pred, states = model.forecast(data=validation_data)

fig, ax = plot_states(
    data_processor=data_processor,
    states=states,
    states_to_plot=["scaled exp"],
    states_type="smooth",
)
plot_data(
    data_processor=data_processor,
    plot_train_data=True,
    plot_test_data=False,
    plot_validation_data=True,
    sub_plot=ax[0],
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_val_pred,
    std_validation_pred=std_val_pred,
    sub_plot=ax[0],
    color="k",
)


model1 = Model(localtrend, ar, periodic)
model1.filter(data=train_data)
model1.smoother(matrix_inversion_tol=1e-12)
mu_val_pred1, std_val_pred1, states1 = model1.forecast(data=validation_data)

fig, ax = plot_states(
    data_processor=data_processor,
    states=states1,
    states_to_plot=["level"],
    states_type="smooth",
)
plot_data(
    data_processor=data_processor,
    plot_train_data=True,
    plot_test_data=False,
    plot_validation_data=True,
    sub_plot=ax[0],
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_val_pred1,
    std_validation_pred=std_val_pred1,
    sub_plot=ax[0],
    color="k",
)
# Plot smooth

# fig, ax = plot_states(
#     data_processor=data_processor, states=model1.states, states_type="posterior"
# )
# fig, ax = plot_states(
#     data_processor=data_processor,
#     states=model1.states,
#     states_type="smooth",
#     # states_to_plot=(
#     #     "latent level",
#     #     "latent trend",
#     #     "exp scale factor",
#     #     "exp",
#     #     "scaled exp",
#     #     "level",
#     #     "trend",
#     # ),
# )


model2 = Model(exponential, localtrend, ar, periodic)
model2.filter(data=train_data)
model2.smoother(matrix_inversion_tol=1e-12)
mu_val_pred2, std_val_pred2, states2 = model2.forecast(data=validation_data)

fig, ax = plot_states(
    data_processor=data_processor,
    states=states2,
    states_to_plot=["scaled exp"],
    states_type="smooth",
)
plot_data(
    data_processor=data_processor,
    plot_train_data=True,
    plot_test_data=False,
    plot_validation_data=True,
    sub_plot=ax[0],
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_val_pred2,
    std_validation_pred=std_val_pred2,
    sub_plot=ax[0],
    color="k",
)
# Plot smooth

fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type="posterior",
    # states_to_plot=(
    #     "latent level",
    #     "latent trend",
    #     "exp scale factor",
    #     "exp",
    # "W2bar"
    #     "scaled exp",
    #     "level",
    #     "trend",
    # ),
)
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

plt.show()

# # Plot Prior
# fig, ax = plot_states(
#     data_processor=data_processor,
#     states=model.states,
#     states_type="prior",
# )

# # Plot posterior
# fig, ax = plot_states(
#     data_processor=data_processor, states=model.states, states_type="posterior"
# )
# # Plot smooth
# fig, ax = plot_states(
#     data_processor=data_processor,
#     states=model.states,
#     states_type="smooth",
#     # states_to_plot=(
#     #     "latent level",
#     #     "latent trend",
#     #     "exp scale factor",
#     #     "exp",
#     #     "scaled exp",
#     #     "level",
#     #     "trend",
#     # ),
# )

# plt.figure(figsize=(10, 8))
# plt.plot(
#     df.index,
#     model.states.get_mean("scaled exp", "smooth")
#     + model.states.get_mean("level", "smooth")
#     + model.states.get_mean("periodic 1", "smooth"),
#     color="purple",
# )
# scaled_exp_index = model.get_states_index("scaled exp")
# level_index = model.get_states_index("level")
# periodic_index = model.get_states_index("periodic 1")
# cov_scaled_exp_level = []
# cov_scaled_exp_periodic = []
# cov_level_periodic = []
# for i in range(len(model.states.get_mean("level", "smooth"))):
#     cov_scaled_exp_level.append(
#         model.states.var_smooth[i][scaled_exp_index, level_index]
#     )
#     cov_scaled_exp_periodic.append(
#         model.states.var_smooth[i][scaled_exp_index, periodic_index]
#     )
#     cov_level_periodic.append(model.states.var_smooth[i][level_index, periodic_index])
# cov_scaled_exp_level = np.array(cov_scaled_exp_level)
# cov_scaled_exp_periodic = np.array(cov_scaled_exp_periodic)
# cov_level_periodic = np.array(cov_level_periodic)

# plt.fill_between(
#     data_processor.get_time("all"),
#     model.states.get_mean("scaled exp", "smooth")
#     + model.states.get_mean("level", "smooth")
#     + model.states.get_mean("periodic 1", "smooth")
#     + np.sqrt(
#         model.states.get_std("scaled exp", "smooth") ** 2
#         + model.states.get_std("level", "smooth") ** 2
#         + model.states.get_std("periodic 1", "smooth") ** 2
#         + 2 * (cov_scaled_exp_level + cov_scaled_exp_periodic + cov_level_periodic)
#     ),
#     model.states.get_mean("scaled exp", "smooth")
#     + model.states.get_mean("level", "smooth")
#     + model.states.get_mean("periodic 1", "smooth")
#     - np.sqrt(
#         model.states.get_std("scaled exp", "smooth") ** 2
#         + model.states.get_std("level", "smooth") ** 2
#         + 2 * (cov_scaled_exp_level + cov_scaled_exp_periodic + cov_level_periodic)
#     ),
#     color="purple",
#     alpha=0.3,
# )

# plt.scatter(df.index, data_processor.get_data("all"), color="red", s=1)

# plt.show()
