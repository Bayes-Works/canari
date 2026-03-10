import copy
from scipy.optimize import minimize
import scipy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from canari import (
    DataProcess,
    Model,
    plot_states,
    plot_data,
    plot_prediction,
)
from canari.component import (
    Exponential,
    WhiteNoise,
    LocalTrend,
    Periodic,
    LocalLevel,
    Autoregression,
    LocalAcceleration,
)
from matplotlib.lines import Line2D
from prophet import Prophet
df_raw = pd.read_csv(
    "/Users/michelwu/Desktop/Exp DAT/reel_data/LTU014PIAEVA920.DAT",
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
mask = ~np.isnan(df)
df = df[mask].resample("W").mean()


date_1 = "2010-07-04"
date_2 = "2016-10-27"
mask1 = df.index < date_1
df_part1 = df[mask1]
print(df_part1.iloc[350:])
mask2 = (df.index >= date_1) & (df.index < date_2)
df_part2 = df[mask2]
# mask_data = df.index < date_2
# df_part = df[mask_data]

first_year = df_part1.index.min().year
last_year = df_part1.index.max().year
years = [str(year) for year in range(first_year, last_year - 4)]
validation_start_str = df_part1.loc[str(last_year - 1)].index[0].strftime("%Y-%m-%d")
last_date_str=df_part1.index[-1].strftime("%Y-%m-%d")
# test_start_str = df.loc[str(last_year - 1)].index[0].strftime("%Y-%m-%d")

data_processor1 = DataProcess(
    data=df_part1,
    train_start=df_part1.loc[years[0]].index[0].strftime("%Y-%m-%d"),
    validation_start=validation_start_str,
    validation_end=last_date_str,
    test_start=last_date_str,
    output_col=[0],
    standardization=False,
)


train_data, validation_data, test_data, all_data = data_processor1.get_splits()
df_train=pd.DataFrame(index=train_data["time"], data={'y':train_data["y"].flatten()})

mask2 = (df.index >= date_1) & (df.index < date_2)
df_part2 = df[mask2]
data_processor2 = DataProcess(
    data=df_part2,
    train_split=0.9,
    validation_split=0.1,
    output_col=[0],
    standardization=False,
)
train_data2, validation_data2, _, all_data2 = data_processor2.get_splits()
index_debut_validation = len(train_data2["y"])
date_debut_val = df_part2.index[index_debut_validation]
date_fin_val = df_part2.index[-1]

train_data, validation_data, test_data, all_data = data_processor1.get_splits()
df_train2=pd.DataFrame(index=train_data2["time"], data={'y':train_data2["y"].flatten()})

expo=Exponential()
local_level=LocalLevel()
ar=Autoregression(phi=0,mu_states=[0,0,0,0])
wn=WhiteNoise()
periodic=Periodic(period=52)
model=Model(expo,ar,periodic,local_level)
model.auto_initialize_comp(data_training=df_train2,ratio_training=0.6)
model.filter(data=train_data2)
mu_val, std_val, states = model.forecast(data=validation_data2)


# scaled_exp_index = model.get_states_index("scaled exp")
# periodic_index = model.get_states_index("periodic 1")
# autoregression_index = model.get_states_index("autoregression")
# level_index = model.get_states_index("level")
# cov_scaled_exp_auto = []
# cov_scaled_exp_periodic = []
# cov_periodic_auto = []
# cov_periodic_level = []
# cov_scaled_exp_level = []
# cov_level_auto = []
# for i in range(len(model.states.get_mean("scaled exp", states_type="smooth"))):
#     cov_scaled_exp_periodic.append(
#         model.states.var_posterior[i][scaled_exp_index, periodic_index]
#     )
#     cov_scaled_exp_auto.append(
#         model.states.var_posterior[i][
#             scaled_exp_index, model.get_states_index("autoregression")
#         ]
#     )
#     cov_periodic_auto.append(
#         model.states.var_posterior[i][periodic_index, autoregression_index]
#     )
#     cov_periodic_level.append(model.states.var_posterior[i][periodic_index, level_index])
#     cov_scaled_exp_level.append(
#         model.states.var_posterior[i][scaled_exp_index, level_index]
#     )
#     cov_level_auto.append(model.states.var_posterior[i][level_index, autoregression_index])

# cov_scaled_exp_auto = np.array(cov_scaled_exp_auto)
# cov_scaled_exp_periodic = np.array(cov_scaled_exp_periodic)
# cov_periodic_auto = np.array(cov_periodic_auto)
# cov_scaled_exp_level = np.array(cov_scaled_exp_level)
# cov_level_auto = np.array(cov_level_auto)
# cov_periodic_level = np.array(cov_periodic_level)

# y_mean_expo_sinus_level2 = (
#     model.states.get_mean("scaled exp", states_type="posterior")
#     + model.states.get_mean("periodic 1", states_type="posterior")
#     # + model.states.get_mean("autoregression", states_type="posterior")
#     + model.states.get_mean("level", states_type="posterior")
# )
# y_std_expo_sinus_level2 = np.sqrt(
#     model.states.get_std("scaled exp", states_type="posterior") ** 2
#     + model.states.get_std("periodic 1", states_type="posterior") ** 2
#     # + model.states.get_std("autoregression", states_type="posterior") ** 2
#     + model.states.get_std("level", states_type="posterior") ** 2
#     + 2
#     * (
#         cov_scaled_exp_periodic
#         # + cov_scaled_exp_auto
#         # + cov_periodic_auto
#         + cov_scaled_exp_level
#         # + cov_level_auto
#         + cov_periodic_level
#     )
# )
# plt.figure(figsize=(3.75, 1.3))
# plt.plot(
#     df_part2.index,
#     y_mean_expo_sinus_level2,
#     color="purple",
#     linewidth=1,
# )
# plt.fill_between(
#     df_part2.index,
#     y_mean_expo_sinus_level2 + y_std_expo_sinus_level2,
#     y_mean_expo_sinus_level2 - y_std_expo_sinus_level2,
#     color="purple",
#     alpha=0.3,
# )
# plt.scatter(
#     df_part2.index,
#     data_processor2.get_data("all"),
#     s=0.2,
#     color="red",
# )

# plt.plot(
#     df_part2.index,
#     model.states.get_mean("scaled exp", states_type="posterior"),
#     color="orange",
#     linewidth=1,
#     alpha=0.5,
# )
# plt.fill_between(
#     df_part2.index,
#     model.states.get_mean("scaled exp", states_type="posterior")
#     + model.states.get_std("scaled exp", states_type="posterior"),
#     model.states.get_mean("scaled exp", states_type="posterior")
#     - model.states.get_std("scaled exp", states_type="posterior"),
#     color="orange",
#     alpha=0.3,
# )


# ax = plt.gca()  # Récupérer l'axe courant
# ax.xaxis.set_major_locator(mdates.YearLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# fig, axes = plot_states(
#     data_processor=data_processor2,
#     states=model.states,
#     states_type="posterior",
#     # states_to_plot=(
#     #     # "latent level",
#     #     "latent trend",
#     #     "exp scale factor",
#     #     #     "exp",
#     #     "scaled exp",
#     #     "level",
#     #     #     "trend",
#     #     "periodic 1",
#     #     # "autoregression",
#     # ),
# )
# if not isinstance(axes, (list, np.ndarray)):
#     axes = [axes]

#     # Réduit les ticks sur chaque subplot
# for ax in axes:
#     for k, label in enumerate(ax.get_xticklabels()):
#         label.set_visible(k % 2 == 0)
# axes[0].tick_params(labelbottom=False)

# fig.set_size_inches(5, 2.5)

# plt.figure(figsize=(5, 2.5))

# plt.plot(
#     df_part2.index,
#     y_mean_expo_sinus_level2,
#     color="purple",
#     linewidth=1,
# )
# plt.fill_between(
#     df_part2.index,
#     y_mean_expo_sinus_level2 + y_std_expo_sinus_level2,
#     y_mean_expo_sinus_level2 - y_std_expo_sinus_level2,
#     color="purple",
#     alpha=0.3,
# )
# plt.scatter(
#     df_part2.index,
#     data_processor2.get_data("all"),
#     s=0.2,
#     color="red",
# )
# plt.plot(
#     df_part2.index,
#     y_mean_expo_sinus_level2,
#     color="purple",
#     linewidth=1,
#     label="Modèle global",  # Ajout d'un label pour la légende
# )

# # --- AJOUT DE LA ZONE DE VALIDATION ---
# plt.axvspan(
#     date_debut_val,
#     date_fin_val,
#     color="green",
#     alpha=0.1,
#     label="Zone Validation (Part 2)",
# )
# plt.show()

# def exponential_func(params,t):
#     a,b,c,d,e=params
#     return a*(np.exp(-b*t)-1)+c*np.sin(2*np.pi*(t+d)/52)+e

def exponential_func(t, a, b, c, d, e):
    return a * (np.exp(-b * t) - 1) + c * np.sin(2 * np.pi * (t + d) / 52) + e

t=np.arange(len(train_data2["y"]))
t_tot=np.arange(len(all_data2["y"]))

# model_main=exponential_func((34,0.001,1.5,33,-12.2),t)
# plt.plot(t,model_main,color="blue")
# plt.scatter(t,all_data2["y"],s=0.2,color="red")
from scipy.optimize import curve_fit
# plt.show()t = np.arange(len(all_data2["y"]))
# Extract the values
y_observed = train_data2["y"].flatten()

# Create a mask to find valid indices
mask_valid = ~np.isnan(y_observed) & ~np.isinf(y_observed)

# Filter both t and y
t_clean = t[mask_valid]
y_clean = y_observed[mask_valid]

# 3. Utilisez curve_fit (p0 est votre estimation initiale pour aider l'algorithme)
initial_guess = [34, 0.001, 1.5, 33, -12.2]

# Now run the fit on the cleaned data
lower_bounds = [0, 0, -np.inf, 0, -np.inf]
upper_bounds = [np.inf, 1, np.inf, 52, np.inf]

popt, pcov = curve_fit(
    exponential_func, t_clean, y_clean, 
    p0=initial_guess, 
    bounds=(lower_bounds, upper_bounds),
    maxfev=10000
)

# popt contient maintenant vos paramètres optimaux : [a, b, c, d, e]
print("Paramètres optimisés :", popt)

# 4. Affichage
plt.figure(figsize=(10, 6))
plt.scatter(t_tot, all_data2["y"].flatten(), s=0.2, color="red", label="Données")
plt.plot(t_tot, exponential_func(t_tot, *popt), color="blue", label="Fit automatique")
plt.legend()
plt.show()