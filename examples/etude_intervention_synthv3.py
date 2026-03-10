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

os.makedirs("LTU_Etude_intervention_synthv3", exist_ok=True)

LINE_WIDTH = 1
data_file = "/Users/michelwu/Desktop/Exponential component/donnes_synth_intervention_completev3.csv"
df_raw = pd.read_csv(data_file, sep=";", parse_dates=["temps"], index_col="temps")
df = df_raw[["exponential"]]
X_latent_level = df_raw[["X_EL"]]
X_latent_trend = df_raw[["X_ET"]]
X_exp_scale_factor = df_raw[["X_A"]]
X_local_level = df_raw[["X_local_level"]]
scaled_exp_list = df_raw[["exp_pur"]]

intervention_date = "2030-12-08"

mask1 = df.index <= intervention_date
df_part1 = df[mask1]

mask2 = df.index > intervention_date
df_part2 = df[mask2]

####
data_processor1 = DataProcess(
    data=df_part1,
    train_split=0.7,
    validation_split=0.3,
    output_col=[0],
    standardization=False,
)

train_data, validation_data, _, all_data = data_processor1.get_splits()

start_date_offset = df_part1.index.min()
end_date_offset = start_date_offset + pd.DateOffset(months=12)
df_first_full_year = df_part1.loc[start_date_offset:end_date_offset]
off_set_mean_series = np.mean(df_first_full_year, axis=0)
off_set_mean = off_set_mean_series.iloc[0]
# off_set_mean = -4.9

# Fait la même chose pour std
off_set_std_series = np.std(df_first_full_year, axis=0)
off_set_std = off_set_std_series.iloc[0]
print(off_set_mean)
print(off_set_std)

p = 52
periode = 52


def exponential_periodique_locallevel_parametrique(a, b, c, d, e, t):
    return a * (np.exp(-b * t) - 1) + c * np.sin(((2 * np.pi) / 52) * (t + d)) + e


def mse(params, t, y_true):
    a, b, c, d, e = params
    y_pred = exponential_periodique_locallevel_parametrique(a, b, c, d, e, t)
    mask = ~np.isnan(y_true)
    return np.mean((y_true[mask] - y_pred[mask]) ** 2)


def circular_mean(angles_list, period):
    """
    Calcule la moyenne d'une liste d'angles (ex: 0-12) de manière robuste.
    """
    if not angles_list:
        return np.nan

    # Convertit les angles (ex: 0-12) en radians (0-1pi)
    angles_rad = np.array(angles_list) * (2 * np.pi / period)

    # Calcule les composantes x et y moyennes
    x_mean = np.nanmean(np.cos(angles_rad))
    y_mean = np.nanmean(np.sin(angles_rad))

    # Reconvertit le vecteur moyen (x_mean, y_mean) en un angle (radians)
    mean_rad = np.arctan2(y_mean, x_mean)

    # Reconvertit l'angle de radians en "période" (ex: mois)
    mean_period_angle = mean_rad * (period / (2 * np.pi))

    # Applique le modulo final
    return mean_period_angle % period


def log_likelihood_validation(y_true, mu_pred, std_pred):
    """Log-vraisemblance gaussienne."""
    y_true = y_true.flatten()
    mu_pred = mu_pred.flatten()
    std_pred = std_pred.flatten()

    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    mu_pred = mu_pred[mask]
    std_pred = std_pred[mask]

    ll = -0.5 * np.sum(
        np.log(2 * np.pi * std_pred**2) + ((y_true - mu_pred) ** 2) / (std_pred**2)
    )
    return ll


amplitude_found = []
phase_found = []
ω = 2 * np.pi / periode
sinus_detrend_liste = []

s = train_data["y"].flatten()
mask_s = ~np.isnan(s)
sinus_detrend = []
trend_estimate = np.zeros_like(s)
half_left = (periode - 1) // 2
half_right = periode // 2
for j in range(0, len(s)):
    start = max(0, j - half_left)
    end = min(len(s), j + half_right + 1)
    segment = s[start:end]
    mask_segment = ~np.isnan(segment)
    trend_estimate[j] = np.mean(segment[mask_segment])
sinus_detrend_estimate = s - trend_estimate
min_cycle = []
max_cycle = []
phase_cycle = []
phase_segment = []
for i in range(0, len(s) - periode + 1):
    segment = s[i : i + periode]
    mask_segment_local = ~np.isnan(segment)
    if len(segment) == periode:
        min_cycle.append(np.min(segment[mask_segment_local]))
        max_cycle.append(np.max(segment[mask_segment_local]))
        phase_u = []
        for k in range(len(segment) - 1):
            val1 = segment[k]
            val2 = segment[k + 1]

            # VÉRIFICATION : Ignorer cette paire si l'une des valeurs est NaN
            if np.isnan(val1) or np.isnan(val2):
                continue  # Passe à l'itération k suivante

            num = val1 * np.sin(ω)
            den = val2 - val1 * np.cos(ω)
            θ_k = np.arctan2(num, den)
            φ_k = (θ_k / ω - (k + i)) % periode
            phase_u.append(φ_k)
        phase_segment.append(circular_mean(phase_u, periode))
max_moy = np.mean(max_cycle)
min_moy = np.mean(min_cycle)
amplitude_found = (max_moy - min_moy) / 2
phase_found = circular_mean(phase_segment, periode)
# phase_found = 3
s = train_data["y"].flatten()

t = np.arange(len(train_data["y"]))
scale_list = np.linspace(2, 40, 7)
trend_list = np.logspace(-8, -1, 6)
point_depart = [
    [s, t, amplitude_found, phase_found, off_set_mean]
    for s in scale_list
    for t in trend_list
]
bounds = [
    (2, 50),  # a
    (1e-8, 0.1),  # b
    (
        amplitude_found - 0.5,
        amplitude_found + 0.5,
    ),  # c
    (-periode // 2, periode + periode // 2),  # d
    (
        off_set_mean - off_set_std,
        off_set_mean + off_set_std,
    ),  # e
]
validation_LL = []
for idx, point in enumerate(point_depart):
    result = minimize(
        mse,
        point,
        args=(t, s),
        method="L-BFGS-B",
        bounds=bounds,
    )
    params_opt = result.x
    scale, latent_trend, periodic_scale, periodic_phase, offset = params_opt
    mse_boucle = mse(params_opt, t, s)
    var_diag = np.array(
        [
            (abs(scale) * 0.4) ** 2,  # var pour a (scale)
            (abs(latent_trend) * 0.45) ** 2,  # var pour b (latent_trend)
            (abs(periodic_scale) * 0.075) ** 2,  # var pour c (periodic_scale)
            (abs(periodic_phase) * 0.075) ** 2,  # var pour d (phase)
            (abs(offset) * 0.02) ** 2,  # var pour e (offset)
        ]
    )
    exponential = Exponential(
        mu_states=[
            0,
            latent_trend,
            scale,
            0,
            0,
        ],
        var_states=[
            0.00001**2,
            var_diag[1],
            var_diag[0],
            0,
            0,
        ],
    )
    locallevel = LocalLevel(
        mu_states=[offset],
        var_states=[var_diag[4]],
        std_error=0,
    )
    C = periodic_scale
    phi = periodic_phase
    p = 52
    a0 = C * np.sin((2 * np.pi / p) * phi)
    b0 = C * np.cos((2 * np.pi / p) * phi)
    w = 2 * np.pi / p
    A_periodic_inv = np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
    signal = A_periodic_inv @ np.array([[a0], [b0]])
    variance_signal = (
        A_periodic_inv
        @ np.array([[var_diag[2], 0], [0, var_diag[2]]])
        @ A_periodic_inv.T
    )
    periodic = Periodic(
        mu_states=[signal[0].item(), signal[1].item()],
        var_states=[variance_signal[0, 0].item(), variance_signal[1, 1].item()],
        period=p,
    )
    var_W2bar_prior = 0.05 * mse_boucle
    AR_process_error_var_prior = mse_boucle
    ar = Autoregression(
        phi=0,
        mu_states=[0, 0, 0, AR_process_error_var_prior],
        var_states=[0, 0, 1e-6, var_W2bar_prior],
    )
    model = Model(exponential, ar, periodic, locallevel)
    model.filter(data=train_data)
    model.smoother(matrix_inversion_tol=1e-12)
    mu_val_pred, std_val_pred, states = model.forecast(data=validation_data)
    validation_LL.append(
        log_likelihood_validation(validation_data["y"], mu_val_pred, std_val_pred)
    )

indice = np.argmax(validation_LL)
point_choisi = point_depart[indice]
print(point_choisi)
y_train_vali = np.concatenate((train_data["y"], validation_data["y"])).flatten()
t_train_vali = np.arange(len(y_train_vali))
result = minimize(
    mse,
    point_choisi,
    args=(t_train_vali, y_train_vali),
    method="L-BFGS-B",
    bounds=bounds,
)
params_opt = result.x
mse_train = mse(params_opt, t_train_vali, y_train_vali)
print(mse_train)
plt.plot(t_train_vali, y_train_vali, color="red")
plt.scatter(t_train_vali, y_train_vali, color="red")

scale, latent_trend, periodic_scale, periodic_phase, offset = params_opt
var_diag = np.array(
    [
        (abs(scale) * 0.4) ** 2,  # var pour a (scale)
        (abs(latent_trend) * 0.45) ** 2,  # var pour b (latent_trend)
        (abs(periodic_scale) * 0.075) ** 2,  # var pour c (periodic_scale)
        (abs(periodic_phase) * 0.075) ** 2,  # var pour d (phase)
        (abs(offset) * 0.02) ** 2,  # var pour e (offset)
    ]
)
exponential = Exponential(
    mu_states=[
        0,
        latent_trend,
        scale,
        0,
        0,
    ],
    var_states=[
        0.00001**2,
        var_diag[1],
        var_diag[0],
        0,
        0,
    ],
)
locallevel = LocalLevel(
    mu_states=[offset],
    var_states=[var_diag[4]],
    std_error=0,
)
C = periodic_scale
phi = periodic_phase
p = 52
a0 = C * np.sin((2 * np.pi / p) * phi)
b0 = C * np.cos((2 * np.pi / p) * phi)
w = 2 * np.pi / p
A_periodic_inv = np.array([[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]])
signal = A_periodic_inv @ np.array([[a0], [b0]])
variance_signal = (
    A_periodic_inv @ np.array([[var_diag[2], 0], [0, var_diag[2]]]) @ A_periodic_inv.T
)

periodic = Periodic(
    mu_states=[signal[0].item(), signal[1].item()],
    var_states=[variance_signal[0, 0].item(), variance_signal[1, 1].item()],
    period=p,
)
var_W2bar_prior = mse_train * 0.05
AR_process_error_var_prior = mse_train
ar = Autoregression(
    phi=0,
    mu_states=[0, 0, 0, AR_process_error_var_prior],
    var_states=[0, 0, 1e-6, var_W2bar_prior],
)
data_train_vali = {}
model = Model(exponential, ar, periodic, locallevel)
model.filter(data=all_data)


states_data1 = model.states
mu_data1 = model.states.mu_prior
var_data1 = model.states.var_prior
std_data1 = np.sqrt(np.diagonal(var_data1, axis1=1, axis2=2))
std_vec1 = std_data1[:, :, None]

print(model.states.mu_posterior[-1])
print(model.mu_obs_predict)
scaled_exp_index = model.get_states_index("scaled exp")
periodic_index = model.get_states_index("periodic 1")
autoregression_index = model.get_states_index("autoregression")
level_index = model.get_states_index("level") 
cov_scaled_exp_auto = []
cov_scaled_exp_periodic = []
cov_periodic_auto = []
cov_periodic_level = []
cov_scaled_exp_level = []
cov_level_auto = []
for i in range(len(model.states.get_mean("scaled exp", states_type="smooth"))):
    cov_scaled_exp_periodic.append(
        model.states.var_prior[i][scaled_exp_index, periodic_index]
    )
    cov_scaled_exp_auto.append(
        model.states.var_prior[i][
            scaled_exp_index, model.get_states_index("autoregression")
        ]
    )
    cov_periodic_auto.append(
        model.states.var_prior[i][periodic_index, autoregression_index]
    )
    cov_periodic_level.append(model.states.var_prior[i][periodic_index, level_index])
    cov_scaled_exp_level.append(
        model.states.var_prior[i][scaled_exp_index, level_index]
    )
    cov_level_auto.append(model.states.var_prior[i][level_index, autoregression_index])

cov_scaled_exp_auto = np.array(cov_scaled_exp_auto)
cov_scaled_exp_periodic = np.array(cov_scaled_exp_periodic)
cov_periodic_auto = np.array(cov_periodic_auto)
cov_scaled_exp_level = np.array(cov_scaled_exp_level)
cov_level_auto = np.array(cov_level_auto)
cov_periodic_level = np.array(cov_periodic_level)

y_mean_expo_sinus_level = (
    model.states.get_mean("scaled exp", states_type="prior")
    + model.states.get_mean("periodic 1", states_type="prior")
    # + model.states.get_mean("autoregression", states_type="prior")
    + model.states.get_mean("level", states_type="prior")
)
y_std_expo_sinus_level = np.sqrt(
    model.states.get_std("scaled exp", states_type="prior") ** 2
    + model.states.get_std("periodic 1", states_type="prior") ** 2
    # + model.states.get_std("autoregression", states_type="prior") ** 2
    + model.states.get_std("level", states_type="prior") ** 2
    + 2
    * (
        cov_scaled_exp_periodic
        # + cov_scaled_exp_auto
        # + cov_periodic_auto
        + cov_scaled_exp_level
        # + cov_level_auto
        + cov_periodic_level
    )
)

data_processor2 = DataProcess(
    data=df_part2,
    train_split=0.8,
    validation_split=0.2,
    output_col=[0],
    standardization=False,
)

train_data2, validation_data2, _, all_data2 = data_processor2.get_splits()
index_debut_validation = len(train_data2["y"])
date_debut_val = df_part2.index[index_debut_validation]
date_fin_val = df_part2.index[-1]

# ## Cas déterministe :
# model.mu_states[0] = 0.320255
# model.mu_states[1] = 0.0011
# model.mu_states[2] = 30
# model.mu_states[3] = np.exp(-model.mu_states[0] + 0.5 * model.var_states[0, 0]) - 1
# model.mu_states[4] = model.mu_states[2] * model.mu_states[3] + model.var_states[2, 3]
# model.mu_states[-1] = -7
# # mask_diag = np.eye(model.var_states.shape[0], dtype=bool)
# # model.var_states[~mask_diag] = 0.0

# idx_reset = [0, 1, 2, 3, 4, model.mu_states.shape[0] - 1]

# # 2. Créer un masque de "ce qu'il faut garder"
# # Par défaut, on veut tout garder...
# mask_keep = np.ones(model.var_states.shape, dtype=bool)

# # ... SAUF les liens qui impliquent nos indices touchés
# # On dit : si une ligne OU une colonne fait partie des indices reset, on ne garde pas (False)
# mask_keep[idx_reset, :] = False
# mask_keep[:, idx_reset] = False

# # 3. Exception : On veut quand même garder la diagonale (la variance propre) des éléments reset
# # (Car vous allez probablement la booster juste après, ou vous voulez garder l'incertitude actuelle)
# np.fill_diagonal(mask_keep, True)
# model.var_states[~mask_keep] = 0.0

# # Cas déterministe avec variance :
# model.mu_states[0] = model.mu_states[0] * 0.6
# model.mu_states[1] = 0.00119
# model.mu_states[2] = 27
# model.mu_states[3] = np.exp(-model.mu_states[0] + 0.5 * model.var_states[0, 0]) - 1
# model.mu_states[4] = model.mu_states[2] * model.mu_states[3] + model.var_states[2, 3]
# model.mu_states[-1] = -6.9
# model.var_states[0, 0] += (model.mu_states[0].item() * 0.05) ** 2
# model.var_states[1, 1] += (model.mu_states[1].item() * 0.1) ** 2
# model.var_states[2, 2] += (model.mu_states[2].item() * 0.1) ** 2
# model.var_states[-1, -1] += (model.mu_states[-1].item() * 0.015) ** 2
# idx_reset = [0, 1, 2, 3, 4, model.mu_states.shape[0] - 1]

# # 2. Créer un masque de "ce qu'il faut garder"
# # Par défaut, on veut tout garder...
# mask_keep = np.ones(model.var_states.shape, dtype=bool)

# # ... SAUF les liens qui impliquent nos indices touchés
# # On dit : si une ligne OU une colonne fait partie des indices reset, on ne garde pas (False)
# mask_keep[idx_reset, :] = False
# mask_keep[:, idx_reset] = False

# # 3. Exception : On veut quand même garder la diagonale (la variance propre) des éléments reset
# # (Car vous allez probablement la booster juste après, ou vous voulez garder l'incertitude actuelle)
# np.fill_diagonal(mask_keep, True)
# model.var_states[~mask_keep] = 0.0

# Cas off avec plus de variance
model.var_states[0, 0] += (model.mu_states[0].item() * 0.25) ** 2
model.var_states[1, 1] += (model.mu_states[1].item() * 0.05) ** 2
model.var_states[2, 2] += (model.mu_states[2].item() * 0.35) ** 2
model.var_states[-1, -1] += (model.mu_states[-1].item() * 0.05) ** 2
model.mu_states[0] = model.mu_states[0] * 0.2
model.mu_states[1] += -model.mu_states[1]*0.5
model.mu_states[2] += model.mu_states[2]*0.75
# model.mu_states[3] = np.exp(-model.mu_states[0] + 0.5 * model.var_states[0, 0]) - 1
# model.mu_states[4] = model.mu_states[2] * model.mu_states[3] + model.var_states[2, 3]
# print(model.mu_states[2] * model.mu_states[3] + model.var_states[2, 3])
# model.mu_states[-1] += 4
model.mu_states[-1] += model.states.mu_posterior[-1][4] - ((np.exp(-model.mu_states[0] + 0.5 * model.var_states[0, 0]) - 1)* model.mu_states[2]) - 1.5
# model.var_states[0, 0] += (model.mu_states[0].item() * 0.20) ** 2
# model.var_states[1, 1] += (model.mu_states[1].item() * 0.6) ** 2
# model.var_states[2, 2] += (model.mu_states[2].item() * 0.8) ** 2
# model.var_states[-1, -1] += (model.mu_states[-1].item() * 0.015) ** 2
idx_reset = [0, 1, 2, 3, 4,]

# 2. Créer un masque de "ce qu'il faut garder"
# Par défaut, on veut tout garder...
mask_keep = np.ones(model.var_states.shape, dtype=bool)

# ... SAUF les liens qui impliquent nos indices touchés
# On dit : si une ligne OU une colonne fait partie des indices reset, on ne garde pas (False)
mask_keep[idx_reset, :] = False
mask_keep[:, idx_reset] = False

# 3. Exception : On veut quand même garder la diagonale (la variance propre) des éléments reset
# (Car vous allez probablement la booster juste après, ou vous voulez garder l'incertitude actuelle)
np.fill_diagonal(mask_keep, True)
model.var_states[~mask_keep] = 0.0

# model.mu_states[2] += 8
# model.mu_states[1] += -0.00015
# model.mu_states[-1] += (
#     -0.5
#     + model.mu_states[4]
#     - ((model.mu_states[2]) * model.mu_states[3] + model.var_states[2, 3])
# )
# model.mu_states[2] = np.exp(model.mu_states[0] + 0.5 * model.var_states[0, 0]) - 1
# model.mu_states[-1] += (
#     -2
# + model.mu_states[4]
# - ((model.mu_states[2]) * model.mu_states[3] + model.var_states[2, 3])
# )
# model.mu_states[-1] += -2
# model.var_states[1, 1] += (model.mu_states[1] * 0.005) ** 2
# model.var_states[2, 2] += (model.mu_states[2] * 0.001) ** 2
# model.var_states[-1, -1] += (model.mu_states[1] * 0.05) ** 2


model.filter(data=train_data2)
mu_val, std_val, states = model.forecast(data=validation_data2)
states_data2 = model.states
print(model.states.mu_prior[0])
mu_data2 = model.states.mu_prior
var_data2 = model.states.var_prior
std_data2 = np.sqrt(np.diagonal(var_data2, axis1=1, axis2=2))
std_vec2 = std_data2[:, :, None]

mu_data_all = mu_data1 + mu_data2
var_data_all = var_data1 + var_data2
std_vec_all = np.concatenate((std_vec1, std_vec2), axis=0)

mu_data_all = np.array(mu_data_all)
var_data_all = np.array(var_data_all)


# states_alldata = np.concatenate((states_data1, states_data2))
fig, ax = plot_states(
    data_processor=data_processor2,
    states=states,
    states_to_plot=["scaled exp"],
    states_type="prior",
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
    mean_validation_pred=mu_val,
    std_validation_pred=std_val,
    sub_plot=ax[0],
    color="k",
)
fig.set_size_inches(5, 2.5)

plt.scatter(df_part2.index, data_processor2.get_data("all"), color="red", s=2.5)

scaled_exp_index = model.get_states_index("scaled exp")
periodic_index = model.get_states_index("periodic 1")
autoregression_index = model.get_states_index("autoregression")
level_index = model.get_states_index("level")
cov_scaled_exp_auto = []
cov_scaled_exp_periodic = []
cov_periodic_auto = []
cov_periodic_level = []
cov_scaled_exp_level = []
cov_level_auto = []
for i in range(len(model.states.get_mean("scaled exp", states_type="smooth"))):
    cov_scaled_exp_periodic.append(
        model.states.var_prior[i][scaled_exp_index, periodic_index]
    )
    cov_scaled_exp_auto.append(
        model.states.var_prior[i][
            scaled_exp_index, model.get_states_index("autoregression")
        ]
    )
    cov_periodic_auto.append(
        model.states.var_prior[i][periodic_index, autoregression_index]
    )
    cov_periodic_level.append(model.states.var_prior[i][periodic_index, level_index])
    cov_scaled_exp_level.append(
        model.states.var_prior[i][scaled_exp_index, level_index]
    )
    cov_level_auto.append(model.states.var_prior[i][level_index, autoregression_index])

cov_scaled_exp_auto = np.array(cov_scaled_exp_auto)
cov_scaled_exp_periodic = np.array(cov_scaled_exp_periodic)
cov_periodic_auto = np.array(cov_periodic_auto)
cov_scaled_exp_level = np.array(cov_scaled_exp_level)
cov_level_auto = np.array(cov_level_auto)
cov_periodic_level = np.array(cov_periodic_level)

y_mean_expo_sinus_level2 = (
    model.states.get_mean("scaled exp", states_type="prior")
    + model.states.get_mean("periodic 1", states_type="prior")
    # + model.states.get_mean("autoregression", states_type="prior")
    + model.states.get_mean("level", states_type="prior")
)
y_std_expo_sinus_level2 = np.sqrt(
    model.states.get_std("scaled exp", states_type="prior") ** 2
    + model.states.get_std("periodic 1", states_type="prior") ** 2
    # + model.states.get_std("autoregression", states_type="prior") ** 2
    + model.states.get_std("level", states_type="prior") ** 2
    + 2
    * (
        cov_scaled_exp_periodic
        # + cov_scaled_exp_auto
        # + cov_periodic_auto
        + cov_scaled_exp_level
        # + cov_level_auto
        + cov_periodic_level
    )
)
plt.figure(figsize=(3.75, 1.3))
plt.plot(
    df_part2.index,
    y_mean_expo_sinus_level2,
    color="purple",
    linewidth=LINE_WIDTH,
)
plt.fill_between(
    df_part2.index,
    y_mean_expo_sinus_level2 + y_std_expo_sinus_level2,
    y_mean_expo_sinus_level2 - y_std_expo_sinus_level2,
    color="purple",
    alpha=0.3,
)
plt.scatter(
    df_part2.index,
    data_processor2.get_data("all"),
    s=0.2,
    color="red",
)

plt.plot(
    df_part2.index,
    model.states.get_mean("scaled exp", states_type="prior"),
    color="orange",
    linewidth=LINE_WIDTH,
    alpha=0.5,
)
plt.fill_between(
    df_part2.index,
    model.states.get_mean("scaled exp", states_type="prior")
    + model.states.get_std("scaled exp", states_type="prior"),
    model.states.get_mean("scaled exp", states_type="prior")
    - model.states.get_std("scaled exp", states_type="prior"),
    color="orange",
    alpha=0.3,
)


ax = plt.gca()  # Récupérer l'axe courant
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig, axes = plot_states(
    data_processor=data_processor2,
    states=model.states,
    states_type="prior",
    states_to_plot=(
        # "latent level",
        "latent trend",
        "exp scale factor",
        #     "exp",
        "scaled exp",
        "level",
        #     "trend",
        "periodic 1",
        # "autoregression",
    ),
)
if not isinstance(axes, (list, np.ndarray)):
    axes = [axes]

    # Réduit les ticks sur chaque subplot
for ax in axes:
    for k, label in enumerate(ax.get_xticklabels()):
        label.set_visible(k % 2 == 0)
axes[0].tick_params(labelbottom=False)

fig.set_size_inches(5, 2.5)

plt.figure(figsize=(5, 2.5))
plt.plot(
    df_part1.index,
    y_mean_expo_sinus_level,
    color="purple",
    linewidth=LINE_WIDTH,
)
plt.fill_between(
    df_part1.index,
    y_mean_expo_sinus_level + y_std_expo_sinus_level,
    y_mean_expo_sinus_level - y_std_expo_sinus_level,
    color="purple",
    alpha=0.3,
)
plt.scatter(
    df_part1.index,
    data_processor1.get_data("all"),
    s=0.2,
    color="red",
)


plt.plot(
    df_part2.index,
    y_mean_expo_sinus_level2,
    color="purple",
    linewidth=LINE_WIDTH,
)
plt.fill_between(
    df_part2.index,
    y_mean_expo_sinus_level2 + y_std_expo_sinus_level2,
    y_mean_expo_sinus_level2 - y_std_expo_sinus_level2,
    color="purple",
    alpha=0.3,
)
plt.scatter(
    df_part2.index,
    data_processor2.get_data("all"),
    s=0.2,
    color="red",
)
plt.plot(
    df_part2.index,
    y_mean_expo_sinus_level2,
    color="purple",
    linewidth=LINE_WIDTH,
    label="Modèle global",  # Ajout d'un label pour la légende
)

# --- AJOUT DE LA ZONE DE VALIDATION ---
plt.axvspan(
    date_debut_val,
    date_fin_val,
    color="green",
    alpha=0.1,
    label="Zone Validation (Part 2)",
)

ax = plt.gca()  # Récupérer l'axe courant
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# indices 11 : 0 XEL 1 XET 2 XES 3 exp 4 scaled exp 5 6 7 8 9 periodic 1 10 periodic 2 11 level


def plot_states(mu, std, states_indices=None, states_names=None):
    """
    mu, std : arrays de shape (T, N_states, 1)
    states_indices : liste des indices d'états à afficher (ex: [0, 2, 5]). Si None, affiche tout.
    states_names : liste des noms pour le titre (optionnel)
    """

    # Si aucun indice n'est spécifié, on prend tout
    if states_indices is None:
        states_indices = range(mu.shape[1])

    n_plots = len(states_indices)

    # Création d'une figure avec des sous-graphiques (1 colonne, n lignes)
    # figsize : largeur=12, hauteur = 3 * nombre de graphiques
    fig, axes = plt.subplots(
        nrows=n_plots, ncols=1, figsize=(12, 3 * n_plots), sharex=True
    )

    # Si on n'a qu'un seul état, axes n'est pas une liste, on le met dans une liste
    if n_plots == 1:
        axes = [axes]

    # Boucle sur les états choisis
    for i, ax in enumerate(axes):
        state_idx = states_indices[i]

        # 1. Extraction des données (et suppression de la dimension inutile '1')
        # On passe de (725, 1) à (725,)
        y = mu[:, state_idx, 0]
        err = std[:, state_idx, 0]
        x = df_part1.index.append(df_part2.index)  # ou votre vecteur temps réel

        # 2. Bornes sup et inf
        lower_bound = y - err
        upper_bound = y + err

        # 3. Plot de la moyenne (Ligne solide)
        ax.plot(x, y, label="Moyenne", color="blue", linewidth=1)
        if state_idx == model.get_states_index("latent level"):
            ax.plot(x, X_latent_level, color="black", linestyle="--")

        if state_idx == model.get_states_index("latent trend"):
            ax.plot(x, X_latent_trend, color="black", linestyle="--")
        if state_idx == model.get_states_index("exp scale factor"):
            ax.plot(x, X_exp_scale_factor, color="black", linestyle="--")
        if state_idx == model.get_states_index("level"):
            ax.plot(x, X_local_level, color="black", linestyle="--")
        if state_idx == model.get_states_index("scaled exp"):
            ax.plot(x, scaled_exp_list, color="black", linestyle="--")

        # 4. Plot de la zone d'incertitude (Zone ombrée)
        # alpha gère la transparence (0.0 transparent, 1.0 opaque)
        ax.fill_between(
            x,
            lower_bound,
            upper_bound,
            color="blue",
            alpha=0.3,
            label="std dev",
            edgecolor="none",
        )

        # Esthétique
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylabel(model.states.states_name[state_idx], fontsize=10)

    plt.xlabel("Temps (pas)")
    plt.tight_layout()  # Ajuste automatiquement les espacements pour que ce soit joli
    plt.show()


# --- UTILISATION ---

# Par exemple, on veut afficher seulement l'état 0, l'état 3 et l'état 11

indices_a_voir = [
    model.get_states_index("latent level"),
    model.get_states_index("latent trend"),
    model.get_states_index("exp scale factor"),
    model.get_states_index("exp"),
    model.get_states_index("scaled exp"),
    model.get_states_index("periodic 1"),
    model.get_states_index("autoregression"),
    model.get_states_index("level"),
]
noms_des_etats = model.states.states_name  # Votre liste complète
plot_states(
    mu_data_all, std_vec_all, states_indices=indices_a_voir, states_names=noms_des_etats
)
