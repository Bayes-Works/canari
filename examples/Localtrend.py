import numpy as np
import pandas as pd
import copy
from scipy import stats
from matplotlib import pyplot as plt, cm
import sys
import os
from pathlib import Path
project_root = Path.cwd().resolve().parents[1]
sys.path.append(str(project_root))
from canari import DataProcess, plot_data


# set plotting parameters
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": False,
    "pgf.rcfonts": False,
})

# set plotting style
# plt.style.use("seaborn-v0_8-colorblind")

import matplotlib as mpl
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}"
})

df_raw = pd.read_csv(
    '/Users/michelwu/Desktop/Exponential component/2650F162.CSV',    sep=";",  # Semicolon as delimiter
    quotechar='"',  # Double quotes as text qualifier
    engine="python",  # Python engine for complex cases
    na_values=[""],  # Treat empty strings as NaN
    skipinitialspace=True,  # Skip spaces after delimiter
    encoding="ISO-8859-1",
    parse_dates=["Date"],
    index_col="Date"
    )
df= df_raw["Deplacements cumulatif X (mm)"]
df = df+5
df=df.iloc[65:]
print(len(df))


# df = df_raw.iloc[1:,6]
# time=pd.to_datetime(df_raw.iloc[1:,3])
# df.index = time
# df.index.name = "time"
# df.columns = ["crack opening"]
# df.head()

df = df.resample("M").mean()
df.head()
print(len(df))

#Définition Matrice

A=np.array([[1,1],[0,1]])
C=np.array([1,0])

R=1**2

T=len(df)-1
tliste=np.linspace(0,T,T+1)

def one_step_predict_exp(A,M_Posterior,V_Posterior):
    M_Prior=A@M_Posterior
    V_Prior=A@V_Posterior@np.transpose(A)
    return M_Prior,V_Prior

def one_step_update_exp(E_Prior, V_Prior, y_obs, C, R):
    G = C @ V_Prior @ np.reshape(C, (-1, 1)) + R
    K = V_Prior @ np.reshape(C, (-1, 1)) * (1 / G)
    ychap = C @ E_Prior
    rt = y_obs - ychap
    V_Posterior = (np.eye(len(C)) - K @ np.reshape(C, (1, 2))) @ V_Prior
    V_Posterior = 0.5 * (V_Posterior + V_Posterior.T)

    E_Posterior = E_Prior + K * rt
    return E_Posterior, V_Posterior

LL = 0  # Initialisation du likelihood

M_X0 = np.array([[1.5], [-0.01]])
V_X0 = np.array([[(0.2)**2 , 0 ], [0 , (0.1)**2]])
M_Prior, V_Prior = one_step_predict_exp(A, M_X0, V_X0)

if not np.isnan(df.iloc[1]):
    LL += stats.norm.logpdf(df.iloc[1], loc=M_Prior[0], scale=np.sqrt(V_Prior[0, 0]))
    M_Posterior, V_Posterior = one_step_update_exp(M_Prior, V_Prior, df.iloc[1], C, R)
else:
    M_Posterior,V_Posterior=M_Prior,V_Prior

M_Xliste = [M_X0.copy()]
M_Xliste.append(M_Posterior.copy())
V_Xliste = [V_X0.copy()]
V_Xliste.append(V_Posterior.copy())
V_Xlistediag = [np.diag(V_X0)]
V_Xlistediag.append(np.diag(V_Posterior))

M_Xliste_prior = [M_X0.copy()]
M_Xliste_prior.append(M_Prior.copy())
V_Xliste_prior = [V_X0.copy()]
V_Xliste_prior.append(V_Prior.copy())
V_Xlistediag_prior = [np.diag(V_X0)]
V_Xlistediag_prior.append(np.diag(V_Prior))

for i in range(2, T + 1):
    M_Prior, V_Prior = one_step_predict_exp(A, M_Posterior, V_Posterior)
    M_Xliste_prior.append(M_Prior.copy())
    V_Xliste_prior.append(V_Prior.copy())
    V_Xlistediag_prior.append(np.diag(V_Prior))
    if not np.isnan(df.iloc[i]):
        LL += stats.norm.logpdf(df.iloc[i], loc=M_Prior[0], scale=np.sqrt(V_Prior[0, 0]))
        M_Posterior, V_Posterior = one_step_update_exp(M_Prior, V_Prior, df.iloc[i], C, R)
    else:
        M_Posterior,V_Posterior=M_Prior,V_Prior

    M_Xliste.append(M_Posterior.copy())
    V_Xliste.append(V_Posterior.copy())
    V_Xlistediag.append(np.diag(V_Posterior))

print(LL)
# Extraction des données
x_eLpredit = [arr[0].item() for arr in M_Xliste]
x_eTpredit = [arr[1].item() for arr in M_Xliste]

Vx_eLpredit = [arr[0] for arr in V_Xlistediag]
Vx_eTpredit = [arr[1] for arr in V_Xlistediag]

x_eLprior = [arr[0].item() for arr in M_Xliste_prior]
x_eTprior = [arr[1].item() for arr in M_Xliste_prior]

Vx_eLprior = [arr[0] for arr in V_Xlistediag_prior]
Vx_eTprior = [arr[1] for arr in V_Xlistediag_prior]

A_lin = A  # Sous-matrice pour le smoother (3x3)

M_smooth_base = [None] * len(M_Xliste)
V_smooth_base = [None] * len(V_Xliste)

# Initialisation avec le dernier posterior
M_smooth_base[-1] = M_Xliste[-1].copy()
V_smooth_base[-1] = V_Xliste[-1].copy()

for t in range(T - 1, -1, -1):
    # Prédictions et filtres
    M_filt = M_Xliste[t]
    V_filt = V_Xliste[t]
    M_pred = M_Xliste_prior[t + 1]
    V_pred = V_Xliste_prior[t + 1]

    # Gain RTS sur les 3 premières dimensions
    A_lin = A
    G = V_filt @ A_lin.T @ np.linalg.inv(V_pred)

    # Lissage sur les 3 premières dimensions uniquement
    M_smooth_base[t] = M_filt.copy()
    M_smooth_base[t] += G @ (M_smooth_base[t + 1][:3] - M_pred[:3])

    V_smooth_base[t] = V_filt.copy()
    V_smooth_base[t] += G @ (V_smooth_base[t + 1][:3, :3] - V_pred[:3, :3]) @ G.T

x_eLsmooth = [arr[0].item() for arr in M_smooth_base]
x_eTsmooth = [arr[1].item() for arr in M_smooth_base]

Vx_eLsmooth = [np.diag(v)[0] for v in V_smooth_base]
Vx_eTsmooth = [np.diag(v)[1] for v in V_smooth_base]

fig, ax = plt.subplots(4, 1, figsize=(5, 3))

# X_EL
# ax[0].plot(tliste, X_EL_True, color='black', linestyle='--', label='True')
ax[0].plot(tliste, df, color='red')
ax[0].scatter(tliste, df, color='red',s=2.5)
ax[0].plot(tliste, x_eLpredit, color='blue', label='Posterior')
ax[0].plot(tliste, x_eLsmooth, color='cyan', label='Smooth')
ax[0].fill_between(tliste, x_eLpredit - np.sqrt(Vx_eLpredit), x_eLpredit + np.sqrt(Vx_eLpredit),
                   color='blue', alpha=0.2)
ax[0].fill_between(tliste, x_eLsmooth - np.sqrt(Vx_eLsmooth), x_eLsmooth + np.sqrt(Vx_eLsmooth),
                   color='cyan', alpha=0.4)
ax[0].plot(tliste, x_eLprior, color='purple', linestyle=':', label='Prior')
ax[0].fill_between(tliste, x_eLprior - np.sqrt(Vx_eLprior), x_eLprior + np.sqrt(Vx_eLprior),
                   color='purple', alpha=0.15)
ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax[0].set_ylabel(r'$X_{EL}$')

# X_ET
# ax[1].plot(tliste, X_ET_True, color='black', linestyle='--')
ax[1].plot(tliste, x_eTpredit, color='blue')
ax[1].plot(tliste, x_eTsmooth, color='cyan', label='Smooth')
ax[1].fill_between(tliste, x_eTpredit - np.sqrt(Vx_eTpredit), x_eTpredit + np.sqrt(Vx_eTpredit),
                   color='blue', alpha=0.2)
ax[1].fill_between(tliste, x_eTsmooth - np.sqrt(Vx_eTsmooth), x_eTsmooth + np.sqrt(Vx_eTsmooth),
                   color='cyan', alpha=0.4)
ax[1].plot(tliste, x_eTprior, color='purple', linestyle=':')
ax[1].fill_between(tliste, x_eTprior - np.sqrt(Vx_eTprior), x_eTprior + np.sqrt(Vx_eTprior),
                   color='purple', alpha=0.15)
ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax[1].set_ylabel(r'$X_{ET}$')

plt.show()