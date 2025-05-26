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
df = df-1
# df=df.iloc[65:]
print(len(df))


# df = df_raw.iloc[1:,6]
# time=pd.to_datetime(df_raw.iloc[1:,3])
# df.index = time
# df.index.name = "time"
# df.columns = ["crack opening"]
# df.head()

fig = plt.subplots(figsize=(12, 3))
plt.scatter(df.index, df.values, color="r")
plt.title("Orginal data")

df = df.resample("M").mean()
df.head()
print(len(df))

fig = plt.subplots(figsize=(12, 3))
plt.scatter(df.index, df.values, color="r")
plt.title("Resample (4M) data")



M_exp= lambda M_XEL,V_XEL,M_XET,V_XET,Cov_XEL_XET : np.exp(-M_XEL-M_XET+0.5*(V_XEL+V_XET+2*Cov_XEL_XET))
V_exp= lambda M_XEL,V_XEL,M_XET,V_XET,Cov_XEL_XET : ((M_exp(M_XEL,V_XEL,M_XET,V_XET,Cov_XEL_XET))**2)*(np.exp(V_XEL+V_XET+2*Cov_XEL_XET)-1)
Cov_exp02= lambda M_XEL,V_XEL,M_XET,V_XET,Cov_XEL_XET : -(M_exp(M_XEL,V_XEL,M_XET,V_XET,Cov_XEL_XET))*(V_XEL+Cov_XEL_XET)
Cov_exp12= lambda M_XEL,V_XEL,M_XET,V_XET,Cov_XEL_XET : -(M_exp(M_XEL,V_XEL,M_XET,V_XET,Cov_XEL_XET))*(V_XET+Cov_XEL_XET)

#Fonction pour actualiser avec amplitude en plus

M_GMA= lambda M_X1,M_X2,CovX1X2 : M_X1*M_X2+CovX1X2
Cov_GMA= lambda M_X1,M_X2,CovX1X3,CovX2X3 :CovX1X3*M_X2+CovX2X3*M_X1
V_GMA= lambda M_X1,M_X2,CovX1X2,V_X1,V_X2 : V_X1*V_X2+CovX1X2**2+2*CovX1X2*M_X1*M_X2+V_X1*M_X2**2+V_X2*M_X1**2

#Fonction à approximer

f = lambda X_A,X_EL : X_A*(np.exp(-X_EL)-1)


#Initialisation

M_X0 = np.array([[0], [0.005],[7]])
V_X0 = np.array([[(0.2)**2 , 0 , 0], [0 , (0.05)**2 , 0],[0, 0 , (2.5)**2]])

M_Exp0=np.array([M_exp(M_X0[0],V_X0[0,0],M_X0[1],V_X0[1,1],V_X0[1,0])-1])
V_Exp0=np.array([V_exp(M_X0[0],V_X0[0,0],M_X0[1],V_X0[1,1],V_X0[1,0]).item()])
a=(Cov_exp02(M_X0[0],V_X0[0,0],M_X0[1],V_X0[1,1],V_X0[1,0]).item())/V_X0[0,0]
Cov_X0Exp0=np.array([[Cov_exp02(M_X0[0],V_X0[0,0],M_X0[1],V_X0[1,1],V_X0[1,0]).item()],[Cov_exp12(M_X0[0],V_X0[0,0],M_X0[1],V_X0[1,1],V_X0[1,0]).item()],
                     [(a*V_X0[0,2]).item()]])
M_X0Exp0=np.vstack((M_X0,M_Exp0))
V_X0Exp0=np.block([[V_X0,Cov_X0Exp0],[np.transpose(Cov_X0Exp0),V_Exp0]])


M_GMA0=np.array(M_GMA(M_X0Exp0[2],M_X0Exp0[3],V_X0Exp0[2,3]).item())
Cov_X0Exp0GMA0=np.array([[Cov_GMA(M_X0Exp0[2],M_X0Exp0[3],V_X0Exp0[2,0],V_X0Exp0[3,0]).item()],
                         [Cov_GMA(M_X0Exp0[2],M_X0Exp0[3],V_X0Exp0[2,1],V_X0Exp0[3,1]).item()],
                         [Cov_GMA(M_X0Exp0[2],M_X0Exp0[3],V_X0Exp0[2,2],V_X0Exp0[3,2]).item()],
                         [Cov_GMA(M_X0Exp0[2],M_X0Exp0[3],V_X0Exp0[2,3],V_X0Exp0[3,3]).item()]])
V_GMA0=np.array([V_GMA(M_X0Exp0[2],M_X0Exp0[3],V_X0Exp0[2,3],V_X0Exp0[2,2],V_X0Exp0[3,3]).item()])
M_X0Exp0GMA0=np.vstack((M_X0Exp0,M_GMA0))
V_X0Exp0GMA0=np.block([[V_X0Exp0,Cov_X0Exp0GMA0],[np.transpose(Cov_X0Exp0GMA0),V_GMA0]])





#Définition Matrice

A=np.array([[1,1,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,0,0],
            [0,0,0,0,0]])

C=np.array([0,0,0,0,1])

R=0.5**2

#Création des observations

# X_EL_0=stats.norm.rvs(loc=M_X0Exp0GMA0[0],scale=np.sqrt(V_X0Exp0GMA0[0,0]),size=1).item()
# X_ET_0=stats.norm.rvs(loc=M_X0Exp0GMA0[1],scale=np.sqrt(V_X0Exp0GMA0[1,1]),size=1).item()
# X_A_0=stats.norm.rvs(loc=M_X0Exp0GMA0[2],scale=np.sqrt(V_X0Exp0GMA0[2,2]),size=1).item()

# X_0=np.array([[X_EL_0],[X_ET_0],[X_A_0],[0],[f(X_A_0,X_EL_0).item()+stats.norm.rvs(loc=0,scale=np.sqrt(R),size=1).item()]])

#Modele

# Xliste=[X_0]
T=len(df)-1
# T=70
# expreelliste=[f(X_A_0,X_EL_0).item()]
tliste=np.linspace(0,T,T+1)


# for t in range(T):
#     X_t=A@X_0
#     X_t[4]=f(X_t[2],X_t[0]).item()+stats.norm.rvs(loc=0,scale=R,size=1)
#     expreelliste.append(f(X_t[2],X_t[0]).item())
#     Xliste.append(X_t)
#     X_0=X_t

# X_EL_True=[arr[0] for arr in Xliste]
# X_ET_True=[arr[1] for arr in Xliste]
# X_A_True=[arr[2] for arr in Xliste]
# Exp_X_EL_True=[arr[4] for arr in Xliste]

#Affichage du modèle

# fig,ax=plt.subplots(4,1,figsize=(10,3))

# ax[0].plot(tliste,X_EL_True,color='black', linestyle='--')
# ax[0].set_ylabel(r'$X_{EL}$')
# ax[0].set_xlabel(r"t")

# ax[1].plot(tliste,X_ET_True,color='black', linestyle='--')
# ax[1].set_ylabel(r'$X_{ET}$')
# ax[1].set_xlabel(r"t")

# ax[2].plot(tliste,X_A_True,color='black', linestyle='--')
# ax[2].set_ylabel(r'$X_{A}$')
# ax[2].set_xlabel(r"t")

# ax[3].plot(tliste,Exp_X_EL_True,color='red')
# ax[3].plot(tliste, expreelliste, color="black", linestyle='--')
# ax[3].set_ylabel(r'$exp(X_{EL}$)')
# ax[3].set_xlabel(r"t")

# plt.show()

#Fonctions de predictions et d'update pour le Kalman filter

def one_step_predict_exp(A,M_Posterior,V_Posterior):
    M_Prior=A@M_Posterior
    V_Prior=A@V_Posterior@np.transpose(A)
    M_Prior[3]=M_exp(M_Prior[0],V_Prior[0,0],M_Prior[1],V_Prior[1,1],V_Prior[1,0])-1
    V_Prior[3,0]=Cov_exp02(M_Prior[0],V_Prior[0,0],M_Prior[1],V_Prior[1,1],V_Prior[1,0]).item()
    V_Prior[0,3]=V_Prior[3,0]
    V_Prior[3,1]=Cov_exp12(M_Prior[0],V_Prior[0,0],M_Prior[1],V_Prior[1,1],V_Prior[1,0]).item()
    V_Prior[1,3]=V_Prior[3,1]
    a=V_Prior[3,0]/V_Prior[0,0]
    V_Prior[3,2]=a*V_Prior[2,0]
    V_Prior[2,3]=V_Prior[3,2]
    V_Prior[3,3]=V_exp(M_Prior[0],V_Prior[0,0],M_Prior[1],V_Prior[1,1],V_Prior[1,0]).item()
    M_Prior[4]=M_GMA(M_Prior[2],M_Prior[3],V_Prior[2,3]).item()
    V_Prior[4,0]=Cov_GMA(M_Prior[2],M_Prior[3],V_Prior[2,0],V_Prior[3,0]).item()
    V_Prior[0,4]=V_Prior[4,0]
    V_Prior[4,1]=Cov_GMA(M_Prior[2],M_Prior[3],V_Prior[2,1],V_Prior[3,1]).item()
    V_Prior[1,4]=V_Prior[4,1]
    V_Prior[4,2]=Cov_GMA(M_Prior[2],M_Prior[3],V_Prior[2,2],V_Prior[3,2]).item()
    V_Prior[2,4]=V_Prior[4,2]
    V_Prior[4,3]=Cov_GMA(M_Prior[2],M_Prior[3],V_Prior[2,3],V_Prior[3,3]).item()
    V_Prior[3,4]=V_Prior[4,3]
    V_Prior[4,4]=V_GMA(M_Prior[2],M_Prior[3],V_Prior[2,3],V_Prior[2,2],V_Prior[3,3]).item()
    return M_Prior,V_Prior

# M_GMA0=np.array(M_GMA(M_X0Exp0[2],M_X0Exp0[3],V_X0Exp0[2,3]).item())
# Cov_X0Exp0GMA0=np.array([[Cov_GMA(M_X0Exp0[2],M_X0Exp0[3],V_X0Exp0[2,0],V_X0Exp0[3,0]).item()],
#                          [Cov_GMA(M_X0Exp0[2],M_X0Exp0[3],V_X0Exp0[2,1],V_X0Exp0[3,1]).item()],
#                          [Cov_GMA(M_X0Exp0[2],M_X0Exp0[3],V_X0Exp0[2,2],V_X0Exp0[3,2]).item()],
#                          [Cov_GMA(M_X0Exp0[2],M_X0Exp0[3],V_X0Exp0[2,3],V_X0Exp0[3,3]).item()]])
# V_GMA0=np.array([V_GMA(M_X0Exp0[2],M_X0Exp0[3],V_X0Exp0[2,3],V_X0Exp0[2,2],V_X0Exp0[3,3]).item()])


def one_step_update_exp(E_Prior, V_Prior, y_obs, C, R):
    G = C @ V_Prior @ np.reshape(C, (-1, 1)) + R
    K = V_Prior @ np.reshape(C, (-1, 1)) * (1 / G)
    ychap = C @ E_Prior
    rt = y_obs - ychap
    V_Posterior = (np.eye(len(C)) - K @ np.reshape(C, (1, 5))) @ V_Prior
    V_Posterior = 0.5 * (V_Posterior + V_Posterior.T)

    E_Posterior = E_Prior + K * rt
    return E_Posterior, V_Posterior

LL = 0  # Initialisation du likelihood

# Kalman Filter avec .copy()
M_Prior, V_Prior = one_step_predict_exp(A, M_X0Exp0GMA0, V_X0Exp0GMA0)
# LL += stats.norm.logpdf(Exp_X_EL_True[1], loc=M_Prior[2], scale=np.sqrt(V_Prior[2, 2]))
if not np.isnan(df.iloc[1]):
    M_Posterior, V_Posterior = one_step_update_exp(M_Prior, V_Prior, df.iloc[1], C, R)
else:
    M_Posterior,V_Posterior=M_Prior,V_Prior
# M_Posterior, V_Posterior = one_step_update_exp(M_Prior, V_Prior, df.iloc[1], C, R)

M_Xliste = [M_X0Exp0GMA0.copy()]
M_Xliste.append(M_Posterior.copy())
V_Xliste = [V_X0Exp0GMA0.copy()]
V_Xliste.append(V_Posterior.copy())
V_Xlistediag = [np.diag(V_X0Exp0GMA0)]
V_Xlistediag.append(np.diag(V_Posterior))

M_Xliste_prior = [M_X0Exp0GMA0.copy()]
M_Xliste_prior.append(M_Prior.copy())
V_Xliste_prior = [V_X0Exp0GMA0.copy()]
V_Xliste_prior.append(V_Prior.copy())
V_Xlistediag_prior = [np.diag(V_X0Exp0GMA0)]
V_Xlistediag_prior.append(np.diag(V_Prior))

for i in range(2, T + 1):
    M_Prior, V_Prior = one_step_predict_exp(A, M_Posterior, V_Posterior)
    M_Xliste_prior.append(M_Prior.copy())
    V_Xliste_prior.append(V_Prior.copy())
    V_Xlistediag_prior.append(np.diag(V_Prior))
    # LL += stats.norm.logpdf(colonne_7[i], loc=M_Prior[4], scale=np.sqrt(V_Prior[4, 4]))
    if not np.isnan(df.iloc[i]):
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
x_apredit = [arr[2].item() for arr in M_Xliste]
explistepredit = [arr[4].item() for arr in M_Xliste]

Vx_eLpredit = [arr[0] for arr in V_Xlistediag]
Vx_eTpredit = [arr[1] for arr in V_Xlistediag]
Vx_apredit = [arr[2] for arr in V_Xlistediag]
Vexplistepredit = [arr[4] for arr in V_Xlistediag]

x_eLprior = [arr[0].item() for arr in M_Xliste_prior]
x_eTprior = [arr[1].item() for arr in M_Xliste_prior]
x_aprior = [arr[2].item() for arr in M_Xliste_prior]
exp_prior = [arr[4].item() for arr in M_Xliste_prior]

Vx_eLprior = [arr[0] for arr in V_Xlistediag_prior]
Vx_eTprior = [arr[1] for arr in V_Xlistediag_prior]
Vx_aprior = [arr[2] for arr in V_Xlistediag_prior]
Vexp_prior = [arr[4] for arr in V_Xlistediag_prior]

# # Correction des longueurs
# tliste_prior = tliste[:len(x_eLprior)]

##Smoother
A_lin = A  # Sous-matrice pour le smoother (3x3)

M_smooth_base = [None] * len(M_Xliste)
V_smooth_base = [None] * len(V_Xliste)

# Initialisation avec le dernier posterior
M_smooth_base[-1] = M_Xliste[-1].copy()
V_smooth_base[-1] = V_Xliste[-1].copy()



# Lissage backward
for t in range(T - 1, -1, -1):
    # Prédictions et filtres
    M_filt = M_Xliste[t]
    V_filt = V_Xliste[t]
    M_pred = M_Xliste_prior[t + 1]
    V_pred = V_Xliste_prior[t + 1]

    # Gain RTS sur les 3 premières dimensions
    A_lin = A[:3, :3]
    G = V_filt[:3, :3] @ A_lin.T @ np.linalg.inv(V_pred[:3, :3])

    # Lissage sur les 3 premières dimensions uniquement
    M_smooth_base[t] = M_filt.copy()
    M_smooth_base[t][:3] += G @ (M_smooth_base[t + 1][:3] - M_pred[:3])

    V_smooth_base[t] = V_filt.copy()
    V_smooth_base[t][:3, :3] += G @ (V_smooth_base[t + 1][:3, :3] - V_pred[:3, :3]) @ G.T

def smooth_extended(M_smooth_base, V_smooth_base):
    M_smooth_complete = M_smooth_base.copy()
    V_smooth_complete = V_smooth_base.copy()

    # Calcul de la composante 3 (exp)
    M_smooth_complete[3] = M_exp(M_smooth_complete[0], V_smooth_complete[0, 0],
                                 M_smooth_complete[1], V_smooth_complete[1, 1],
                                 V_smooth_complete[1, 0]) - 1

    V_smooth_complete[3, 0] = Cov_exp02(M_smooth_complete[0], V_smooth_complete[0, 0],
                                        M_smooth_complete[1], V_smooth_complete[1, 1],
                                        V_smooth_complete[1, 0]).item()
    V_smooth_complete[0, 3] = V_smooth_complete[3, 0]

    V_smooth_complete[3, 1] = Cov_exp12(M_smooth_complete[0], V_smooth_complete[0, 0],
                                        M_smooth_complete[1], V_smooth_complete[1, 1],
                                        V_smooth_complete[1, 0]).item()
    V_smooth_complete[1, 3] = V_smooth_complete[3, 1]

    a = V_smooth_complete[3, 0] / V_smooth_complete[0, 0]
    V_smooth_complete[3, 2] = a * V_smooth_complete[2, 0]
    V_smooth_complete[2, 3] = V_smooth_complete[3, 2]

    V_smooth_complete[3, 3] = V_exp(M_smooth_complete[0], V_smooth_complete[0, 0],
                                    M_smooth_complete[1], V_smooth_complete[1, 1],
                                    V_smooth_complete[1, 0]).item()

    # Calcul de la composante 4 (amplitude * exp)
    M_smooth_complete[4] = M_GMA(M_smooth_complete[2], M_smooth_complete[3],
                                 V_smooth_complete[2, 3]).item()

    V_smooth_complete[4, 0] = Cov_GMA(M_smooth_complete[2], M_smooth_complete[3],
                                      V_smooth_complete[2, 0], V_smooth_complete[3, 0]).item()
    V_smooth_complete[0, 4] = V_smooth_complete[4, 0]

    V_smooth_complete[4, 1] = Cov_GMA(M_smooth_complete[2], M_smooth_complete[3],
                                      V_smooth_complete[2, 1], V_smooth_complete[3, 1]).item()
    V_smooth_complete[1, 4] = V_smooth_complete[4, 1]

    V_smooth_complete[4, 2] = Cov_GMA(M_smooth_complete[2], M_smooth_complete[3],
                                      V_smooth_complete[2, 2], V_smooth_complete[3, 2]).item()
    V_smooth_complete[2, 4] = V_smooth_complete[4, 2]

    V_smooth_complete[4, 3] = Cov_GMA(M_smooth_complete[2], M_smooth_complete[3],
                                      V_smooth_complete[2, 3], V_smooth_complete[3, 3]).item()
    V_smooth_complete[3, 4] = V_smooth_complete[4, 3]

    V_smooth_complete[4, 4] = V_GMA(M_smooth_complete[2], M_smooth_complete[3],
                                    V_smooth_complete[2, 3], V_smooth_complete[2, 2],
                                    V_smooth_complete[3, 3]).item()

    return M_smooth_complete, V_smooth_complete

M_smooth_full = []
V_smooth_full = []

for t in range(len(M_smooth_base)):
    M_full, V_full = smooth_extended(M_smooth_base[t], V_smooth_base[t])
    M_smooth_full.append(M_full)
    V_smooth_full.append(V_full)



x_eLsmooth = [arr[0].item() for arr in M_smooth_full]
x_eTsmooth = [arr[1].item() for arr in M_smooth_full]
x_asmooth = [arr[2].item() for arr in M_smooth_full]
expsmooth = [arr[4].item() for arr in M_smooth_full]

Vx_eLsmooth = [np.diag(v)[0] for v in V_smooth_full]
Vx_eTsmooth = [np.diag(v)[1] for v in V_smooth_full]
Vx_asmooth = [np.diag(v)[2] for v in V_smooth_full]
Vexp_smooth = [np.diag(v)[4] for v in V_smooth_full]

fig, ax = plt.subplots(4, 1, figsize=(10, 3))

# X_EL
# ax[0].plot(tliste, X_EL_True, color='black', linestyle='--', label='True')
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

# X_a
# ax[2].plot(tliste, X_A_True, color='black', linestyle='--')
ax[2].plot(tliste, x_apredit, color='blue')
ax[2].plot(tliste, x_asmooth, color='cyan', label='Smooth')
ax[2].fill_between(tliste, x_apredit - np.sqrt(Vx_apredit), x_apredit + np.sqrt(Vx_apredit),
                   color='blue', alpha=0.2)
ax[2].fill_between(tliste, x_asmooth - np.sqrt(Vx_asmooth), x_asmooth + np.sqrt(Vx_asmooth),
                   color='cyan', alpha=0.4)
ax[2].plot(tliste, x_aprior, color='purple', linestyle=':')
ax[2].fill_between(tliste, x_aprior - np.sqrt(Vx_aprior), x_aprior + np.sqrt(Vx_aprior),
                   color='purple', alpha=0.15)
ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax[2].set_ylabel(r'$X_{A}$')


# exp(X_EL)
# ax[3].plot(tliste, expreelliste, color="black", linestyle='--')
ax[3].plot(tliste, df, color='red')
ax[3].scatter(tliste, df, color='red')
ax[3].plot(tliste, explistepredit, color='blue')
ax[3].fill_between(tliste, explistepredit - np.sqrt(Vexplistepredit), explistepredit + np.sqrt(Vexplistepredit),
                   color='blue', alpha=0.2)
ax[3].plot(tliste, expsmooth, color='cyan')
ax[3].fill_between(tliste, expsmooth - np.sqrt(Vexp_smooth), expsmooth + np.sqrt(Vexp_smooth),
                   color='cyan', alpha=0.4)
ax[3].plot(tliste, exp_prior, color='purple', linestyle=':')
ax[3].fill_between(tliste, exp_prior - np.sqrt(Vexp_prior), exp_prior + np.sqrt(Vexp_prior),
                   color='purple', alpha=0.15)
ax[3].set_ylabel(r'$X_Aexp(X_{EL})$')
ax[3].set_xlabel(r"t")

plt.tight_layout()
(plt.show())

