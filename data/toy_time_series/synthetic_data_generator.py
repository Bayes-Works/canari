import os
import numpy as np
import csv
from scipy import stats
from matplotlib import pyplot as plt, cm
from datetime import datetime, timedelta

np.random.seed(42)
T = 208
espace = np.linspace(0, T - 1, T)
X_A_true = 10
X_ET_true = 0.015
sigma_v = 0.05
X_local_level = 2
X_local_trend = -0.005
valeur_exp = np.zeros(T)
X_local_level_liste = np.zeros(T)
X_EL_true_level = np.zeros(T)
date_debut = datetime(2023, 1, 1)
dates = [date_debut + timedelta(weeks=t) for t in range(T)]

for t in range(T):  # need to be adapted to the components you want in the time series
    valeur_exp[t] = (
        X_A_true * (np.exp(-t * X_ET_true) - 1)
        + 2 * np.cos(2 * np.pi * t / 52)
        + stats.norm.rvs(loc=0, scale=sigma_v, size=1)
        + X_local_level
        + X_local_trend * t
    )
    X_local_level_liste[t] = X_local_level + X_local_trend * t
    X_EL_true_level[t] = X_ET_true * t

plt.plot(espace, valeur_exp, color="red")
plt.show()

script_dir = os.path.dirname(__file__)

output_path = os.path.join(
    script_dir, "name_of_file.csv"
)  # Change the name of the file here

with open(output_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(
        [
            "temps",
            "exponential",
            "X_EL",
            "X_ET",
            "X_A",
            "X_local_level",
            "X_local_trend",
        ]  # name of each column in the csv files
    )
    for t in range(T):  # the content of each column
        writer.writerow(
            [
                dates[t].strftime("%Y-%m-%d"),
                valeur_exp[t],
                X_EL_true_level[t],
                X_ET_true,
                X_A_true,
                X_local_level_liste[t],
                X_local_trend,
            ]
        )
