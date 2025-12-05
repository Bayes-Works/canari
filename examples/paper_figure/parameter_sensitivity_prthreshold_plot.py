# Read CSV file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import ast

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          'lines.linewidth' : 1,
          }
plt.rcParams.update(params)
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

df_il = pd.read_csv("saved_results/prob_eva/syn_simple_regen_ts_results_il_val.csv")
df_il2 = pd.read_csv("saved_results/prob_eva/syn_simple_regen_ts_results_il_threshold1_5.csv")
df_il3 = pd.read_csv("saved_results/prob_eva/syn_simple_regen_ts_results_il_threshold2_0.csv")
df_il4 = pd.read_csv("saved_results/prob_eva/syn_simple_regen_ts_results_il_threshold3_0.csv")
df_il5 = pd.read_csv("saved_results/prob_eva/syn_simple_regen_ts_results_il_threshold5_0.csv")
df_il6 = pd.read_csv("saved_results/prob_eva/syn_simple_regen_ts_results_il_threshold10_0.csv")

# Multiply the df_il["anomaly_magnitude"] by 52
df_il["anomaly_magnitude"] = np.abs(df_il["anomaly_magnitude"]) * 52
df_il2["anomaly_magnitude"] = np.abs(df_il2["anomaly_magnitude"]) * 52
df_il3["anomaly_magnitude"] = np.abs(df_il3["anomaly_magnitude"]) * 52
df_il4["anomaly_magnitude"] = np.abs(df_il4["anomaly_magnitude"]) * 52
df_il5["anomaly_magnitude"] = np.abs(df_il5["anomaly_magnitude"]) * 52
df_il6["anomaly_magnitude"] = np.abs(df_il6["anomaly_magnitude"]) * 52

df_il["anomaly_detected_index"] = df_il["anomaly_detected_index"].apply(ast.literal_eval)
df_il2["anomaly_detected_index"] = df_il2["anomaly_detected_index"].apply(ast.literal_eval)
df_il3["anomaly_detected_index"] = df_il3["anomaly_detected_index"].apply(ast.literal_eval)
df_il4["anomaly_detected_index"] = df_il4["anomaly_detected_index"].apply(ast.literal_eval)
df_il5["anomaly_detected_index"] = df_il5["anomaly_detected_index"].apply(ast.literal_eval)
df_il6["anomaly_detected_index"] = df_il6["anomaly_detected_index"].apply(ast.literal_eval)

# Compute detection_rate, for each anomaly magnitude, when df_il["detection_time"] == 260, it means that the anomaly is not detected
df_il["detection_rate"] = df_il["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)
df_il2["detection_rate"] = df_il2["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)
df_il3["detection_rate"] = df_il3["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)
df_il4["detection_rate"] = df_il4["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)
df_il5["detection_rate"] = df_il5["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)
df_il6["detection_rate"] = df_il6["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)

# Compute the false alarm rate for each method
anm_time_range_begin = 250
# Sum all the anomaly_start_index
sum_anm_start_index = df_il["anomaly_start_index"].sum() - anm_time_range_begin * df_il.shape[0]
false_alarms_il = 0
neg_detect_indices = []
for i in range(df_il.shape[0]):
    if df_il.iloc[i]["detection_time"] < 0:
        false_alarms_il += len(df_il.iloc[i]["anomaly_detected_index"])
        neg_detect_indices.append(i)
# Delete the rows with negative detection time
df_il = df_il.drop(index=neg_detect_indices)
false_alarm_rate_il = false_alarms_il * 10 / (sum_anm_start_index/52)
# print("False alarm rate for IL: ", false_alarm_rate_il, "per 10 years")

false_alarms_skf = 0
neg_detect_indices = []
for i in range(df_il2.shape[0]):
    if df_il2.iloc[i]["detection_time"] < 0:
        false_alarms_skf += len(df_il2.iloc[i]["anomaly_detected_index"])
        neg_detect_indices.append(i)
df_il2 = df_il2.drop(index=neg_detect_indices)
false_alarm_rate_skf = false_alarms_skf * 10 / (sum_anm_start_index/52)
# print("False alarm rate for SKF: ", false_alarm_rate_skf, "per 10 years")

false_alarms_mp = 0
neg_detect_indices = []
for i in range(df_il3.shape[0]):
    if df_il3.iloc[i]["detection_time"] < 0:
        false_alarms_mp += len(df_il3.iloc[i]["anomaly_detected_index"])
        neg_detect_indices.append(i)
df_il3 = df_il3.drop(index=neg_detect_indices)
false_alarm_rate_mp = false_alarms_mp * 10 / (sum_anm_start_index/52)
# print("False alarm rate for MP: ", false_alarm_rate_mp, "per 10 years")

false_alarms_prophet = 0
neg_detect_indices = []
for i in range(df_il4.shape[0]):
    if df_il4.iloc[i]["detection_time"] < 0:
        false_alarms_prophet += len(df_il4.iloc[i]["anomaly_detected_index"])
        neg_detect_indices.append(i)
df_il4 = df_il4.drop(index=neg_detect_indices)
false_alarm_rate_prophet = false_alarms_prophet * 10 / (sum_anm_start_index/52)
# print("False alarm rate for Prophet: ", false_alarm_rate_prophet, "per 10 years")

false_alarms_il5 = 0
neg_detect_indices = []
for i in range(df_il5.shape[0]):
    if df_il5.iloc[i]["detection_time"] < 0:
        false_alarms_prophet += len(df_il5.iloc[i]["anomaly_detected_index"])
        neg_detect_indices.append(i)
df_il5 = df_il5.drop(index=neg_detect_indices)
false_alarm_rate_prophet = false_alarms_prophet * 10 / (sum_anm_start_index/52)
# print("False alarm rate for Prophet: ", false_alarm_rate_prophet, "per 10 years")

false_alarms_il6 = 0
neg_detect_indices = []
for i in range(df_il6.shape[0]):
    if df_il6.iloc[i]["detection_time"] < 0:
        false_alarms_il6 += len(df_il6.iloc[i]["anomaly_detected_index"])
        neg_detect_indices.append(i)
df_il6 = df_il6.drop(index=neg_detect_indices)
false_alarm_rate_il6 = false_alarms_il6 * 10 / (sum_anm_start_index/52)
# print("False alarm rate for Prophet: ", false_alarm_rate_prophet, "per 10 years")

# Get anomaly_detected_index from df_il4["anomaly_detected_index"]
df_il["alarms_num"] = df_il["anomaly_detected_index"].apply(
    lambda x: len(x) if len(x) > 0 else 0
)
df_il2["alarms_num"] = df_il2["anomaly_detected_index"].apply(
    lambda x: len(x) if len(x) > 0 else 0
)
df_il3["alarms_num"] = df_il3["anomaly_detected_index"].apply(
    lambda x: len(x) if len(x) > 0 else 0
)
df_il4["alarms_num"] = df_il4["anomaly_detected_index"].apply(
    lambda x: len(x) if len(x) > 0 else 0
)
df_il5["alarms_num"] = df_il5["anomaly_detected_index"].apply(
    lambda x: len(x) if len(x) > 0 else 0
)
df_il6["alarms_num"] = df_il6["anomaly_detected_index"].apply(
    lambda x: len(x) if len(x) > 0 else 0
)

# Set alarms_num to 0 if "detection_rate" is 0
df_il.loc[df_il["detection_rate"] == 0, "alarms_num"] = 0
df_il2.loc[df_il2["detection_rate"] == 0, "alarms_num"] = 0
df_il3.loc[df_il3["detection_rate"] == 0, "alarms_num"] = 0
df_il4.loc[df_il4["detection_rate"] == 0, "alarms_num"] = 0
df_il5.loc[df_il5["detection_rate"] == 0, "alarms_num"] = 0
df_il6.loc[df_il6["detection_rate"] == 0, "alarms_num"] = 0

# For the same anomaly magnitude, compute the mean and variance of df_il["mse_LL"], df_il["mse_LT"], and df_il["detection_time"], stored them in a new dataframe
df_il_mean = df_il.groupby("anomaly_magnitude").agg(
    {
        "mse_LL": ["mean", "std"],
        "mse_LT": ["mean", "std"],
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
        "alarms_num": ["mean", "std"],
    }
)
df_il2_mean = df_il2.groupby("anomaly_magnitude").agg(
    {
        "mse_LL": ["mean", "std"],
        "mse_LT": ["mean", "std"],
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
        "alarms_num": ["mean", "std"],
    }
)
df_il3_mean = df_il3.groupby("anomaly_magnitude").agg(
    {
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
        "alarms_num": ["mean", "std"],
    }
)

df_il4_mean = df_il4.groupby("anomaly_magnitude").agg(
    {
        "mse_LL": ["mean", "std"],
        "mse_LT": ["mean", "std"],
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
        "alarms_num": ["mean", "std"],
    }
)
df_il5_mean = df_il5.groupby("anomaly_magnitude").agg(
    {
        "mse_LL": ["mean", "std"],
        "mse_LT": ["mean", "std"],
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
        "alarms_num": ["mean", "std"],
    }
)
df_il6_mean = df_il6.groupby("anomaly_magnitude").agg(
    {
        "mse_LL": ["mean", "std"],
        "mse_LT": ["mean", "std"],
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
        "alarms_num": ["mean", "std"],
    }
)

# Plot the mean and std of df_il["mse_LL"], df_il["mse_LT"], and df_il["detection_time"] for each anomaly magnitude
# fig, ax = plt.subplots(3, 1, figsize=(3, 5), constrained_layout=True)
fig, ax = plt.subplots(3, 1, figsize=(3, 2.5), constrained_layout=True)


# Plot for detection_time
ax[0].plot(df_il_mean.index, df_il_mean["detection_time"]["mean"], label=r"\textbf{$\gamma$=1.1}")
ax[0].fill_between(
    df_il_mean.index,
    df_il_mean["detection_time"]["mean"] - df_il_mean["detection_time"]["std"],
    df_il_mean["detection_time"]["mean"] + df_il_mean["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_il2_mean.index, df_il2_mean["detection_time"]["mean"], label="1.5")
ax[0].fill_between(
    df_il2_mean.index,
    df_il2_mean["detection_time"]["mean"] - df_il2_mean["detection_time"]["std"],
    df_il2_mean["detection_time"]["mean"] + df_il2_mean["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_il3_mean.index, df_il3_mean["detection_time"]["mean"], label="2")
ax[0].fill_between(
    df_il3_mean.index,
    df_il3_mean["detection_time"]["mean"] - df_il3_mean["detection_time"]["std"],
    df_il3_mean["detection_time"]["mean"] + df_il3_mean["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_il4_mean.index, df_il4_mean["detection_time"]["mean"], label="3")
ax[0].fill_between(
    df_il4_mean.index,
    df_il4_mean["detection_time"]["mean"] - df_il4_mean["detection_time"]["std"],
    df_il4_mean["detection_time"]["mean"] + df_il4_mean["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_il5_mean.index, df_il5_mean["detection_time"]["mean"], label="5")
ax[0].fill_between(
    df_il5_mean.index,
    df_il5_mean["detection_time"]["mean"] - df_il5_mean["detection_time"]["std"],
    df_il5_mean["detection_time"]["mean"] + df_il5_mean["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_il6_mean.index, df_il6_mean["detection_time"]["mean"], label="10", color = "tab:brown")
ax[0].fill_between(
    df_il6_mean.index,
    df_il6_mean["detection_time"]["mean"] - df_il6_mean["detection_time"]["std"],
    df_il6_mean["detection_time"]["mean"] + df_il6_mean["detection_time"]["std"],
    alpha=0.2,
    color = "tab:brown"
)
ax[0].set_ylabel(r"$\Delta_t(\mathrm{y})$")
# ax[2].set_yticks([0, 52, 104, 156, 208, 260])
ax[0].set_yticks([0, 52, 104, 156])
ax[0].set_yticklabels([0, 1, 2, 3])
ax[0].set_xscale('log')
ax[0].set_ylim(0, 52 * 3.05)
ax[0].set_xticklabels([])
# ax[0].legend(ncol=2)
# Show the legend outside the plot
# ax[0].legend(bbox_to_anchor=(0, 2.5), loc='upper left', borderaxespad=0., ncol=2)

# Plot for detection_rate
ax[1].plot(df_il_mean.index, df_il_mean["detection_rate"]["mean"], label="IL")
ax[1].plot(df_il2_mean.index, df_il2_mean["detection_rate"]["mean"], label="SKF")
ax[1].plot(df_il3_mean.index, df_il3_mean["detection_rate"]["mean"], label="Matrix profile")
ax[1].plot(df_il4_mean.index, df_il4_mean["detection_rate"]["mean"], label="Prophet")
ax[1].plot(df_il5_mean.index, df_il5_mean["detection_rate"]["mean"], label="IL5")
ax[1].plot(df_il6_mean.index, df_il6_mean["detection_rate"]["mean"], label="IL10", color = "tab:brown")
ax[1].set_ylabel(r"$\mathcal{P}_{\mathtt{DET}}$")
ax[1].set_ylim(-0.05, 1.05)
ax[1].set_yticks([0, 0.5, 1])
ax[1].set_xscale('log')
ax[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax[1].set_xticklabels([])

# Plot the number of false alarms for prophet
ax[2].plot(df_il_mean.index, df_il_mean["alarms_num"]["mean"], label="2-64-32", color = "tab:blue")
ax[2].fill_between(
    df_il_mean.index,
    df_il_mean["alarms_num"]["mean"] - df_il_mean["alarms_num"]["std"],
    df_il_mean["alarms_num"]["mean"] + df_il_mean["alarms_num"]["std"],
    alpha=0.2,
    color = "tab:blue"
)
ax[2].plot(df_il2_mean.index, df_il2_mean["alarms_num"]["mean"], label="1-32", color = "tab:orange")
ax[2].fill_between(
    df_il2_mean.index,
    df_il2_mean["alarms_num"]["mean"] - df_il2_mean["alarms_num"]["std"],
    df_il2_mean["alarms_num"]["mean"] + df_il2_mean["alarms_num"]["std"],
    alpha=0.2,
    color = "tab:orange"
)
ax[2].plot(df_il3_mean.index, df_il3_mean["alarms_num"]["mean"], label="2-256", color = "tab:green")
ax[2].fill_between(
    df_il3_mean.index,
    df_il3_mean["alarms_num"]["mean"] - df_il3_mean["alarms_num"]["std"],
    df_il3_mean["alarms_num"]["mean"] + df_il3_mean["alarms_num"]["std"],
    alpha=0.2,
    color = "tab:green"
)
ax[2].plot(df_il4_mean.index, df_il4_mean["alarms_num"]["mean"], label="Prophet", color = "tab:red")
ax[2].fill_between(
    df_il4_mean.index,
    df_il4_mean["alarms_num"]["mean"] - df_il4_mean["alarms_num"]["std"],
    df_il4_mean["alarms_num"]["mean"] + df_il4_mean["alarms_num"]["std"],
    alpha=0.2,
    color = "tab:red"
)
ax[2].plot(df_il5_mean.index, df_il5_mean["alarms_num"]["mean"], label="IL5", color = "tab:purple")
ax[2].fill_between(
    df_il5_mean.index,
    df_il5_mean["alarms_num"]["mean"] - df_il5_mean["alarms_num"]["std"],
    df_il5_mean["alarms_num"]["mean"] + df_il5_mean["alarms_num"]["std"],
    alpha=0.2,
    color = "tab:purple"
)
ax[2].plot(df_il6_mean.index, df_il6_mean["alarms_num"]["mean"], label="IL10", color = "tab:brown")
ax[2].fill_between(
    df_il6_mean.index,
    df_il6_mean["alarms_num"]["mean"] - df_il6_mean["alarms_num"]["std"],
    df_il6_mean["alarms_num"]["mean"] + df_il6_mean["alarms_num"]["std"],
    alpha=0.2,
    color = "tab:brown"
)
ax[2].set_xlabel("Anomaly Magnitude (unit/$y$)")
ax[2].set_ylabel(r"$\#_{\mathtt{ALM}}$")
# ax[2].set_yscale('symlog', linthresh=1e2)
ax[2].set_yscale('symlog')
ax[2].set_ylim(-0.01, 2e1)
ax[2].set_xscale('log')

fig.align_ylabels(ax)

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.3)
plt.savefig('syn_ts_results_legend.png', dpi=300)
plt.show()