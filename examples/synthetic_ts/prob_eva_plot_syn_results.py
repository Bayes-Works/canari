# Read CSV file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import ast

params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params)
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

df_il = pd.read_csv("saved_results/prob_eva/syn_simple_regen_ts_results_il.csv")
df_skf = pd.read_csv("saved_results/prob_eva/syn_simple_regen_ts_results_skf.csv")
df_mp = pd.read_csv("saved_results/prob_eva/syn_simple_regen_ts_results_mp.csv")
df_prophet = pd.read_csv("saved_results/prob_eva/syn_simple_regen_ts_results_prophet_online.csv")

# Multiply the df_il["anomaly_magnitude"] by 52
df_il["anomaly_magnitude"] = np.abs(df_il["anomaly_magnitude"]) * 52
df_skf["anomaly_magnitude"] = np.abs(df_skf["anomaly_magnitude"]) * 52
df_mp["anomaly_magnitude"] = np.abs(df_mp["anomaly_magnitude"]) * 52
df_prophet["anomaly_magnitude"] = np.abs(df_prophet["anomaly_magnitude"]) * 52

df_il["anomaly_detected_index"] = df_il["anomaly_detected_index"].apply(ast.literal_eval)
df_skf["anomaly_detected_index"] = df_skf["anomaly_detected_index"].apply(ast.literal_eval)
df_mp["anomaly_detected_index"] = df_mp["anomaly_detected_index"].apply(ast.literal_eval)
df_prophet["anomaly_detected_index"] = df_prophet["anomaly_detected_index"].apply(ast.literal_eval)

# Compute detection_rate, for each anomaly magnitude, when df_il["detection_time"] == 260, it means that the anomaly is not detected
df_il["detection_rate"] = df_il["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)
df_skf["detection_rate"] = df_skf["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)
df_mp["detection_rate"] = df_mp["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)
df_prophet["detection_rate"] = df_prophet["detection_time"].apply(
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
print("False alarm rate for IL: ", false_alarm_rate_il, "per 10 years")

false_alarms_skf = 0
neg_detect_indices = []
for i in range(df_skf.shape[0]):
    if df_skf.iloc[i]["detection_time"] < 0:
        false_alarms_skf += len(df_skf.iloc[i]["anomaly_detected_index"])
        neg_detect_indices.append(i)
df_skf = df_skf.drop(index=neg_detect_indices)
false_alarm_rate_skf = false_alarms_skf * 10 / (sum_anm_start_index/52)
print("False alarm rate for SKF: ", false_alarm_rate_skf, "per 10 years")

false_alarms_mp = 0
neg_detect_indices = []
for i in range(df_mp.shape[0]):
    if df_mp.iloc[i]["detection_time"] < 0:
        false_alarms_mp += len(df_mp.iloc[i]["anomaly_detected_index"])
        neg_detect_indices.append(i)
df_mp = df_mp.drop(index=neg_detect_indices)
false_alarm_rate_mp = false_alarms_mp * 10 / (sum_anm_start_index/52)
print("False alarm rate for MP: ", false_alarm_rate_mp, "per 10 years")

false_alarms_prophet = 0
neg_detect_indices = []
for i in range(df_prophet.shape[0]):
    if df_prophet.iloc[i]["detection_time"] < 0:
        false_alarms_prophet += len(df_prophet.iloc[i]["anomaly_detected_index"])
        neg_detect_indices.append(i)
df_prophet = df_prophet.drop(index=neg_detect_indices)
false_alarm_rate_prophet = false_alarms_prophet * 10 / (sum_anm_start_index/52)
print("False alarm rate for Prophet: ", false_alarm_rate_prophet, "per 10 years")

df_il["false_alarms_num"] = df_il["anomaly_detected_index"].apply(
    lambda x: len(x) if len(x) > 0 else 0
)
df_skf["false_alarms_num"] = df_skf["anomaly_detected_index"].apply(
    lambda x: len(x) if len(x) > 0 else 0
)
df_mp["false_alarms_num"] = df_mp["anomaly_detected_index"].apply(
    lambda x: len(x) if len(x) > 0 else 0
)
df_prophet["false_alarms_num"] = df_prophet["anomaly_detected_index"].apply(
    lambda x: len(x) if len(x) > 0 else 0
)

# For the same anomaly magnitude, compute the mean and variance of df_il["mse_LL"], df_il["mse_LT"], and df_il["detection_time"], stored them in a new dataframe
df_il_mean = df_il.groupby("anomaly_magnitude").agg(
    {
        "mse_LL": ["mean", "std"],
        "mse_LT": ["mean", "std"],
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
        "false_alarms_num": ["mean", "std"],
    }
)
df_skf_mean = df_skf.groupby("anomaly_magnitude").agg(
    {
        "mse_LL": ["mean", "std"],
        "mse_LT": ["mean", "std"],
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
        "false_alarms_num": ["mean", "std"],
    }
)
df_skf_whitenoise_mean = df_mp.groupby("anomaly_magnitude").agg(
    {
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
        "false_alarms_num": ["mean", "std"],
    }
)

df_prophet_mean = df_prophet.groupby("anomaly_magnitude").agg(
    {
        "mse_LL": ["mean", "std"],
        "mse_LT": ["mean", "std"],
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
        "false_alarms_num": ["mean", "std"],
    }
)

# Plot the mean and std of df_il["mse_LL"], df_il["mse_LT"], and df_il["detection_time"] for each anomaly magnitude
fig, ax = plt.subplots(3, 1, figsize=(10, 6))
# # Plot for mse_LL
# ax[0].plot(df_il_mean.index, df_il_mean["mse_LL"]["mean"], label="IL")
# ax[0].fill_between(
#     df_il_mean.index,
#     df_il_mean["mse_LL"]["mean"] - df_il_mean["mse_LL"]["std"],
#     df_il_mean["mse_LL"]["mean"] + df_il_mean["mse_LL"]["std"],
#     alpha=0.2,
# )

# ax[0].plot(df_skf_mean.index, df_skf_mean["mse_LL"]["mean"], label="SKF")
# ax[0].fill_between(
#     df_skf_mean.index,
#     df_skf_mean["mse_LL"]["mean"] - df_skf_mean["mse_LL"]["std"],
#     df_skf_mean["mse_LL"]["mean"] + df_skf_mean["mse_LL"]["std"],
#     alpha=0.2,
# )

# ax[0].plot(df_skf_whitenoise_mean.index, df_skf_whitenoise_mean["mse_LL"]["mean"], label="SKF (whitenoise)")
# ax[0].fill_between(
#     df_skf_whitenoise_mean.index,
#     df_skf_whitenoise_mean["mse_LL"]["mean"] - df_skf_whitenoise_mean["mse_LL"]["std"],
#     df_skf_whitenoise_mean["mse_LL"]["mean"] + df_skf_whitenoise_mean["mse_LL"]["std"],
#     alpha=0.2,
# )
# ax[0].plot(df_prophet_mean.index, df_prophet_mean["mse_LL"]["mean"], label="Prophet")
# ax[0].fill_between(
#     df_prophet_mean.index,
#     df_prophet_mean["mse_LL"]["mean"] - df_prophet_mean["mse_LL"]["std"],
#     df_prophet_mean["mse_LL"]["mean"] + df_prophet_mean["mse_LL"]["std"],
#     alpha=0.2,
# )
# ax[0].set_ylabel(r"MSE($x^{\mathrm{LL}}$)")
# # ax[0].set_ylabel(r"MAPE($x^{\mathrm{LL}}$)")
# ax[0].legend(ncol=2)
# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
# ax[0].set_xticklabels([])

# Plot for mse_LT
# ax[1].plot(df_il_mean.index, df_il_mean["mse_LT"]["mean"], label="IL")
# ax[1].fill_between(
#     df_il_mean.index,
#     df_il_mean["mse_LT"]["mean"] - df_il_mean["mse_LT"]["std"],
#     df_il_mean["mse_LT"]["mean"] + df_il_mean["mse_LT"]["std"],
#     alpha=0.2,
# )
# ax[1].plot(df_skf_mean.index, df_skf_mean["mse_LT"]["mean"], label="SKF")
# ax[1].fill_between(
#     df_skf_mean.index,
#     df_skf_mean["mse_LT"]["mean"] - df_skf_mean["mse_LT"]["std"],
#     df_skf_mean["mse_LT"]["mean"] + df_skf_mean["mse_LT"]["std"],
#     alpha=0.2,
# )
# ax[1].plot(df_skf_whitenoise_mean.index, df_skf_whitenoise_mean["mse_LT"]["mean"], label="SKF (whitenoise)")
# ax[1].fill_between(
#     df_skf_whitenoise_mean.index,
#     df_skf_whitenoise_mean["mse_LT"]["mean"] - df_skf_whitenoise_mean["mse_LT"]["std"],
#     df_skf_whitenoise_mean["mse_LT"]["mean"] + df_skf_whitenoise_mean["mse_LT"]["std"],
#     alpha=0.2,
# )
# ax[1].plot(df_prophet_mean.index, df_prophet_mean["mse_LT"]["mean"], label="Prophet")
# ax[1].fill_between(
#     df_prophet_mean.index,
#     df_prophet_mean["mse_LT"]["mean"] - df_prophet_mean["mse_LT"]["std"],
#     df_prophet_mean["mse_LT"]["mean"] + df_prophet_mean["mse_LT"]["std"],
#     alpha=0.2,
# )
# ax[1].set_ylabel(r"MSE($x^{\mathrm{LT}}$)")
# # ax[1].set_ylabel(r"MAPE($x^{\mathrm{LT}}$)")
# # Format x-axis ticks with scientific notation
# ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
# ax[1].set_xticklabels([])

# Plot for detection_time
ax[0].plot(df_il_mean.index, df_il_mean["detection_time"]["mean"], label="IL")
ax[0].fill_between(
    df_il_mean.index,
    df_il_mean["detection_time"]["mean"] - df_il_mean["detection_time"]["std"],
    df_il_mean["detection_time"]["mean"] + df_il_mean["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_skf_mean.index, df_skf_mean["detection_time"]["mean"], label="SKF")
ax[0].fill_between(
    df_skf_mean.index,
    df_skf_mean["detection_time"]["mean"] - df_skf_mean["detection_time"]["std"],
    df_skf_mean["detection_time"]["mean"] + df_skf_mean["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_skf_whitenoise_mean.index, df_skf_whitenoise_mean["detection_time"]["mean"], label="Matrix profile")
ax[0].fill_between(
    df_skf_whitenoise_mean.index,
    df_skf_whitenoise_mean["detection_time"]["mean"] - df_skf_whitenoise_mean["detection_time"]["std"],
    df_skf_whitenoise_mean["detection_time"]["mean"] + df_skf_whitenoise_mean["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_prophet_mean.index, df_prophet_mean["detection_time"]["mean"], label="Prophet")
ax[0].fill_between(
    df_prophet_mean.index,
    df_prophet_mean["detection_time"]["mean"] - df_prophet_mean["detection_time"]["std"],
    df_prophet_mean["detection_time"]["mean"] + df_prophet_mean["detection_time"]["std"],
    alpha=0.2,
)
ax[0].set_ylabel(r"$\Delta t\ (\mathrm{yr})$")
# ax[2].set_yticks([0, 52, 104, 156, 208, 260])
ax[0].set_yticks([0, 52, 104, 156])
ax[0].set_yticklabels([0, 1, 2, 3])
ax[0].set_xscale('log')
ax[0].set_ylim(0, 52 * 3.05)
ax[0].set_xticklabels([])
ax[0].legend(ncol=2)

# Plot for detection_rate
ax[1].plot(df_il_mean.index, df_il_mean["detection_rate"]["mean"], label="IL")
ax[1].plot(df_skf_mean.index, df_skf_mean["detection_rate"]["mean"], label="SKF")
ax[1].plot(df_skf_whitenoise_mean.index, df_skf_whitenoise_mean["detection_rate"]["mean"], label="Matrix profile")
ax[1].plot(df_prophet_mean.index, df_prophet_mean["detection_rate"]["mean"], label="Prophet")
# ax[3].set_xlabel("Anomaly Magnitude (unit/year)")
ax[1].set_ylabel(r"$\mathcal{P}_{\mathrm{detect}}$")
# ax[3].set_ylabel(r"$\Pr_{\mathrm{detect}}$")
ax[1].set_ylim(-0.05, 1.05)
ax[1].set_yticks([0, 0.5, 1])
ax[1].set_xscale('log')
# ax[3].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax[1].set_xticklabels([])

# Plot the number of false alarms for prophet
ax[2].plot(df_il_mean.index, df_il_mean["false_alarms_num"]["mean"], label="IL", color = "tab:blue")
ax[2].fill_between(
    df_il_mean.index,
    df_il_mean["false_alarms_num"]["mean"] - df_il_mean["false_alarms_num"]["std"],
    df_il_mean["false_alarms_num"]["mean"] + df_il_mean["false_alarms_num"]["std"],
    alpha=0.2,
    color = "tab:blue"
)
ax[2].plot(df_skf_mean.index, df_skf_mean["false_alarms_num"]["mean"], label="SKF", color = "tab:orange")
ax[2].fill_between(
    df_skf_mean.index,
    df_skf_mean["false_alarms_num"]["mean"] - df_skf_mean["false_alarms_num"]["std"],
    df_skf_mean["false_alarms_num"]["mean"] + df_skf_mean["false_alarms_num"]["std"],
    alpha=0.2,
    color = "tab:orange"
)
ax[2].plot(df_skf_whitenoise_mean.index, df_skf_whitenoise_mean["false_alarms_num"]["mean"], label="Matrix profile", color = "tab:green")
ax[2].fill_between(
    df_skf_whitenoise_mean.index,
    df_skf_whitenoise_mean["false_alarms_num"]["mean"] - df_skf_whitenoise_mean["false_alarms_num"]["std"],
    df_skf_whitenoise_mean["false_alarms_num"]["mean"] + df_skf_whitenoise_mean["false_alarms_num"]["std"],
    alpha=0.2,
    color = "tab:green"
)
ax[2].plot(df_prophet_mean.index, df_prophet_mean["false_alarms_num"]["mean"], label="Prophet", color = "tab:red")
ax[2].fill_between(
    df_prophet_mean.index,
    df_prophet_mean["false_alarms_num"]["mean"] - df_prophet_mean["false_alarms_num"]["std"],
    df_prophet_mean["false_alarms_num"]["mean"] + df_prophet_mean["false_alarms_num"]["std"],
    alpha=0.2,
    color = "tab:red"
)
ax[2].set_xlabel("Anomaly Magnitude (unit/year)")
ax[2].set_ylabel("\# re. alarms")
ax[2].set_xscale('log')

plt.tight_layout()
plt.show()