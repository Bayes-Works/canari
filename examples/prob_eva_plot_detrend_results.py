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

# df_il = pd.read_csv("saved_results/prob_eva/detrended_ts5_results_il.csv")
df_il = pd.read_csv("saved_results/prob_eva/detrended_ts5better_results_il.csv")

# df_skf = pd.read_csv("saved_results/prob_eva/detrended_ts5_results_skf.csv")
# df_skf = pd.read_csv("saved_results/prob_eva/detrended_ts5_results_skf_tuned_threshold.csv")
df_skf = pd.read_csv("saved_results/prob_eva/detrended_ts5_results_skf_better_ssm.csv")
df_skf_whitenoise = pd.read_csv("saved_results/prob_eva/detrended_ts5_whitenoise_skf.csv")

# df_prophet = pd.read_csv("saved_results/prob_eva/toy_simple_results_prophet_online_baseline.csv")

# Multiply the df_il["anomaly_magnitude"] by 52
df_il["anomaly_magnitude"] = np.abs(df_il["anomaly_magnitude"]) * 52
df_skf["anomaly_magnitude"] = np.abs(df_skf["anomaly_magnitude"]) * 52
df_skf_whitenoise["anomaly_magnitude"] = np.abs(df_skf_whitenoise["anomaly_magnitude"]) * 52
# df_prophet["anomaly_magnitude"] = df_prophet["anomaly_magnitude"] * 52

# Compute detection_rate, for each anomaly magnitude, when df_il["detection_time"] == 260, it means that the anomaly is not detected
df_il["detection_rate"] = df_il["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)
df_skf["detection_rate"] = df_skf["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)
df_skf_whitenoise["detection_rate"] = df_skf_whitenoise["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)
# df_prophet["detection_rate"] = df_prophet["detection_time"].apply(
#     lambda x: 0 if x == 260 else 1
# )

# Get anomaly_detected_index from df_prophet["anomaly_detected_index"]
df_il["anomaly_detected_index"] = df_il["anomaly_detected_index"].apply(ast.literal_eval)
df_il["false_alarms_num"] = df_il["anomaly_detected_index"].apply(
    lambda x: len(x) - 1 if len(x) > 1 else 0
)
df_skf["anomaly_detected_index"] = df_skf["anomaly_detected_index"].apply(ast.literal_eval)
df_skf["false_alarms_num"] = df_skf["anomaly_detected_index"].apply(
    lambda x: len(x) - 1 if len(x) > 1 else 0
)
df_skf_whitenoise["anomaly_detected_index"] = df_skf_whitenoise["anomaly_detected_index"].apply(ast.literal_eval)
df_skf_whitenoise["false_alarms_num"] = df_skf_whitenoise["anomaly_detected_index"].apply(
    lambda x: len(x) - 1 if len(x) > 1 else 0
)
# df_prophet["anomaly_detected_index"] = df_prophet["anomaly_detected_index"].apply(ast.literal_eval)
# df_prophet["false_alarms_num"] = df_prophet["anomaly_detected_index"].apply(
#     lambda x: len(x) - 1 if len(x) > 1 else 0
# )

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
df_skf_whitenoise_mean = df_skf_whitenoise.groupby("anomaly_magnitude").agg(
    {
        "mse_LL": ["mean", "std"],
        "mse_LT": ["mean", "std"],
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
        "false_alarms_num": ["mean", "std"],
    }
)

# df_prophet_mean = df_prophet.groupby("anomaly_magnitude").agg(
#     {
#         "mse_LL": ["mean", "std"],
#         "mse_LT": ["mean", "std"],
#         "detection_time": ["mean", "std"],
#         "detection_rate": ["mean", "std"],
#         "false_alarms_num": ["mean", "std"],
#     }
# )

# Plot the mean and std of df_il["mse_LL"], df_il["mse_LT"], and df_il["detection_time"] for each anomaly magnitude
fig, ax = plt.subplots(5, 1, figsize=(10, 6))
# Plot for mse_LL
ax[0].plot(df_il_mean.index, df_il_mean["mse_LL"]["mean"], label="IL")
ax[0].fill_between(
    df_il_mean.index,
    df_il_mean["mse_LL"]["mean"] - df_il_mean["mse_LL"]["std"],
    df_il_mean["mse_LL"]["mean"] + df_il_mean["mse_LL"]["std"],
    alpha=0.2,
)

ax[0].plot(df_skf_mean.index, df_skf_mean["mse_LL"]["mean"], label="SKF")
ax[0].fill_between(
    df_skf_mean.index,
    df_skf_mean["mse_LL"]["mean"] - df_skf_mean["mse_LL"]["std"],
    df_skf_mean["mse_LL"]["mean"] + df_skf_mean["mse_LL"]["std"],
    alpha=0.2,
)

ax[0].plot(df_skf_whitenoise_mean.index, df_skf_whitenoise_mean["mse_LL"]["mean"], label="SKF (whitenoise)")
ax[0].fill_between(
    df_skf_whitenoise_mean.index,
    df_skf_whitenoise_mean["mse_LL"]["mean"] - df_skf_whitenoise_mean["mse_LL"]["std"],
    df_skf_whitenoise_mean["mse_LL"]["mean"] + df_skf_whitenoise_mean["mse_LL"]["std"],
    alpha=0.2,
)
# ax[0].plot(df_prophet_mean.index, df_prophet_mean["mse_LL"]["mean"], label="Prophet")
# ax[0].fill_between(
#     df_prophet_mean.index,
#     df_prophet_mean["mse_LL"]["mean"] - df_prophet_mean["mse_LL"]["std"],
#     df_prophet_mean["mse_LL"]["mean"] + df_prophet_mean["mse_LL"]["std"],
#     alpha=0.2,
# )
ax[0].set_ylabel(r"MSE($x^{\mathrm{LL}}$)")
# ax[0].set_ylabel(r"MAPE($x^{\mathrm{LL}}$)")
ax[0].legend()
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_xticklabels([])

# Plot for mse_LT
ax[1].plot(df_il_mean.index, df_il_mean["mse_LT"]["mean"], label="IL")
ax[1].fill_between(
    df_il_mean.index,
    df_il_mean["mse_LT"]["mean"] - df_il_mean["mse_LT"]["std"],
    df_il_mean["mse_LT"]["mean"] + df_il_mean["mse_LT"]["std"],
    alpha=0.2,
)
ax[1].plot(df_skf_mean.index, df_skf_mean["mse_LT"]["mean"], label="SKF")
ax[1].fill_between(
    df_skf_mean.index,
    df_skf_mean["mse_LT"]["mean"] - df_skf_mean["mse_LT"]["std"],
    df_skf_mean["mse_LT"]["mean"] + df_skf_mean["mse_LT"]["std"],
    alpha=0.2,
)
ax[1].plot(df_skf_whitenoise_mean.index, df_skf_whitenoise_mean["mse_LT"]["mean"], label="SKF (whitenoise)")
ax[1].fill_between(
    df_skf_whitenoise_mean.index,
    df_skf_whitenoise_mean["mse_LT"]["mean"] - df_skf_whitenoise_mean["mse_LT"]["std"],
    df_skf_whitenoise_mean["mse_LT"]["mean"] + df_skf_whitenoise_mean["mse_LT"]["std"],
    alpha=0.2,
)
# ax[1].plot(df_prophet_mean.index, df_prophet_mean["mse_LT"]["mean"], label="Prophet")
# ax[1].fill_between(
#     df_prophet_mean.index,
#     df_prophet_mean["mse_LT"]["mean"] - df_prophet_mean["mse_LT"]["std"],
#     df_prophet_mean["mse_LT"]["mean"] + df_prophet_mean["mse_LT"]["std"],
#     alpha=0.2,
# )
ax[1].set_ylabel(r"MSE($x^{\mathrm{LT}}$)")
# ax[1].set_ylabel(r"MAPE($x^{\mathrm{LT}}$)")
# Format x-axis ticks with scientific notation
ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_xticklabels([])

# Plot for detection_time
ax[2].plot(df_il_mean.index, df_il_mean["detection_time"]["mean"], label="IL")
ax[2].fill_between(
    df_il_mean.index,
    df_il_mean["detection_time"]["mean"] - df_il_mean["detection_time"]["std"],
    df_il_mean["detection_time"]["mean"] + df_il_mean["detection_time"]["std"],
    alpha=0.2,
)
ax[2].plot(df_skf_mean.index, df_skf_mean["detection_time"]["mean"], label="SKF")
ax[2].fill_between(
    df_skf_mean.index,
    df_skf_mean["detection_time"]["mean"] - df_skf_mean["detection_time"]["std"],
    df_skf_mean["detection_time"]["mean"] + df_skf_mean["detection_time"]["std"],
    alpha=0.2,
)
ax[2].plot(df_skf_whitenoise_mean.index, df_skf_whitenoise_mean["detection_time"]["mean"], label="SKF (whitenoise)")
ax[2].fill_between(
    df_skf_whitenoise_mean.index,
    df_skf_whitenoise_mean["detection_time"]["mean"] - df_skf_whitenoise_mean["detection_time"]["std"],
    df_skf_whitenoise_mean["detection_time"]["mean"] + df_skf_whitenoise_mean["detection_time"]["std"],
    alpha=0.2,
)
# ax[2].plot(df_prophet_mean.index, df_prophet_mean["detection_time"]["mean"], label="Prophet")
# ax[2].fill_between(
#     df_prophet_mean.index,
#     df_prophet_mean["detection_time"]["mean"] - df_prophet_mean["detection_time"]["std"],
#     df_prophet_mean["detection_time"]["mean"] + df_prophet_mean["detection_time"]["std"],
#     alpha=0.2,
# )
ax[2].set_ylabel(r"$\Delta t\ (\mathrm{yr})$")
# ax[2].set_yticks([0, 52, 104, 156, 208, 260])
ax[2].set_yticks([0, 52, 104, 156])
ax[2].set_yticklabels([0, 1, 2, 3])
ax[2].set_xscale('log')
ax[2].set_ylim(0, 52 * 3.05)
ax[2].set_xticklabels([])

# Plot for detection_rate
ax[3].plot(df_il_mean.index, df_il_mean["detection_rate"]["mean"], label="IL")
ax[3].plot(df_skf_mean.index, df_skf_mean["detection_rate"]["mean"], label="SKF")
ax[3].plot(df_skf_whitenoise_mean.index, df_skf_whitenoise_mean["detection_rate"]["mean"], label="SKF (whitenoise)")
# ax[3].plot(df_prophet_mean.index, df_prophet_mean["detection_rate"]["mean"], label="Prophet")
# ax[3].set_xlabel("Anomaly Magnitude (unit/year)")
ax[3].set_ylabel(r"$\mathcal{P}_{\mathrm{detect}}$")
# ax[3].set_ylabel(r"$\Pr_{\mathrm{detect}}$")
ax[3].set_ylim(-0.05, 1.05)
ax[3].set_yticks([0, 0.5, 1])
ax[3].set_xscale('log')
# ax[3].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax[3].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax[3].set_xticklabels([])

# Plot the number of false alarms for prophet
ax[4].plot(df_il_mean.index, df_il_mean["false_alarms_num"]["mean"], label="IL", color = "tab:blue")
ax[4].fill_between(
    df_il_mean.index,
    df_il_mean["false_alarms_num"]["mean"] - df_il_mean["false_alarms_num"]["std"],
    df_il_mean["false_alarms_num"]["mean"] + df_il_mean["false_alarms_num"]["std"],
    alpha=0.2,
    color = "tab:blue"
)
ax[4].plot(df_skf_mean.index, df_skf_mean["false_alarms_num"]["mean"], label="SKF", color = "tab:orange")
ax[4].fill_between(
    df_skf_mean.index,
    df_skf_mean["false_alarms_num"]["mean"] - df_skf_mean["false_alarms_num"]["std"],
    df_skf_mean["false_alarms_num"]["mean"] + df_skf_mean["false_alarms_num"]["std"],
    alpha=0.2,
    color = "tab:orange"
)
ax[4].plot(df_skf_whitenoise_mean.index, df_skf_whitenoise_mean["false_alarms_num"]["mean"], label="SKF (whitenoise)", color = "tab:green")
ax[4].fill_between(
    df_skf_whitenoise_mean.index,
    df_skf_whitenoise_mean["false_alarms_num"]["mean"] - df_skf_whitenoise_mean["false_alarms_num"]["std"],
    df_skf_whitenoise_mean["false_alarms_num"]["mean"] + df_skf_whitenoise_mean["false_alarms_num"]["std"],
    alpha=0.2,
    color = "tab:green"
)
# ax[4].plot(df_prophet_mean.index, df_prophet_mean["false_alarms_num"]["mean"], label="Prophet", color = "tab:green")
# ax[4].fill_between(
#     df_prophet_mean.index,
#     df_prophet_mean["false_alarms_num"]["mean"] - df_prophet_mean["false_alarms_num"]["std"],
#     df_prophet_mean["false_alarms_num"]["mean"] + df_prophet_mean["false_alarms_num"]["std"],
#     alpha=0.2,
#     color = "tab:green"
# )
ax[4].set_xlabel("Anomaly Magnitude (unit/year)")
ax[4].set_ylabel("\# re. alarms")
ax[4].set_xscale('log')

plt.tight_layout()
plt.show()