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

# Get the total length of the test time series
test_ts_df = pd.read_csv("data/prob_eva_syn_time_series/syn_rsic_simple_ts_gen.csv")
# Get one row of the column test_ts_df["values"]
test_ts_len = len(np.array(eval(test_ts_df.iloc[0]["values"])).flatten())

df_rsic = pd.read_csv("saved_results/prob_eva/syn_simple_ts_results_rsic_itv_info.csv")
# df_skf = pd.read_csv("saved_results/prob_eva/syn_complex_regen_ts_results_skf.csv")

# Keep df_rsic["anomaly_magnitude"] for LL anomaly
df_rsic["anomaly_magnitude"] = np.abs(df_rsic["anomaly_magnitude"])
# df_skf["anomaly_magnitude"] = np.abs(df_skf["anomaly_magnitude"]) * 52

df_rsic["anomaly_detected_index"] = df_rsic["anomaly_detected_index"].apply(ast.literal_eval)
# df_skf["anomaly_detected_index"] = df_skf["anomaly_detected_index"].apply(ast.literal_eval)

print('total detection points in RSIC: ', df_rsic['anomaly_detected_index'].apply(len).sum())

# Process df_rsic to get the detection info of the second anomaly only
df_rsic["detection_index_after_anm1"] = pd.Series(dtype=object)
for index, row in df_rsic.iterrows():
    anm1_start_index = row["anomaly_start_index1"]
    anm2_start_index = row["anomaly_start_index2"]

    # Determine if any detection between anm1_start_index and anm2_start_index
    first_anm_detected = False
    detected_indices = row["anomaly_detected_index"]
    if any((detected_index >= anm1_start_index) and (detected_index < anm2_start_index) for detected_index in detected_indices):
        first_anm_detected = True
        first_anm_detect_index = min([detected_index for detected_index in detected_indices if (detected_index >= anm1_start_index) and (detected_index < anm2_start_index)])
        df_rsic.at[index, "first_anm_detect_index"] = first_anm_detect_index
    else:
        df_rsic.at[index, "first_anm_detect_index"] = None
    
    if first_anm_detected:
        # Remove all the detected indices before first_anm_detect_index and including first_anm_detect_index
        detected_indices_after_anm1 = [detected_index for detected_index in detected_indices if detected_index > first_anm_detect_index]
        df_rsic.at[index, "detection_index_after_anm1"] = detected_indices_after_anm1
        # Get the first detection time after the second anomaly
        detection_time = 52 * 3 + 1  # Default value if not detected within 3 years
        for detected_index in detected_indices_after_anm1:
            if detected_index >= anm2_start_index:
                detection_time = detected_index - anm2_start_index
                break
        df_rsic.at[index, "detection_time"] = detection_time
    else:
        df_rsic.at[index, "detection_index_after_anm1"] = None
        df_rsic.at[index, "detection_time"] = None

# Remove the rows where detection_time is None
df_rsic = df_rsic[df_rsic["detection_time"].notnull()]
df_rsic["detection_time"] = df_rsic["detection_time"].astype(int)


# Compute detection_rate, for each anomaly magnitude, when df_rsic["detection_time"] == 260, it means that the anomaly is not detected
df_rsic["detection_rate"] = df_rsic["detection_time"].apply(
    lambda x: 0 if x >= 52 * 3 else 1
)
# df_skf["detection_rate"] = df_skf["detection_time"].apply(
#     lambda x: 0 if x >= 52 * 3 else 1
# )

# Compute the false alarm rate for each method
all_time_after_anm1 = (test_ts_len - df_rsic["first_anm_detect_index"] - 1).sum()
false_alarm_rsic = np.sum(df_rsic["detection_index_after_anm1"].apply(lambda x: len(x))) - df_rsic["detection_rate"].sum()
false_alarm_rate_rsic = round(false_alarm_rsic * 10 / (all_time_after_anm1/52), 2)
print("False alarm rate for RSIC: ", false_alarm_rate_rsic, "per 10 years")

# false_alarms_skf = 0
# neg_detect_indices = []
# for i in range(df_skf.shape[0]):
#     if df_skf.iloc[i]["detection_time"] < 0:
#         false_alarms_skf += len(df_skf.iloc[i]["anomaly_detected_index"])
#         neg_detect_indices.append(i)
# df_skf = df_skf.drop(index=neg_detect_indices)
# false_alarm_rate_skf = false_alarms_skf * 10 / (sum_anm_start_index/52)
# print("False alarm rate for SKF: ", false_alarm_rate_skf, "per 10 years")

# Get anomaly_detected_index from df_prophet["anomaly_detected_index"]
df_rsic["alarms_num"] = df_rsic["anomaly_detected_index"].apply(
    lambda x: len(x) if len(x) > 0 else 0
)
# df_skf["alarms_num"] = df_skf["anomaly_detected_index"].apply(
#     lambda x: len(x) if len(x) > 0 else 0
# )


# For the same anomaly magnitude, compute the mean and variance of df_rsic["mse_LL"], df_rsic["mse_LT"], and df_rsic["detection_time"], stored them in a new dataframe
df_rsic_group = df_rsic.groupby("anomaly_magnitude").agg(
    {
        "detection_time": ["mean", "std"],
        "detection_rate": ["mean", "std"],
    }
)
# df_skf_mean = df_skf.groupby("anomaly_magnitude").agg(
#     {
#         "detection_time": ["mean", "std"],
#         "detection_rate": ["mean", "std"],
#     }
# )

# Plot the mean and std of df_rsic["mse_LL"], df_rsic["mse_LT"], and df_rsic["detection_time"] for each anomaly magnitude
# fig, ax = plt.subplots(2, 1, figsize=(6, 2.5), constrained_layout=True)
fig, ax = plt.subplots(2, 1, figsize=(3, 2.5), constrained_layout=True)


# Plot for detection_time
ax[0].plot(df_rsic_group.index, df_rsic_group["detection_time"]["mean"], label=r"\textbf{RSI}")
ax[0].fill_between(
    df_rsic_group.index,
    df_rsic_group["detection_time"]["mean"] - df_rsic_group["detection_time"]["std"],
    df_rsic_group["detection_time"]["mean"] + df_rsic_group["detection_time"]["std"],
    alpha=0.2,
)
# ax[0].plot(df_skf_mean.index, df_skf_mean["detection_time"]["mean"], label="SKF")
# ax[0].fill_between(
#     df_skf_mean.index,
#     df_skf_mean["detection_time"]["mean"] - df_skf_mean["detection_time"]["std"],
#     df_skf_mean["detection_time"]["mean"] + df_skf_mean["detection_time"]["std"],
#     alpha=0.2,
# )
ax[0].set_ylabel(r"$\Delta_t(\mathrm{y})$")
# ax[2].set_yticks([0, 52, 104, 156, 208, 260])
ax[0].set_yticks([0, 52, 104, 156])
ax[0].set_yticklabels([0, 1, 2, 3])
ax[0].set_xscale('log')
ax[0].set_ylim(0, 52 * 3.05)
ax[0].set_xticklabels([])
# ax[0].legend(ncol=2)
# Show the legend outside the plot
# ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# ax[0].legend(bbox_to_anchor=(0, 2.5), loc='upper left', borderaxespad=0., ncol=4)

# Plot for detection_rate
ax[1].plot(df_rsic_group.index, df_rsic_group["detection_rate"]["mean"], label="IL")
# ax[1].plot(df_skf_mean.index, df_skf_mean["detection_rate"]["mean"], label="SKF")
# ax[3].set_xlabel("Anomaly Magnitude (unit/year)")
ax[1].set_ylabel(r"$\mathcal{P}_{\mathtt{DET}}$")
# ax[3].set_ylabel(r"$\Pr_{\mathrm{detect}}$")
ax[1].set_ylim(-0.05, 1.05)
ax[1].set_yticks([0, 0.5, 1])
ax[1].set_xscale('log')
# ax[3].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax[1].set_xticklabels([])


ax[1].set_xlabel("Anomaly Magnitude (unit/$y$)")

fig.align_ylabels(ax)

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.3)
# plt.savefig('syn_ts_results_legend.png', dpi=300)
plt.show()