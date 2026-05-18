# Read CSV file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import ast

from matplotlib import ticker
from examples.anm_classifier.prob_eva.prob_process_csv_results import _process_detection_df
from examples.anm_classifier.prob_eva.prob_process_csv_results_bl_10ts import _process_detection_df_bl_10ts
from examples.anm_classifier.prob_eva.prob_process_csv_results_skf import _process_detection_df_skf
from examples.anm_classifier.prob_eva.prob_plot_results_get_color_points import _get_color_points

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
# test_ts_df = pd.read_csv("data/prob_eva_syn_time_series/syn_rsic_simple_ts_gen_lltolt.csv")
# test_ts_len = len(np.array(eval(test_ts_df.iloc[0]["values"])).flatten())

test_ts_num = 10
test_ts_len_all = []
for i in range(test_ts_num):
    if i == 2:
        actual_ts_index = 11
    else:
        actual_ts_index = i + 1
    test_ts_df = pd.read_csv("data/prob_eva_syn_time_series/detrend_rsic_simple_ts"+str(actual_ts_index)+"_gen_lttolt.csv")
    raw = test_ts_df.iloc[0]["values"].replace("nan", "None")
    arr = np.array(ast.literal_eval(raw), dtype=float)
    test_ts_len = len(arr.flatten())
    test_ts_len_all.append(test_ts_len)
print("Test time series lengths: ", test_ts_len_all)

# Input
first_anm_type = 'lt'
second_anm_type = 'lt'

print('Results for first anomaly type: ', first_anm_type, ' and second anomaly type: ', second_anm_type)
print('######################### RSIC #########################')
false_alarm_rate_rsic, df_rsic_group = _process_detection_df_bl_10ts(
    all_test_ts_len=test_ts_len_all,
    csv_path_all=[
        "saved_results/prob_eva/detrend_ts1_results_rsic_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts2_results_rsic_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts11_results_rsic_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts4_results_rsic_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts5_results_rsic_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts6_results_rsic_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts7_results_rsic_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts8_results_rsic_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts9_results_rsic_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts10_results_rsic_"+first_anm_type+"to"+second_anm_type+".csv",
    ],
    evaluate_itv_type = True,
    plot_detection_map = False,
    first_anm_type = first_anm_type,
)
print("False alarm rate for RSIC: ", false_alarm_rate_rsic, "per 10 years")

print('######################### RSI #########################')
false_alarm_rate_rsi, df_rsi_group = _process_detection_df_bl_10ts(
    all_test_ts_len=test_ts_len_all,
    csv_path_all=[
        "saved_results/prob_eva/detrend_ts1_results_rsi_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts2_results_rsi_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts11_results_rsi_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts4_results_rsi_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts5_results_rsi_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts6_results_rsi_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts7_results_rsi_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts8_results_rsi_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts9_results_rsi_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts10_results_rsi_"+first_anm_type+"to"+second_anm_type+".csv",
    ],
    evaluate_itv_type = False,
    plot_detection_map = False,
    collapse_consecutive_detections=True,
    first_anm_type = first_anm_type,
)
print("False alarm rate for RSI: ", false_alarm_rate_rsi, "per 10 years")

print('######################### SKF #########################')
false_alarm_rate_skf, df_skf_group = _process_detection_df_bl_10ts(
    all_test_ts_len=test_ts_len_all,
    csv_path_all=[
        "saved_results/prob_eva/detrend_ts1_results_skf_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts2_results_skf_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts11_results_skf_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts4_results_skf_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts5_results_skf_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts6_results_skf_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts7_results_skf_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts8_results_skf_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts9_results_skf_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts10_results_skf_"+first_anm_type+"to"+second_anm_type+".csv",
    ],
    evaluate_itv_type = False,
    plot_detection_map = False,
    collapse_consecutive_detections=True,
    first_anm_type = first_anm_type,
)
print("False alarm rate for SKF: ", false_alarm_rate_skf, "per 10 years")

print('######################### DAMP #########################')
false_alarm_rate_damp, df_damp_group = _process_detection_df_bl_10ts(
    all_test_ts_len=test_ts_len_all,
    csv_path_all=[
        "saved_results/prob_eva/detrend_ts1_results_damp_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts2_results_damp_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts11_results_damp_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts4_results_damp_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts5_results_damp_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts6_results_damp_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts7_results_damp_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts8_results_damp_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts9_results_damp_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts10_results_damp_"+first_anm_type+"to"+second_anm_type+".csv",
    ],
    evaluate_itv_type = False,
    plot_detection_map = False,
    collapse_consecutive_detections=True,
    first_anm_type = first_anm_type,
)
print("False alarm rate for DAMP: ", false_alarm_rate_damp, "per 10 years")

print('######################### Prophet #########################')
false_alarm_rate_prophet, df_prophet_group = _process_detection_df_bl_10ts(
    all_test_ts_len=test_ts_len_all,
    csv_path_all=[
        "saved_results/prob_eva/detrend_ts1_results_prophet_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts2_results_prophet_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts11_results_prophet_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts4_results_prophet_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts5_results_prophet_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts6_results_prophet_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts7_results_prophet_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts8_results_prophet_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts9_results_prophet_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts10_results_prophet_"+first_anm_type+"to"+second_anm_type+".csv",
    ],
    evaluate_itv_type = False,
    plot_detection_map = False,
    collapse_consecutive_detections=True,
    first_anm_type = first_anm_type,
)
print("False alarm rate for Prophet: ", false_alarm_rate_prophet, "per 10 years")

print('######################### LSTMED #########################')
false_alarm_rate_lstmed, df_lstmed_group = _process_detection_df_bl_10ts(
    all_test_ts_len=test_ts_len_all,
    csv_path_all=[
        "saved_results/prob_eva/detrend_ts1_results_lstmed_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts2_results_lstmed_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts11_results_lstmed_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts4_results_lstmed_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts5_results_lstmed_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts6_results_lstmed_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts7_results_lstmed_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts8_results_lstmed_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts9_results_lstmed_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts10_results_lstmed_"+first_anm_type+"to"+second_anm_type+".csv",
    ],
    evaluate_itv_type = False,
    plot_detection_map = False,
    collapse_consecutive_detections=True,
    first_anm_type = first_anm_type,
)
print("False alarm rate for LSTMED: ", false_alarm_rate_lstmed, "per 10 years")

print('######################### TranAD #########################')
false_alarm_rate_tranad, df_tranad_group = _process_detection_df_bl_10ts(
    all_test_ts_len=test_ts_len_all,
    csv_path_all=[
        "saved_results/prob_eva/detrend_ts1_results_tranad_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts2_results_tranad_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts11_results_tranad_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts4_results_tranad_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts5_results_tranad_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts6_results_tranad_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts7_results_tranad_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts8_results_tranad_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts9_results_tranad_"+first_anm_type+"to"+second_anm_type+".csv",
        "saved_results/prob_eva/detrend_ts10_results_tranad_"+first_anm_type+"to"+second_anm_type+".csv",
    ],
    evaluate_itv_type = False,
    plot_detection_map = False,
    collapse_consecutive_detections=True,
    first_anm_type = first_anm_type,
)
print("False alarm rate for TranAD: ", false_alarm_rate_tranad, "per 10 years")

# Plot the mean and std of df_rsic["mse_LL"], df_rsic["mse_LT"], and df_rsic["detection_time"] for each anomaly magnitude
# fig, ax = plt.subplots(2, 1, figsize=(6, 2.5), constrained_layout=True)
fig, ax = plt.subplots(2, 1, figsize=(3, 2.5), constrained_layout=True)

# Plot for detection_time
ax[0].plot(df_rsic_group.index, df_rsic_group["detection_time"]["mean"], label=r"\textbf{RSIC}")
ax[0].fill_between(
    df_rsic_group.index,
    df_rsic_group["detection_time"]["mean"] - df_rsic_group["detection_time"]["std"],
    df_rsic_group["detection_time"]["mean"] + df_rsic_group["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_rsi_group.index, df_rsi_group["detection_time"]["mean"], label=r"\textbf{RSI}")
ax[0].fill_between(
    df_rsi_group.index,
    df_rsi_group["detection_time"]["mean"] - df_rsi_group["detection_time"]["std"],
    df_rsi_group["detection_time"]["mean"] + df_rsi_group["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_skf_group.index, df_skf_group["detection_time"]["mean"], label=r"\textbf{SKF}")
ax[0].fill_between(
    df_skf_group.index,
    df_skf_group["detection_time"]["mean"] - df_skf_group["detection_time"]["std"],
    df_skf_group["detection_time"]["mean"] + df_skf_group["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_damp_group.index, df_damp_group["detection_time"]["mean"], label=r"\textbf{DAMP}")
ax[0].fill_between(
    df_damp_group.index,
    df_damp_group["detection_time"]["mean"] - df_damp_group["detection_time"]["std"],
    df_damp_group["detection_time"]["mean"] + df_damp_group["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_prophet_group.index, df_prophet_group["detection_time"]["mean"], label=r"\textbf{Prophet}")
ax[0].fill_between(
    df_prophet_group.index,
    df_prophet_group["detection_time"]["mean"] - df_prophet_group["detection_time"]["std"],
    df_prophet_group["detection_time"]["mean"] + df_prophet_group["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_lstmed_group.index, df_lstmed_group["detection_time"]["mean"], label=r"\textbf{LSTMED}")
ax[0].fill_between(
    df_lstmed_group.index,
    df_lstmed_group["detection_time"]["mean"] - df_lstmed_group["detection_time"]["std"],
    df_lstmed_group["detection_time"]["mean"] + df_lstmed_group["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_tranad_group.index, df_tranad_group["detection_time"]["mean"], label=r"\textbf{TranAD}")
ax[0].fill_between(
    df_tranad_group.index,
    df_tranad_group["detection_time"]["mean"] - df_tranad_group["detection_time"]["std"],
    df_tranad_group["detection_time"]["mean"] + df_tranad_group["detection_time"]["std"],
    alpha=0.2,
)
ax[0].set_ylabel(r"$\Delta_t(\mathrm{y})$")
ax[0].set_yticks([0, 52, 104, 156])
ax[0].set_yticklabels([0, 1, 2, 3])
# ax[0].set_xscale('log')
ax[0].set_ylim(0, 52 * 3.05)
ax[0].set_xticklabels([])

# Plot for detection_rate
ax[1].plot(df_rsic_group.index, df_rsic_group["detection_rate"]["mean"], label=r"\textbf{RSIC}")
ax[1].plot(df_rsi_group.index, df_rsi_group["detection_rate"]["mean"], label=r"\textbf{RSI}")
ax[1].plot(df_skf_group.index, df_skf_group["detection_rate"]["mean"], label=r"\textbf{SKF}")
ax[1].plot(df_damp_group.index, df_damp_group["detection_rate"]["mean"], label=r"\textbf{DAMP}")
ax[1].plot(df_prophet_group.index, df_prophet_group["detection_rate"]["mean"], label=r"\textbf{Prophet}")
ax[1].plot(df_lstmed_group.index, df_lstmed_group["detection_rate"]["mean"], label=r"\textbf{LSTMED}")
ax[1].plot(df_tranad_group.index, df_tranad_group["detection_rate"]["mean"], label=r"\textbf{TranAD}")
ax[1].set_ylabel(r"$\mathcal{P}_{\mathtt{DET}}$")
ax[1].set_ylim(-0.05, 1.05)
ax[1].set_yticks([0, 0.5, 1])
# ax[1].set_xscale('log')
ax[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax[1].legend(loc='lower right', fontsize=6)

ax[1].set_xlabel("Anomaly Magnitude (unit/$y$)")

fig.align_ylabels(ax)

# Show first and second anomaly type in the title
plt.suptitle(r"\textbf{P.E.} " + first_anm_type.upper() + r" $\rightarrow$ " + second_anm_type.upper(), fontsize=10)

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.3)
# plt.savefig('syn_ts_results_legend.png', dpi=300)

plt.show()