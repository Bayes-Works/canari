# Read CSV file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import ast
import copy

from matplotlib import ticker
from examples.anm_classifier.prob_eva.prob_process_csv_results import _process_detection_df
from examples.anm_classifier.prob_eva.prob_process_csv_results_bl import _process_detection_df_bl
from examples.anm_classifier.prob_eva.prob_process_csv_results_skf import _process_detection_df_skf

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

test_ts_df = pd.read_csv("data/prob_eva_syn_time_series/syn_rsic_simple_ts_gen_lltolt.csv")
test_ts_len = len(np.array(eval(test_ts_df.iloc[0]["values"])).flatten())

# Input
first_anm_type = 'll'
second_anm_type = 'll'

print('######################### RSIC #########################')
false_alarm_rate_rsic, df_rsic_group = _process_detection_df_bl(
    test_ts_len=test_ts_len,
    # csv_path="saved_results/prob_eva/syn_simple_ts_results_rsic_v1_realjoint3_thresholdfix_lltoll.csv",
    csv_path="saved_results/prob_eva/syn_simple_ts_results_rsic_v2_"+first_anm_type+"to"+second_anm_type+"_test1.csv",
    # csv_path="saved_results/prob_eva/detrend_ts1_results_rsic_"+first_anm_type+"to"+second_anm_type+".csv",
    evaluate_itv_type = True,
    plot_detection_map = False,
    first_anm_type = first_anm_type,
)
print("False alarm rate for RSIC: ", false_alarm_rate_rsic, "per 10 years")

# print('######################### RSI #########################')
# # false_alarm_rate_rsi, df_rsi_group = _process_detection_df_bl(
# #     test_ts_len=test_ts_len,
# #     csv_path="saved_results/prob_eva/syn_simple_ts_results_rsi_" + first_anm_type + "to" + second_anm_type + ".csv",
# #     evaluate_itv_type = False,
# #     plot_detection_map = False,
# #     first_anm_type = first_anm_type,
# # )
# # print("False alarm rate for RSI: ", false_alarm_rate_rsi, "per 10 years")

# false_alarm_rate_rsi, df_rsi_group = _process_detection_df(
#     test_ts_len=test_ts_len,
#     csv_path="saved_results/prob_eva/syn_simple_ts_results_rsi_" + first_anm_type + "to" + second_anm_type + ".csv",
# )
# print("False alarm rate for RSI: ", false_alarm_rate_rsi, "per 10 years")

print('######################### SKF #########################')
false_alarm_rate_skf, df_skf_group = _process_detection_df_skf(
    test_ts_len=test_ts_len,
    csv_path="saved_results/prob_eva/syn_simple_ts_results_skf_" + first_anm_type + "to" + second_anm_type + "_test1.csv",
    evaluate_itv_type = False,
    plot_detection_map = False,
    first_anm_type = first_anm_type,
)
print("False alarm rate for SKF: ", false_alarm_rate_skf, "per 10 years")

print('######################### DAMP #########################')
false_alarm_rate_damp, df_damp_group = _process_detection_df_skf(
    test_ts_len=test_ts_len,
    csv_path="saved_results/prob_eva/syn_simple_ts_results_damp_" + first_anm_type + "to" + second_anm_type + "_test1.csv",
    evaluate_itv_type = False,
    plot_detection_map = False,
    first_anm_type = first_anm_type,
)
print("False alarm rate for DAMP: ", false_alarm_rate_damp, "per 10 years")

print('######################### Prophet #########################')
false_alarm_rate_prophet, df_prophet_group = _process_detection_df_skf(
    test_ts_len=test_ts_len,
    csv_path="saved_results/prob_eva/syn_simple_ts_results_prophet_" + first_anm_type + "to" + second_anm_type + "_test1.csv",
    evaluate_itv_type = False,
    plot_detection_map = False,
    first_anm_type = first_anm_type,
)
print("False alarm rate for Prophet: ", false_alarm_rate_prophet, "per 10 years")

print('######################### LSTMED #########################')
false_alarm_rate_lstmed, df_lstmed_group = _process_detection_df_skf(
    test_ts_len=test_ts_len,
    csv_path="saved_results/prob_eva/syn_simple_ts_results_lstmed_" + first_anm_type + "to" + second_anm_type + "_test1.csv",
    evaluate_itv_type = False,
    plot_detection_map = False,
    first_anm_type = first_anm_type,
)
print("False alarm rate for LSTMED: ", false_alarm_rate_lstmed, "per 10 years")

print('######################### TranAD #########################')
false_alarm_rate_tranad, df_tranad_group = _process_detection_df_skf(
    test_ts_len=test_ts_len,
    csv_path="saved_results/prob_eva/syn_simple_ts_results_tranad_" + first_anm_type + "to" + second_anm_type + "_test1.csv",
    evaluate_itv_type = False,
    plot_detection_map = False,
    first_anm_type = first_anm_type,
)
print("False alarm rate for TranAD: ", false_alarm_rate_tranad, "per 10 years")


# Plot the mean and std of df_rsic["mse_LL"], df_rsic["mse_LT"], and df_rsic["detection_time"] for each anomaly magnitude
fig, ax = plt.subplots(2, 1, figsize=(3, 2.5), constrained_layout=True)
# fig, ax = plt.subplots(2, 1, figsize=(10, 2.5), constrained_layout=True)


# Plot for detection_time
# ax[0].plot(df_rsi_group.index, df_rsi_group["detection_time"]["mean"], label=r"\textbf{RSI}")
# ax[0].fill_between(
#     df_rsi_group.index,
#     df_rsi_group["detection_time"]["mean"] - df_rsi_group["detection_time"]["std"],
#     df_rsi_group["detection_time"]["mean"] + df_rsi_group["detection_time"]["std"],
#     alpha=0.2,
# )
# ax[0].plot(df_skf_group.index, df_skf_group["detection_time"]["mean"], label=r"SKF", color="tab:blue", linewidth=0.7)
# ax[0].fill_between(
#     df_skf_group.index,
#     df_skf_group["detection_time"]["mean"] - df_skf_group["detection_time"]["std"],
#     df_skf_group["detection_time"]["mean"] + df_skf_group["detection_time"]["std"],
#     alpha=0.2,
#     color="tab:blue",
# )
# ax[0].plot(df_prophet_group.index, df_prophet_group["detection_time"]["mean"], label=r"Prophet", color="tab:green", linewidth=0.7)
# ax[0].fill_between(
#     df_prophet_group.index,
#     df_prophet_group["detection_time"]["mean"] - df_prophet_group["detection_time"]["std"],
#     df_prophet_group["detection_time"]["mean"] + df_prophet_group["detection_time"]["std"],
#     alpha=0.2,
#     color="tab:green",
# )
# ############## Dummy values for LSTMED on LT->LL, since LSTMED cannot detect LT anomaly, we set its detection time to be 52*3 (the maximum detection time) and detection rate to be 0, to make it show in the plot. ###############
# # Copy df_lstmed_group from df_prophet_group and fill it with all 0
# df_lstmed_group = copy.deepcopy(df_prophet_group)
# # Refill df_lstmed_group with all 0
# df_lstmed_group["detection_time"]["mean"].loc[:] = 52 * 3
# df_lstmed_group["detection_time"]["std"].loc[:] = 0
# df_lstmed_group["detection_rate"]["mean"].loc[:] = 0
# print(df_lstmed_group)
# ax[0].plot(df_lstmed_group.index, df_lstmed_group["detection_time"]["mean"], label=r"LSTMED", color="tab:purple", linewidth=0.7)
# ax[0].fill_between(
#     df_lstmed_group.index,
#     df_lstmed_group["detection_time"]["mean"] - df_lstmed_group["detection_time"]["std"],
#     df_lstmed_group["detection_time"]["mean"] + df_lstmed_group["detection_time"]["std"],
#     alpha=0.2,
#     color="tab:purple",
# )
# ax[0].plot(df_tranad_group.index, df_tranad_group["detection_time"]["mean"], label=r"TranAD", color="tab:brown", linewidth=0.7)
# ax[0].fill_between(
#     df_tranad_group.index,
#     df_tranad_group["detection_time"]["mean"] - df_tranad_group["detection_time"]["std"],
#     df_tranad_group["detection_time"]["mean"] + df_tranad_group["detection_time"]["std"],
#     alpha=0.2,
#     color="tab:brown",
# )
# ax[0].plot(df_damp_group.index, df_damp_group["detection_time"]["mean"], label=r"DAMP", color="tab:orange")
# ax[0].fill_between(
#     df_damp_group.index,
#     df_damp_group["detection_time"]["mean"] - df_damp_group["detection_time"]["std"],
#     df_damp_group["detection_time"]["mean"] + df_damp_group["detection_time"]["std"],
#     alpha=0.2,
#     color="tab:orange",
# )
ax[0].plot(df_rsic_group.index, df_rsic_group["detection_time"]["mean"], label=r"\textbf{RSI}", color="tab:red", linewidth=1.8)
# ax[0].fill_between(
#     df_rsic_group.index,
#     df_rsic_group["detection_time"]["mean"] - df_rsic_group["detection_time"]["std"],
#     df_rsic_group["detection_time"]["mean"] + df_rsic_group["detection_time"]["std"],
#     alpha=0.2,
#     color="tab:red",
# )
ax[0].set_ylabel(r"$\Delta_t(\mathrm{y})$")
ax[0].set_yticks([0, 52, 104, 156])
ax[0].set_yticklabels([0, 1, 2, 3])
ax[0].set_xscale('log')
ax[0].set_ylim(0, 52 * 3.05)
ax[0].set_xticklabels([])
# ax[0].legend(
#     bbox_to_anchor=(0.5, 1.02),
#     loc='lower center',
#     bbox_transform=ax[0].transAxes,
#     ncol=6,
#     borderaxespad=0.,
#     frameon=True
# )

# Plot for detection_rate
# ax[1].plot(df_rsi_group.index, df_rsi_group["detection_rate"]["mean"], label=r"\textbf{RSI}")
# ax[1].plot(df_skf_group.index, df_skf_group["detection_rate"]["mean"], label=r"\textbf{SKF}", color="tab:blue", linewidth=0.7)
# ax[1].plot(df_prophet_group.index, df_prophet_group["detection_rate"]["mean"], label=r"\textbf{Prophet}", color="tab:green", linewidth=0.7)
# ax[1].plot(df_lstmed_group.index, df_lstmed_group["detection_rate"]["mean"], label=r"\textbf{LSTMED}", color="tab:purple", linewidth=0.7)
# ax[1].plot(df_tranad_group.index, df_tranad_group["detection_rate"]["mean"], label=r"\textbf{TranAD}", color="tab:brown", linewidth=0.7)
# ax[1].plot(df_damp_group.index, df_damp_group["detection_rate"]["mean"], label=r"\textbf{DAMP}", color="tab:orange")
ax[1].plot(df_rsic_group.index, df_rsic_group["detection_rate"]["mean"], label=r"\textbf{RSI}", color="tab:red", linewidth=1.8)
ax[1].set_ylabel(r"$\mathcal{P}_{\mathtt{DET}}$")
ax[1].set_ylim(-0.05, 1.05)
ax[1].set_yticks([0, 0.5, 1])
ax[1].set_xscale('log')
ax[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax[1].legend(loc='lower right', fontsize=6)

ax[1].set_xlabel("Anomaly Magnitude (unit/$y$)")

fig.align_ylabels(ax)

# Show first and second anomaly type in the title
plt.suptitle(r"\textbf{P.E.} " + first_anm_type.upper() + r" $\rightarrow$ " + second_anm_type.upper(), fontsize=10)

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.3)
plt.savefig('syn_ts_results_legend.png', dpi=300)

plt.show()