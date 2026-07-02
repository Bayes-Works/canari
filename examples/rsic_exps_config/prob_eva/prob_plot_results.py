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
test_ts_df = pd.read_csv("data/prob_eva_syn_time_series/syn_rsic_simple_ts_gen_lltolt.csv")
test_ts_len = len(np.array(eval(test_ts_df.iloc[0]["values"])).flatten())

# test_ts_df = pd.read_csv("data/prob_eva_syn_time_series/detrend_rsic_simple_ts1_gen_lttoll.csv")
# raw = test_ts_df.iloc[0]["values"].replace("nan", "None")
# arr = np.array(ast.literal_eval(raw), dtype=float)
# test_ts_len = len(arr.flatten())

# Input
first_anm_type = 'll'
second_anm_type = 'll'

print('Results for first anomaly type: ', first_anm_type, ' and second anomaly type: ', second_anm_type)
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

print('######################### RSI #########################')
# false_alarm_rate_rsi, df_rsi_group = _process_detection_df_bl(
#     test_ts_len=test_ts_len,
#     csv_path="saved_results/prob_eva/syn_simple_ts_results_rsi_" + first_anm_type + "to" + second_anm_type + ".csv",
#     evaluate_itv_type = False,
#     plot_detection_map = False,
#     first_anm_type = first_anm_type,
# )
# print("False alarm rate for RSI: ", false_alarm_rate_rsi, "per 10 years")

false_alarm_rate_rsi, df_rsi_group = _process_detection_df(
    test_ts_len=test_ts_len,
    csv_path="saved_results/prob_eva/syn_simple_ts_results_rsi_" + first_anm_type + "to" + second_anm_type +"_test1.csv",
)
print("False alarm rate for RSI: ", false_alarm_rate_rsi, "per 10 years")

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
ax[0].plot(df_rsi_group.index, df_rsi_group["detection_time"]["mean"], label=r"RSI", color="tab:orange")
ax[0].fill_between(
    df_rsi_group.index,
    df_rsi_group["detection_time"]["mean"] - df_rsi_group["detection_time"]["std"],
    df_rsi_group["detection_time"]["mean"] + df_rsi_group["detection_time"]["std"],
    alpha=0.2,
    color="tab:orange",
)
ax[0].plot(df_skf_group.index, df_skf_group["detection_time"]["mean"], label=r"SKF", color="tab:green")
ax[0].fill_between(
    df_skf_group.index,
    df_skf_group["detection_time"]["mean"] - df_skf_group["detection_time"]["std"],
    df_skf_group["detection_time"]["mean"] + df_skf_group["detection_time"]["std"],
    alpha=0.2,
    color="tab:green",
)
ax[0].plot(df_damp_group.index, df_damp_group["detection_time"]["mean"], label=r"DAMP", color="tab:red")
ax[0].fill_between(
    df_damp_group.index,
    df_damp_group["detection_time"]["mean"] - df_damp_group["detection_time"]["std"],
    df_damp_group["detection_time"]["mean"] + df_damp_group["detection_time"]["std"],
    alpha=0.2,
    color="tab:red",
)
ax[0].plot(df_prophet_group.index, df_prophet_group["detection_time"]["mean"], label=r"Prophet", color="tab:purple")
ax[0].fill_between(
    df_prophet_group.index,
    df_prophet_group["detection_time"]["mean"] - df_prophet_group["detection_time"]["std"],
    df_prophet_group["detection_time"]["mean"] + df_prophet_group["detection_time"]["std"],
    alpha=0.2,
    color="tab:purple",
)
############### Dummy values for LSTMED on LT->LL, since LSTMED cannot detect LT anomaly, we set its detection time to be 52*3 (the maximum detection time) and detection rate to be 0, to make it show in the plot. ###############
# # Copy df_lstmed_group from df_prophet_group and fill it with all 0
# df_lstmed_group = copy.deepcopy(df_prophet_group)
# # Refill df_lstmed_group with all 0
# df_lstmed_group["detection_time"]["mean"].loc[:] = 52 * 3
# df_lstmed_group["detection_time"]["std"].loc[:] = 0
# df_lstmed_group["detection_rate"]["mean"].loc[:] = 0
# print(df_lstmed_group)
######################################################################################################################################################
ax[0].plot(df_lstmed_group.index, df_lstmed_group["detection_time"]["mean"], label=r"LSTMED", color="tab:brown")
ax[0].fill_between(
    df_lstmed_group.index,
    df_lstmed_group["detection_time"]["mean"] - df_lstmed_group["detection_time"]["std"],
    df_lstmed_group["detection_time"]["mean"] + df_lstmed_group["detection_time"]["std"],
    alpha=0.2,
    color="tab:brown",
)
ax[0].plot(df_tranad_group.index, df_tranad_group["detection_time"]["mean"], label=r"TranAD", color="tab:pink")
ax[0].fill_between(
    df_tranad_group.index,
    df_tranad_group["detection_time"]["mean"] - df_tranad_group["detection_time"]["std"],
    df_tranad_group["detection_time"]["mean"] + df_tranad_group["detection_time"]["std"],
    alpha=0.2,
    color="tab:pink",
)
ax[0].plot(df_rsic_group.index, df_rsic_group["detection_time"]["mean"], label=r"\textbf{RSIC}", linewidth=2, color="tab:blue")
ax[0].fill_between(
    df_rsic_group.index,
    df_rsic_group["detection_time"]["mean"] - df_rsic_group["detection_time"]["std"],
    df_rsic_group["detection_time"]["mean"] + df_rsic_group["detection_time"]["std"],
    alpha=0.2,
    color="tab:blue",
)
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
#     ncol=4,
#     borderaxespad=0.,
#     frameon=True
# )

# Plot for detection_rate
ax[1].plot(df_rsi_group.index, df_rsi_group["detection_rate"]["mean"], label=r"\textbf{RSI}", color="tab:orange")
ax[1].plot(df_skf_group.index, df_skf_group["detection_rate"]["mean"], label=r"\textbf{SKF}", color="tab:green")
ax[1].plot(df_damp_group.index, df_damp_group["detection_rate"]["mean"], label=r"\textbf{DAMP}", color="tab:red")
ax[1].plot(df_prophet_group.index, df_prophet_group["detection_rate"]["mean"], label=r"\textbf{Prophet}", color="tab:purple")
ax[1].plot(df_lstmed_group.index, df_lstmed_group["detection_rate"]["mean"], label=r"\textbf{LSTMED}", color="tab:brown")
ax[1].plot(df_tranad_group.index, df_tranad_group["detection_rate"]["mean"], label=r"\textbf{TranAD}", color="tab:pink")
ax[1].plot(df_rsic_group.index, df_rsic_group["detection_rate"]["mean"], label=r"\textbf{RSIC}", linewidth=2, color="tab:blue")
ax[1].set_ylabel(r"$\mathcal{P}_{\mathtt{DET}}$")
ax[1].set_ylim(-0.05, 1.05)
ax[1].set_yticks([0, 0.5, 1])
ax[1].set_xscale('log')
ax[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax[1].legend(loc='lower left', fontsize=6, ncol=1)

ax[1].set_xlabel("Anomaly Magnitude (unit/$y$)")

fig.align_ylabels(ax)

# Show first and second anomaly type in the title
plt.suptitle(r"\textbf{P.E.} " + first_anm_type.upper() + r" $\rightarrow$ " + second_anm_type.upper(), fontsize=10)

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.3)
plt.savefig('syn_ts_results_legend.png', dpi=300)
plt.show()


# ################################### Plot the comparison of the detection maps ###################################
# red_points_rsic, gray_points_rsic, blue_points_rsic, orange_points_rsic = _get_color_points(
#     test_ts_len=test_ts_len,
#     csv_path="saved_results/prob_eva/detrend_ts1_results_rsic_" + first_anm_type + "to" + second_anm_type + ".csv")

# red_points_rsi, gray_points_rsi, blue_points_rsi, orange_points_rsi = _get_color_points(
#     test_ts_len=test_ts_len,
#     csv_path="saved_results/prob_eva/syn_simple_ts_results_rsi_" + first_anm_type + "to" + second_anm_type + ".csv")

# red_points_skf, gray_points_skf, blue_points_skf, orange_points_skf = _get_color_points(
#     test_ts_len=test_ts_len,
#     csv_path="saved_results/prob_eva/syn_simple_ts_results_skf_" + first_anm_type + "to" + second_anm_type + ".csv",
#     SKF=True)
# red_points_prophet, gray_points_prophet, blue_points_prophet, orange_points_prophet = _get_color_points(
#     test_ts_len=test_ts_len,
#     csv_path="saved_results/prob_eva/syn_simple_ts_results_prophet_" + first_anm_type + "to" + second_anm_type + ".csv")
# red_points_lstmed, gray_points_lstmed, blue_points_lstmed, orange_points_lstmed = _get_color_points(
#     test_ts_len=test_ts_len,
#     csv_path="saved_results/prob_eva/syn_simple_ts_results_lstmed_" + first_anm_type + "to" + second_anm_type + ".csv")
# red_points_tranad, gray_points_tranad, blue_points_tranad, orange_points_tranad = _get_color_points(
#     test_ts_len=test_ts_len,
#     csv_path="saved_results/prob_eva/syn_simple_ts_results_tranad_" + first_anm_type + "to" + second_anm_type + ".csv")
# red_points_damp, gray_points_damp, blue_points_damp, orange_points_damp = _get_color_points(
#     test_ts_len=test_ts_len,
#     csv_path="saved_results/prob_eva/syn_simple_ts_results_damp_" + first_anm_type + "to" + second_anm_type + ".csv")

# # --- Build the figure with 2 horizontal bar subplots ---
# solo_categories = [
#     ("Anomaly Start", red_points_rsic, "red"),
# ]
# # Add the blue and orange points to the gray_points_rsic, and set the blue and orange ones to 0
# combined_categories_rsic = [
#     ("Detected",        gray_points_rsic,   "gray"),
#     ("LL Intervention", blue_points_rsic,   "tab:blue"),
#     ("LT Intervention", orange_points_rsic, "tab:orange"),
# ]
# gray_points_rsic_collapse = gray_points_rsic + blue_points_rsic + orange_points_rsic
# blue_points_rsic_collapse = np.zeros_like(blue_points_rsic)
# orange_points_rsic_collapse = np.zeros_like(orange_points_rsic)
# combined_categories_rsic_collapse = [
#     ("Detected",        gray_points_rsic_collapse,   "gray"),
#     ("LL Intervention", blue_points_rsic_collapse,   "tab:blue"),
#     ("LT Intervention", orange_points_rsic_collapse, "tab:orange"),
# ]
# combined_categories_rsi = [
#     ("Detected",        gray_points_rsi,   "gray"),
#     ("LL Intervention", blue_points_rsi,   "tab:blue"),
#     ("LT Intervention", orange_points_rsi, "tab:orange"),
# ]
# combined_categories_skf = [
#     ("Detected",        gray_points_skf,   "gray"),
#     ("LL Intervention", blue_points_skf,   "tab:blue"),
#     ("LT Intervention", orange_points_skf, "tab:orange"),
# ]
# combined_categories_prophet = [
#     ("Detected",        gray_points_prophet,   "gray"),
#     ("LL Intervention", blue_points_prophet,   "tab:blue"),
#     ("LT Intervention", orange_points_prophet, "tab:orange"),
# ]
# combined_categories_lstmed = [
#     ("Detected",        gray_points_lstmed,   "gray"),
#     ("LL Intervention", blue_points_lstmed,   "tab:blue"),
#     ("LT Intervention", orange_points_lstmed, "tab:orange"),
# ]
# combined_categories_tranad = [
#     ("Detected",        gray_points_tranad,   "gray"),
#     ("LL Intervention", blue_points_tranad,   "tab:blue"),
#     ("LT Intervention", orange_points_tranad, "tab:orange"),
# ]
# combined_categories_damp = [
#     ("Detected",        gray_points_damp,   "gray"),
#     ("LL Intervention", blue_points_damp,   "tab:blue"),
#     ("LT Intervention", orange_points_damp, "tab:orange"),
# ]
# fig, axes = plt.subplots(
#     8, 1,
#     figsize=(8, 4),
#     sharex=True,
#     gridspec_kw={"hspace": 0.3, "height_ratios": [1, 1, 1, 1, 1, 1, 1, 1]}
# )

# # Determine x range
# all_times = [p[0] for cat in solo_categories + combined_categories_rsic for p in cat[1]]
# x_min, x_max = min(all_times), max(all_times)

# # Resolution for the density bar (pixels along x)
# x_bins = np.linspace(x_min, x_max, 700)
# bin_width = x_bins[1] - x_bins[0]

# # Get normalized constant for the counts for all rsic, rsi and skf
# grey_count_max = 0
# grey_count_max = max(grey_count_max, max(np.histogram([p[0] for p in gray_points_rsic_collapse], bins=x_bins)[0]))
# grey_count_max = max(grey_count_max, max(np.histogram([p[0] for p in gray_points_rsi], bins=x_bins)[0]))
# grey_count_max = max(grey_count_max, max(np.histogram([p[0] for p in gray_points_skf], bins=x_bins)[0]))
# grey_count_max = max(grey_count_max, max(np.histogram([p[0] for p in gray_points_prophet], bins=x_bins)[0]))
# grey_count_max = max(grey_count_max, max(np.histogram([p[0] for p in gray_points_lstmed], bins=x_bins)[0]))
# grey_count_max = max(grey_count_max, max(np.histogram([p[0] for p in gray_points_tranad], bins=x_bins)[0]))
# grey_count_max = max(grey_count_max, max(np.histogram([p[0] for p in gray_points_damp], bins=x_bins)[0]))
# # --- Subplot 0: Red only ---
# ax = axes[0]
# times = np.array([p[0] for p in red_points_rsic])
# counts, edges = np.histogram(times, bins=x_bins)
# norm_counts = counts / counts.max() if counts.max() > 0 else counts
# for val, left in zip(norm_counts, edges[:-1]):
#     if val > 0:
#         ax.axvspan(left, left + bin_width, ymin=0, ymax=1,
#                     color="red", alpha=float(val) * 0.5)
# ax.set_yticks([])
# ax.set_ylabel("Anomaly")
# ax.set_xlim(0, test_ts_len)
# ax.spines[["top", "right", "left"]].set_visible(False)

# # --- Subplot 1: Gray + Blue + Orange combined ---
# ax = axes[1]
# combined_label_parts = []
# for i, (label, points, color) in enumerate(combined_categories_rsic):
#     if not points:
#         continue
#     times = np.array([p[0] for p in points])
#     counts, edges = np.histogram(times, bins=x_bins)
#     norm_counts = counts / grey_count_max
#     # norm_counts = counts / counts.max() if counts.max() > 0 else counts
#     for val, left in zip(norm_counts, edges[:-1]):
#         if val > 0:
#             ax.axvspan(left, left + bin_width, ymin=0, ymax=1,
#                         color=color, alpha=float(val) * 0.5)
#     combined_label_parts.append(label)
# ax.set_yticks([])
# ax.set_ylabel("RSIC")
# ax.set_xlim(0, test_ts_len)
# ax.spines[["top", "right", "left"]].set_visible(False)

# # --- Subplot 2: Gray + Blue + Orange combined ---
# ax = axes[2]
# combined_label_parts = []
# for i, (label, points, color) in enumerate(combined_categories_rsi):
#     if not points:
#         continue
#     times = np.array([p[0] for p in points])
#     counts, edges = np.histogram(times, bins=x_bins)
#     norm_counts = counts / grey_count_max
#     # norm_counts = counts / counts.max() if counts.max() > 0 else counts
#     for val, left in zip(norm_counts, edges[:-1]):
#         if val > 0:
#             ax.axvspan(left, left + bin_width, ymin=0, ymax=1,
#                         color=color, alpha=float(val) * 0.5)
#     combined_label_parts.append(label)
# ax.set_yticks([])
# ax.set_ylabel("RSI")
# ax.set_xlim(0, test_ts_len)
# ax.spines[["top", "right", "left"]].set_visible(False)

# # --- Subplot 3: Gray + Blue + Orange combined ---
# ax = axes[3]
# combined_label_parts = []
# for i, (label, points, color) in enumerate(combined_categories_skf):
#     if not points:
#         continue
#     times = np.array([p[0] for p in points])
#     counts, edges = np.histogram(times, bins=x_bins)
#     norm_counts = counts / grey_count_max
#     # norm_counts = counts / counts.max() if counts.max() > 0 else counts
#     for val, left in zip(norm_counts, edges[:-1]):
#         if val > 0:
#             ax.axvspan(left, left + bin_width, ymin=0, ymax=1,
#                         color=color, alpha=float(val) * 0.5)
#     combined_label_parts.append(label)
# ax.set_yticks([])
# ax.set_ylabel("SKF")
# ax.set_xlim(0, test_ts_len)
# ax.spines[["top", "right", "left"]].set_visible(False)

# # --- Subplot 4: Gray + Blue + Orange combined ---
# ax = axes[4]
# combined_label_parts = []
# for i, (label, points, color) in enumerate(combined_categories_prophet):
#     if not points:
#         continue
#     times = np.array([p[0] for p in points])
#     counts, edges = np.histogram(times, bins=x_bins)
#     norm_counts = counts / grey_count_max
#     # norm_counts = counts / counts.max() if counts.max() > 0 else counts
#     for val, left in zip(norm_counts, edges[:-1]):
#         if val > 0:
#             ax.axvspan(left, left + bin_width, ymin=0, ymax=1,
#                         color=color, alpha=float(val) * 0.5)
#     combined_label_parts.append(label)
# ax.set_yticks([])
# ax.set_ylabel("Prophet")
# ax.set_xlim(0, test_ts_len)
# ax.spines[["top", "right", "left"]].set_visible(False)

# # --- Subplot 5: Gray + Blue + Orange combined ---
# ax = axes[5]
# combined_label_parts = []
# for i, (label, points, color) in enumerate(combined_categories_lstmed):
#     if not points:
#         continue
#     times = np.array([p[0] for p in points])
#     counts, edges = np.histogram(times, bins=x_bins)
#     norm_counts = counts / grey_count_max
#     # norm_counts = counts / counts.max() if counts.max() > 0 else counts
#     for val, left in zip(norm_counts, edges[:-1]):
#         if val > 0:
#             ax.axvspan(left, left + bin_width, ymin=0, ymax=1,
#                         color=color, alpha=float(val) * 0.5)
#     combined_label_parts.append(label)
# ax.set_yticks([])
# ax.set_ylabel("LSTMED")
# ax.set_xlim(0, test_ts_len)
# ax.spines[["top", "right", "left"]].set_visible(False)

# # --- Subplot 6: Gray + Blue + Orange combined ---
# ax = axes[6]
# combined_label_parts = []
# for i, (label, points, color) in enumerate(combined_categories_tranad):
#     if not points:
#         continue
#     times = np.array([p[0] for p in points])
#     counts, edges = np.histogram(times, bins=x_bins)
#     norm_counts = counts / grey_count_max
#     # norm_counts = counts / counts.max() if counts.max() > 0 else counts
#     for val, left in zip(norm_counts, edges[:-1]):
#         if val > 0:
#             ax.axvspan(left, left + bin_width, ymin=0, ymax=1,
#                         color=color, alpha=float(val) * 0.5)
#     combined_label_parts.append(label)
# ax.set_yticks([])
# ax.set_ylabel("TranAD")
# ax.set_xlim(0, test_ts_len)
# ax.spines[["top", "right", "left"]].set_visible(False)

# # --- Subplot 7: Gray + Blue + Orange combined ---
# ax = axes[7]
# combined_label_parts = []
# for i, (label, points, color) in enumerate(combined_categories_damp):
#     if not points:
#         continue
#     times = np.array([p[0] for p in points])
#     counts, edges = np.histogram(times, bins=x_bins)
#     norm_counts = counts / grey_count_max
#     # norm_counts = counts / counts.max() if counts.max() > 0 else counts
#     for val, left in zip(norm_counts, edges[:-1]):
#         if val > 0:
#             ax.axvspan(left, left + bin_width, ymin=0, ymax=1,
#                         color=color, alpha=float(val) * 0.5)
#     combined_label_parts.append(label)
# ax.set_yticks([])
# ax.set_ylabel("DAMP")
# ax.set_xlim(0, test_ts_len)
# ax.spines[["top", "right", "left"]].set_visible(False)

# # axes[-1].set_xlabel("Time Index")
# # fig.suptitle("Detection Map")
# # Show first and second anomaly type in the title
# fig.suptitle(r"\textbf{Detection Map:} " + first_anm_type.upper() + r" $\rightarrow$ " + second_anm_type.upper(), fontsize=10)

# # Fix title + label clipping
# plt.tight_layout()
# plt.subplots_adjust(left=0.06, top=0.88, bottom=0.15)

# plt.show()