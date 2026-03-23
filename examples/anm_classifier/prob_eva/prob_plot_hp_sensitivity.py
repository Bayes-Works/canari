# Read CSV file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import ast

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

# Input
first_anm_type = 'lt'
second_anm_type = 'lt'

print('######################### HP1 #########################')
false_alarm_rate_hp1, df_hp1_group = _process_detection_df_bl(
    test_ts_len=test_ts_len,
    csv_path="saved_results/prob_eva/syn_simple_ts_results_rsic_v2_wait1_"+first_anm_type+"to"+second_anm_type+".csv",
    evaluate_itv_type = True,
    plot_detection_map = False,
    first_anm_type = first_anm_type,
)
print("False alarm rate for hp1: ", false_alarm_rate_hp1, "per 10 years")

print('######################### HP2 #########################')
false_alarm_rate_hp2, df_hp2_group = _process_detection_df_bl(
    test_ts_len=test_ts_len,
    csv_path="saved_results/prob_eva/syn_simple_ts_results_rsic_v2_wait3_"+first_anm_type+"to"+second_anm_type+".csv",
    evaluate_itv_type = True,
    plot_detection_map = False,
    first_anm_type = first_anm_type,
)
print("False alarm rate for hp2: ", false_alarm_rate_hp2, "per 10 years")

print('######################### HP3 #########################')
false_alarm_rate_hp3, df_hp3_group = _process_detection_df_bl(
    test_ts_len=test_ts_len,
    csv_path="saved_results/prob_eva/syn_simple_ts_results_rsic_v2_"+first_anm_type+"to"+second_anm_type+".csv",
    evaluate_itv_type = True,
    plot_detection_map = False,
    first_anm_type = first_anm_type,
)
print("False alarm rate for hp3: ", false_alarm_rate_hp3, "per 10 years")

print('######################### HP4 #########################')
false_alarm_rate_hp4, df_hp4_group = _process_detection_df_bl(
    test_ts_len=test_ts_len,
    csv_path="saved_results/prob_eva/syn_simple_ts_results_rsic_v2_wait7_"+first_anm_type+"to"+second_anm_type+".csv",
    evaluate_itv_type = True,
    plot_detection_map = False,
    first_anm_type = first_anm_type,
)
print("False alarm rate for hp4: ", false_alarm_rate_hp4, "per 10 years")


# Plot the mean and std of df_rsic["mse_LL"], df_rsic["mse_LT"], and df_rsic["detection_time"] for each anomaly magnitude
# fig, ax = plt.subplots(2, 1, figsize=(6, 2.5), constrained_layout=True)
fig, ax = plt.subplots(2, 1, figsize=(3, 2.5), constrained_layout=True)

# Plot for detection_time
ax[0].plot(df_hp1_group.index, df_hp1_group["detection_time"]["mean"], label=r"$\hat{t}=1\mathrm{yr}$")
ax[0].fill_between(
    df_hp1_group.index,
    df_hp1_group["detection_time"]["mean"] - df_hp1_group["detection_time"]["std"],
    df_hp1_group["detection_time"]["mean"] + df_hp1_group["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_hp2_group.index, df_hp2_group["detection_time"]["mean"], label=r"$\hat{t}=3\mathrm{yrs}$")
ax[0].fill_between(
    df_hp2_group.index,
    df_hp2_group["detection_time"]["mean"] - df_hp2_group["detection_time"]["std"],
    df_hp2_group["detection_time"]["mean"] + df_hp2_group["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_hp3_group.index, df_hp3_group["detection_time"]["mean"], label=r"$\hat{t}=5\mathrm{yrs}$")
ax[0].fill_between(
    df_hp3_group.index,
    df_hp3_group["detection_time"]["mean"] - df_hp3_group["detection_time"]["std"],
    df_hp3_group["detection_time"]["mean"] + df_hp3_group["detection_time"]["std"],
    alpha=0.2,
)
ax[0].plot(df_hp4_group.index, df_hp4_group["detection_time"]["mean"], label=r"$\hat{t}=7\mathrm{yrs}$")
ax[0].fill_between(
    df_hp4_group.index,
    df_hp4_group["detection_time"]["mean"] - df_hp4_group["detection_time"]["std"],
    df_hp4_group["detection_time"]["mean"] + df_hp4_group["detection_time"]["std"],
    alpha=0.2,
)
ax[0].set_ylabel(r"$\Delta_t(\mathrm{y})$")
ax[0].set_yticks([0, 52, 104, 156])
ax[0].set_yticklabels([0, 1, 2, 3])
ax[0].set_xscale('log')
ax[0].set_ylim(0, 52 * 3.05)
ax[0].set_xticklabels([])

# Plot for detection_rate
ax[1].plot(df_hp1_group.index, df_hp1_group["detection_rate"]["mean"], label=r"$\hat{t}=1\mathrm{yr}$")
ax[1].plot(df_hp2_group.index, df_hp2_group["detection_rate"]["mean"], label=r"$\hat{t}=3\mathrm{yrs}$")
ax[1].plot(df_hp3_group.index, df_hp3_group["detection_rate"]["mean"], label=r"$\hat{t}=5\mathrm{yrs}$")
ax[1].plot(df_hp4_group.index, df_hp4_group["detection_rate"]["mean"], label=r"$\hat{t}=7\mathrm{yrs}$")
ax[1].set_ylabel(r"$\mathcal{P}_{\mathtt{DET}}$")
ax[1].set_ylim(-0.05, 1.05)
ax[1].set_yticks([0, 0.5, 1])
ax[1].set_xscale('log')
ax[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax[1].legend(loc='lower right', fontsize=6, ncol=2)

ax[1].set_xlabel("Anomaly Magnitude (unit/$y$)")

fig.align_ylabels(ax)

# Show first and second anomaly type in the title
plt.suptitle(r"\textbf{P.E.} " + first_anm_type.upper() + r" $\rightarrow$ " + second_anm_type.upper(), fontsize=10)

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.3)
# plt.savefig('syn_ts_results_legend.png', dpi=300)

plt.show()