# Read CSV file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import ast

from matplotlib import ticker
from examples.anm_classifier.prob_eva.prob_process_csv_results import _process_detection_df

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
test_ts_len = len(np.array(eval(test_ts_df.iloc[0]["values"])).flatten())

false_alarm_rate_rsic, df_rsic_group = _process_detection_df(
    test_ts_len=test_ts_len,
    csv_path="saved_results/prob_eva/syn_simple_ts_results_rsic_itv_info.csv",
    evaluate_itv_type = True,
)
print("False alarm rate for RSIC: ", false_alarm_rate_rsic, "per 10 years")

false_alarm_rate_rsi, df_rsi_group = _process_detection_df(
    test_ts_len=test_ts_len,
    csv_path="saved_results/prob_eva/syn_simple_ts_results_rsi.csv",
)
print("False alarm rate for RSI: ", false_alarm_rate_rsi, "per 10 years")

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
ax[0].plot(df_rsi_group.index, df_rsi_group["detection_time"]["mean"], label="RSI")
ax[0].fill_between(
    df_rsi_group.index,
    df_rsi_group["detection_time"]["mean"] - df_rsi_group["detection_time"]["std"],
    df_rsi_group["detection_time"]["mean"] + df_rsi_group["detection_time"]["std"],
    alpha=0.2,
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
# ax[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# ax[0].legend(bbox_to_anchor=(0, 2.5), loc='upper left', borderaxespad=0., ncol=4)

# Plot for detection_rate
ax[1].plot(df_rsic_group.index, df_rsic_group["detection_rate"]["mean"], label="RSIC")
ax[1].plot(df_rsi_group.index, df_rsi_group["detection_rate"]["mean"], label="RSI")
# ax[3].set_xlabel("Anomaly Magnitude (unit/year)")
ax[1].set_ylabel(r"$\mathcal{P}_{\mathtt{DET}}$")
# ax[3].set_ylabel(r"$\Pr_{\mathrm{detect}}$")
ax[1].set_ylim(-0.05, 1.05)
ax[1].set_yticks([0, 0.5, 1])
ax[1].set_xscale('log')
# ax[3].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
ax[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax[1].legend(loc='lower right', fontsize=6)
# ax[1].set_xticklabels([])


ax[1].set_xlabel("Anomaly Magnitude (unit/$y$)")

fig.align_ylabels(ax)

# plt.tight_layout(h_pad=0.1, w_pad=0.1)
# plt.subplots_adjust(hspace=0.3)
# plt.savefig('syn_ts_results_legend.png', dpi=300)
plt.show()