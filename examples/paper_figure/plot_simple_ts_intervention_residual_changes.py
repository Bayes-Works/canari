import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari.component import LocalTrend, LstmNetwork, Autoregression
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_states,
    common,
)
from src.hsl_detection import hsl_detection
import pytagi.metric as metric
from matplotlib import gridspec
import pickle
from pytagi import Normalizer
import copy
import matplotlib.dates as mdates

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


# Function to compute autocorrelation function
def acf(x, max_lag):
    """
    Compute the autocorrelation function (ACF) for a 1D array x
    up to lag max_lag.
    
    Returns:
        acf_vals: array of length max_lag+1, where acf_vals[k] = ACF at lag k.
    """
    x = np.asarray(x)
    x = x - np.mean(x)
    n = len(x)

    acf_vals = np.empty(max_lag + 1)

    # variance of series
    denom = np.sum(x * x)

    for lag in range(max_lag + 1):
        num = np.sum(x[:n-lag] * x[lag:])
        acf_vals[lag] = num / denom

    return acf_vals

# Read csv data
results_file = 'saved_results/intervention_ar_changes/no_intervention.csv'
results_df = pd.read_csv(results_file, index_col=0)
time = pd.to_datetime(results_df['time'])
obs = results_df['obs'].values
level_pred_mu = results_df['level_pred_mu'].values
trend_pred_mu = results_df['trend_pred_mu'].values
ar_pred_mu = results_df['ar_pred_mu'].values
level_pred_std = results_df['level_pred_std'].values
trend_pred_std = results_df['trend_pred_std'].values
ar_pred_std = results_df['ar_pred_std'].values
detection_time = results_df.index[results_df['detection_time'] == 1].tolist()
anm_start_time = results_df.index[results_df['anomaly_start_time'] == 1].tolist()

results_file2 = 'saved_results/intervention_ar_changes/intervention.csv'
results_df2 = pd.read_csv(results_file2, index_col=0)
level_pred_mu2 = results_df2['level_pred_mu'].values
trend_pred_mu2 = results_df2['trend_pred_mu'].values
ar_pred_mu2 = results_df2['ar_pred_mu'].values
level_pred_std2 = results_df2['level_pred_std'].values
trend_pred_std2 = results_df2['trend_pred_std'].values
ar_pred_std2 = results_df2['ar_pred_std'].values

ar_after_anm = ar_pred_mu[anm_start_time[0]:]
ar2_after_anm = ar_pred_mu2[anm_start_time[0]:]

acf_ar = acf(ar_after_anm, max_lag=len(ar_after_anm))
acf_ar2 = acf(ar2_after_anm, max_lag=len(ar2_after_anm))

# #  Plot
#  Plot states from pretrained model
# fig = plt.figure(figsize=(6, 5), constrained_layout=True)
# gs = gridspec.GridSpec(3, 1)
fig = plt.figure(figsize=(5, 1.9), constrained_layout=True)
gs = gridspec.GridSpec(3, 2)

ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[2, 0])
ax3 = fig.add_subplot(gs[:, 1])

# Ax0: plot observations and level state
ax0.plot(time, obs, label='obs', color='tab:blue')
ax0.plot(time, level_pred_mu2, label='intervene', color='tab:green')
ax0.fill_between(time, level_pred_mu2 - level_pred_std2, level_pred_mu2 + level_pred_std2, color='tab:green', alpha=0.3)
ax0.axvline(x=time[anm_start_time[0]], color='red', linestyle='--', label='anm start')
ax0.plot(time, level_pred_mu, label='do nothing', color='tab:orange')
ax0.fill_between(time, level_pred_mu - level_pred_std, level_pred_mu + level_pred_std, color='tab:orange', alpha=0.3)
ax0.axvline(x=time[detection_time[0]], color='green', linestyle='--', label='detection')
# ax0.legend(bbox_to_anchor=(0, 3.5), loc='upper left', borderaxespad=0., ncol=3)
ax0.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax0.set_xticklabels([])
ax0.set_ylabel('$x^{\mathtt{LL}}$')


# Ax1: plot trend state
ax1.plot(time, trend_pred_mu, label='Trend state', color='tab:orange')
ax1.fill_between(time, trend_pred_mu - trend_pred_std, trend_pred_mu + trend_pred_std, color='tab:orange', alpha=0.3)
ax1.plot(time, trend_pred_mu2, label='Trend state (with intervention)', color='tab:green')
ax1.fill_between(time, trend_pred_mu2 - trend_pred_std2, trend_pred_mu2 + trend_pred_std2, color='tab:green', alpha=0.3)
ax1.axvline(x=time[anm_start_time[0]], color='red', linestyle='--', label='Anomaly start')
ax1.axvline(x=time[detection_time[0]], color='green', linestyle='--', label='Detection time')
ax1.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax1.set_xticklabels([])
ax1.set_ylabel('$x^{\mathtt{LT}}$')

# Ax2: plot AR state
ax2.plot(time, ar_pred_mu, label='AR state', color='tab:orange')
ax2.fill_between(time, ar_pred_mu - ar_pred_std, ar_pred_mu + ar_pred_std, color='tab:orange', alpha=0.3)
ax2.plot(time, ar_pred_mu2, label='AR state (with intervention)', color='tab:green')
ax2.fill_between(time, ar_pred_mu2 - ar_pred_std2, ar_pred_mu2 + ar_pred_std2, color='tab:green', alpha=0.3)
ax2.axvline(x=time[anm_start_time[0]], color='red', linestyle='--', label='Anomaly start')
ax2.axvline(x=time[detection_time[0]], color='green', linestyle='--', label='Detection time')
ax2.set_ylabel('$x^{\mathtt{AR}}$')
ax2.set_xlabel('Time')
ax2.xaxis.set_major_locator(mdates.YearLocator(base=5))

# Ax3: plot ACF of AR state after anomaly start
lags = np.arange(len(acf_ar))
markerline1, stemlines1, baseline1 = ax3.stem(lags, acf_ar, linefmt='tab:orange', markerfmt='tab:orange', basefmt=" ", label='No intervention')
markerline2, stemlines2, baseline2 = ax3.stem(lags, acf_ar2, linefmt='tab:green', markerfmt='tab:green', basefmt=" ", label='With intervention')
markerline1.set_markersize(1)
markerline2.set_markersize(1)
ax3.set_xlabel('Lag')
ax3.set_ylabel('ACF')
# ax3.legend()

fig.align_ylabels([ax0, ax1, ax2])
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.3)
plt.savefig('ar_changes.png', dpi=300)

plt.show()