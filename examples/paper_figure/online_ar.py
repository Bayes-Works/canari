import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari import (
    DataProcess,
    Model,
    plot_states,
)
from canari.component import LocalTrend, Periodic, WhiteNoise, Autoregression
from matplotlib import gridspec
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          'lines.linewidth' : 1,
          }
plt.rcParams.update(params)

# # Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Data pre-processing
all_data = {}
all_data["y"] = df_raw.values

# Split into train and test
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    train_split=0.5,
    validation_split=0.2,
    output_col=output_col,
    standardization=False,
)
train_data, validation_data, _, _ = data_processor.get_splits()



# Components
sigma_v = np.sqrt(1e-6)
local_trend = LocalTrend(mu_states=[5, 0.0], var_states=[1e-1, 1e-6], std_error=0)
periodic = Periodic(period=52, mu_states=[5 * 5, 0], var_states=[1e-12, 1e-12])
noise = WhiteNoise(std_error=sigma_v)


# Case 4: Fully online ar, learn both phi and process error online. phi should converge to ~0.9, W2bar should converge to ~25.
AR_process_error_var_prior = 50
var_W2bar_prior = 100
ar = Autoregression(
    mu_states=[-0.0621, 0.5, 0, 0, 0, AR_process_error_var_prior],
    var_states=[
        6.36e-05,
        0.25,
        0,
        AR_process_error_var_prior,
        1e-6,
        var_W2bar_prior,
    ],
)

# Normal model
model = Model(
    local_trend,
    periodic,
    ar,
    noise,
)

# # #
model.filter(data=train_data)
model.smoother()


# time = data_processor.get_time(split="all")
# get the train time range for plotting
time = data_processor.get_time(split="train")
states_type = "prior"
# Get the mean and std of the hidden states
level_mean = model.states.get_mean(states_type=states_type, states_name="level",
            standardization=False,scale_const_mean=data_processor.scale_const_mean[data_processor.output_col],scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
level_std = model.states.get_std(states_type=states_type, states_name="level",
            standardization=False,scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
trend_mean = model.states.get_mean(states_type=states_type, states_name="trend",
            standardization=False,scale_const_mean=data_processor.scale_const_mean[data_processor.output_col],scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
trend_std = model.states.get_std(states_type=states_type, states_name="trend",
            standardization=False,scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
ar_mean = model.states.get_mean(states_type=states_type, states_name="autoregression",
            standardization=False,scale_const_mean=data_processor.scale_const_mean[data_processor.output_col],scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
ar_std = model.states.get_std(states_type=states_type, states_name="autoregression",
            standardization=False,scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
phi_mean = model.states.get_mean(states_type=states_type, states_name="phi",
            standardization=False,scale_const_mean=data_processor.scale_const_mean[data_processor.output_col],scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
phi_std = model.states.get_std(states_type=states_type, states_name="phi",
            standardization=False,scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
W2bar_mean = model.states.get_mean(states_type=states_type, states_name="W2bar",
            standardization=False,scale_const_mean=data_processor.scale_const_mean[data_processor.output_col],scale_const_std=data_processor.scale_const_std[data_processor.output_col],)
W2bar_std = model.states.get_std(states_type=states_type, states_name="W2bar",
            standardization=False,scale_const_std=data_processor.scale_const_std[data_processor.output_col],)

fig = plt.figure(figsize=(5.3, 2.2), constrained_layout=True)
gs = gridspec.GridSpec(3, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
# ax3 = plt.subplot(gs[3])

print(len(level_mean))
print(len(trend_mean))
print(len(ar_mean))
print(len(time))
print(df_raw["values"].shape)

# ax0: plot data
ax0.plot(time, df_raw["values"].iloc[:len(time)], 'k')
# ax0.plot(time, level_mean, color="tab:blue")
# ax0.fill_between(time, level_mean - level_std, level_mean + level_std, color="tab:blue", alpha=0.2)
ax0.set_xticklabels([])
ax0.set_ylabel("Obs.")
ax0.xaxis.set_major_locator(mdates.YearLocator(2))
legend_handles = [
    Line2D([0], [0], color='k', label='Obs.'),
    Line2D([0], [0], color='tab:blue', label='Estimates'),
    Line2D([0], [0], color='r', linestyle='--', label='True values'),
]
# ax0.legend(handles=legend_handles, loc='upper right', fontsize=7)
ax0.legend(handles=legend_handles, bbox_to_anchor=(0, 1.9), loc='upper left', borderaxespad=0., ncol=3, frameon=False)

# ax1: plot trend
ax1.plot(time, phi_mean, color="tab:blue")
ax1.fill_between(time, phi_mean - phi_std, phi_mean + phi_std, color="tab:blue", alpha=0.2)
ax1.axhline(y=0.9, color="r", linestyle="--", label=r"True values")
ax1.set_xticklabels([])
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.set_ylabel(r"$\phi^{\mathtt{AR}}$")
ax1.set_ylim(0.8, 1.1)

# ax3: plot autoregression
ax2.plot(time, W2bar_mean, color="tab:blue")
ax2.fill_between(time, W2bar_mean - W2bar_std, W2bar_mean + W2bar_std, color="tab:blue", alpha=0.2)
ax2.axhline(y=25, color="r", linestyle="--", label=r"True values")
# ax2.set_xticklabels([])
ax2.set_ylabel(r"$(\sigma^{\mathtt{AR}})^2$")
# Set x-axis to show every 3 years with no grid lines
ax2.xaxis.set_major_locator(mdates.YearLocator(2))

# # ax4: plot anomaly probability
# ax3.plot(time, filter_marginal_abnorm_prob, color="tab:blue")
# ax3.set_ylim(-0.05, 1.05)
# ax3.set_ylabel("$p_{\mathtt{anm}}$")

# # add grid in all subplots
# for ax in [ax0, ax1, ax2]:
#     ax.grid(axis='x')  # vertical lines only
#     # Set x ticks to every 3 years with no grid lines
#     ax.xaxis.set_major_locator(mdates.YearLocator(3))
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gcf().align_ylabels([ax0, ax1, ax2])

# fig, ax = plot_skf_states(
#     data_processor=data_processor,
#     states=states,
#     states_type="prior",
#     states_to_plot=["level", "trend", "lstm", "autoregression"],
#     model_prob=filter_marginal_abnorm_prob,
#     standardization=False,
# )
# fig.suptitle("SKF hidden states", fontsize=10, y=1)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.subplots_adjust(hspace=0.3)
plt.savefig('online_ar.pdf')
plt.show()
