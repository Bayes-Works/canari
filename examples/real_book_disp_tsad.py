import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from src import (
    LocalLevel,
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    Periodic,
    Autoregression,
    WhiteNoise,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from src.hsl_detection import hsl_detection
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec
import pickle
import src.common as common

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern',
          }
plt.rcParams.update(params)


# # Read data
data_file = "./data/benchmark_data/test_11_data.csv"
df = pd.read_csv(data_file, skiprows=0, delimiter=",")
date_time = pd.to_datetime(df["Date"])
df = df.drop("Date", axis=1)
df.index = date_time
df.index.name = "date_time"

# Remove df["Nombre Date"]
df = df.drop("Nombre Date", axis=1)
df = df.drop("Niveau Reservoir (m)", axis=1)
df = df.drop("Deplacements cumulatif Y (mm)", axis=1)
df = df.drop("Deplacements cumulatif Z (mm)", axis=1)

# Change column name to value
df.columns = ["values"]
# Resampling data
df = df.resample("W").mean()
# Remove 40% of the last rows in df
df = df.iloc[:-int(len(df) * 0.35)]

# Data pre-processing
output_col = [0]
train_split=0.2516
validation_split=0.08494
data_processor = DataProcess(
    data=df,
    time_covariates=["week_of_year"],
    train_split=train_split,
    validation_split=validation_split,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()


####################################################################
######################### Pretrained model #########################
####################################################################
# Load model_dict from local
with open("saved_params/real_book_disp_model.pkl", "rb") as f:
    model_dict = pickle.load(f)

LSTM = LstmNetwork(
        look_back_len=12,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

print("phi_AR =", model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item())
print("sigma_AR =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()))


pretrained_model = Model(
    # LocalTrend(mu_states=model_dict['early_stop_init_mu_states'][0:2].reshape(-1), var_states=np.diag(model_dict['early_stop_init_var_states'][0:2, 0:2])),
    LocalTrend(mu_states=model_dict['early_stop_init_mu_states'][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    LSTM,
    Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()), 
                   phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(), 
                   mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()], 
                   var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
)

pretrained_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

ltd_error = 1e-5

hsl_tsad_agent = hsl_detection(base_model=pretrained_model, data_processor=data_processor, drift_model_process_error_std=ltd_error)

# Get flexible drift model from the beginning
hsl_tsad_agent_pre = hsl_detection(base_model=pretrained_model.load_dict(pretrained_model.get_dict()), data_processor=data_processor, drift_model_process_error_std=ltd_error)
hsl_tsad_agent_pre.filter(train_data)
hsl_tsad_agent_pre.filter(validation_data)
hsl_tsad_agent.drift_model.var_states = hsl_tsad_agent_pre.drift_model.var_states


mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(train_data, buffer_LTd=True)
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.filter(validation_data, buffer_LTd=True)
# hsl_tsad_agent.estimate_LTd_dist()
hsl_tsad_agent.mu_LTd = 4.680432620441044e-05
hsl_tsad_agent.LTd_pdf = common.gaussian_pdf(mu = hsl_tsad_agent.mu_LTd, std = 7.144144502040045e-05)

hsl_tsad_agent.mean_train = 5.7724024e-05
hsl_tsad_agent.std_train = 0.00033764285
hsl_tsad_agent.mean_target = np.array([-1.3664509e-04, -1.7329467e-02, 1.0712653e+02])
hsl_tsad_agent.std_target = np.array([1.1047281e-02, 1.3709501e+00, 6.2511311e+01])

# hsl_tsad_agent.collect_synthetic_samples(num_time_series=1000, save_to_path= 'data/hsl_tsad_training_samples/itv_learn_samples_real_book_disp.csv')
hsl_tsad_agent.nn_train_with = 'tagiv'
hsl_tsad_agent.learn_intervention(training_samples_path='data/hsl_tsad_training_samples/itv_learn_samples_real_book_disp.csv', 
                                  load_model_path='saved_params/NN_detection_model_real_book_disp.pkl', max_training_epoch=50)
# hsl_tsad_agent.detection_threshold = 0.9
mu_obs_preds, std_obs_preds, mu_ar_preds, std_ar_preds = hsl_tsad_agent.detect(test_data, apply_intervention=True)

# #  Plot
state_type = "prior"
#  Plot states from pretrained model
fig = plt.figure(figsize=(5, 4))
gs = gridspec.GridSpec(5, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
ax4 = plt.subplot(gs[4])

from src.data_visualization import determine_time
time = determine_time(data_processor, len(normalized_data["y"]))
plot_data(
    data_processor=data_processor,
    normalization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax0,
    color='k',
)
plot_states(
    data_processor=data_processor,
    normalization=True,
    # states=pretrained_model.states,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['local level'],
    sub_plot=ax0,
    color='tab:blue',
)
ax0.set_ylabel(r'$x^{\mathrm{LL}}$')
ax0.set_xticklabels([])

plot_states(
    data_processor=data_processor,
    normalization=True,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['local trend'],
    sub_plot=ax1,
    color='tab:blue',
)
ax1.set_ylabel(r'$x^{\mathrm{LT}}$')
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    normalization=True,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['lstm'],
    sub_plot=ax2,
    color='tab:blue',
)
ax2.set_ylabel(r'$x^{\mathrm{LSTM}}$')
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    normalization=True,
    states=hsl_tsad_agent.base_model.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax3,
    color='tab:blue',
)
ax3.set_ylabel(r'$x^{\mathrm{AR}}$')
ax3.set_xticklabels([])

ax4.plot(time, hsl_tsad_agent.p_anm_all)
ax4.set_ylabel(r'$p_{\mathrm{anm}}$')
ax4.set_xlim(ax0.get_xlim())
ax4.set_ylim(-0.05, 1.05)

plt.tight_layout(h_pad=0.5, w_pad=0.1)
plt.savefig('hsl.png', dpi=300)

plt.show()