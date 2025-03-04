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
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from matplotlib import gridspec


# # Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Change the trend of data
df_raw.values+=np.arange(len(df_raw.values),dtype="float64").reshape(-1, 1)*0.1

# # Skip resampling data
df = df_raw

# Define parameters
output_col = [0]
num_epoch = 100

data_processor = DataProcess(
    data=df,
    train_split=0.8,
    validation_split=0.2,
    # train_split=1,
    # validation_split=0,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

sigma_v = np.sqrt(1e-6)/(data_processor.norm_const_std[output_col].item()+1e-10)
noise = WhiteNoise(std_error=sigma_v)

# Define AR
AR = Autoregression(std_error=np.sqrt(0.0070628), phi=0.93205819, mu_states=[0.16468303], var_states=[6.05341437e-03])

model = Model(
    # LocalTrend(mu_states=[-0.00902307, 0.0], var_states=[1e-12, 1e-12], std_error=0), # True baseline values without changing the trend
    LocalTrend(mu_states=[-1.19173573, 0.00295942], var_states=[7.53729465e-04, 4.79224907e-09], std_error=0),  # Learned baseline values
    LstmNetwork(
        look_back_len=52,
        num_features=1,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        load_lstm_net='saved_params/lstm_net_learn_with_ar.pth'
    ),
    AR,
)
# model._mu_local_level = -0.00902307
# model.auto_initialize_baseline_states(train_data["y"][0:52*8])


model.filter(train_data)
model.smoother(train_data)
mu_validation_preds, std_validation_preds = model.forecast(validation_data)


# Unstandardize the predictions
mu_validation_preds_unnorm = normalizer.unstandardize(
    mu_validation_preds,
    data_processor.norm_const_mean[output_col],
    data_processor.norm_const_std[output_col],
)
std_validation_preds_unnorm = normalizer.unstandardize_std(
    std_validation_preds,
    data_processor.norm_const_std[output_col],
)

mse = metric.mse(
    mu_validation_preds_unnorm, data_processor.validation_data[:, output_col].flatten()
)

validation_log_lik = metric.log_likelihood(
    prediction=mu_validation_preds_unnorm,
    observation=data_processor.validation_data[:, output_col].flatten(),
    std=std_validation_preds_unnorm,
)


# print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation LL      :{validation_log_lik: 0.4f}")

# # Saved the trained lstm network
# params_path = 'saved_params/lstm_net_learn_with_ar.pth'
# model.lstm_net.load_state_dict(model.early_stop_lstm_param)
# model.lstm_net.save(filename = params_path)

#  Plot
state_type = "prior"
#  Plot states from AR learner
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(6, 1)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])
ax4 = plt.subplot(gs[4])
ax5 = plt.subplot(gs[5])
plot_data(
    data_processor=data_processor,
    normalization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax0,
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds,
    std_validation_pred=std_validation_preds,
    validation_label=[r"$\mu$", f"$\pm\sigma$"],
    sub_plot=ax0,
)
plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type=state_type,
    states_to_plot=['local level'],
    sub_plot=ax0,
)
ax0.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type=state_type,
    states_to_plot=['local trend'],
    sub_plot=ax1,
)
ax1.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type=state_type,
    states_to_plot=['lstm'],
    sub_plot=ax2,
)
ax2.set_xticklabels([])
plot_states(
    data_processor=data_processor,
    states=model.states,
    states_type=state_type,
    states_to_plot=['autoregression'],
    sub_plot=ax3,
)
ax3.set_xticklabels([])
if "phi" in model.states_name:
    plot_states(
        data_processor=data_processor,
        states=model.states,
        states_type=state_type,
        states_to_plot=['phi'],
        sub_plot=ax4,
    )
    ax4.set_xticklabels([])
if "W2bar" in model.states_name:
    plot_states(
        data_processor=data_processor,
        states=model.states,
        states_type=state_type,
        states_to_plot=['W2bar'],
        sub_plot=ax5,
    )
plt.show()
