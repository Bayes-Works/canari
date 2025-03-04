import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import (
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
import copy
from matplotlib import gridspec


# # Read data
data_file = "./data/benchmark_data/test_8_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["displacement_z", "water_level", "temp_min", "temp_max"]

# # Skip resampling data
df = df_raw

# Define parameters
output_col = [0]
num_epoch = 100

data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=0.3,
    validation_split=0.074,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
AR = Autoregression(std_error=np.sqrt(0.00438643), phi=0.41399227, mu_states=[0.01074491], var_states=[6.24768401e-03])

model = Model(
    # LocalTrend(mu_states=[1.70606055e-01, -7.50761283e-04], var_states=[7.91547954e-03, 7.06413244e-07]),
    LocalTrend(mu_states=[-0.65344852, 0.00199401], var_states=[2.00830872e-04, 1.66426896e-08]),
    LstmNetwork(
        look_back_len=22,
        num_features=5,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        load_lstm_net="saved_params/lstm_ts08.pth",
    ),
    AR,
)

model.filter(normalized_data)
model.smoother(normalized_data)
# mu_validation_preds, std_validation_preds = model.forecast(validation_data)

# # Unstandardize the predictions
# mu_validation_preds_unnorm = normalizer.unstandardize(
#     mu_validation_preds,
#     data_processor.norm_const_mean[output_col],
#     data_processor.norm_const_std[output_col],
# )
# std_validation_preds_unnorm = normalizer.unstandardize_std(
#     std_validation_preds,
#     data_processor.norm_const_std[output_col],
# )

# mse = metric.mse(
#     mu_validation_preds_unnorm, data_processor.validation_data[:, output_col].flatten()
# )

# validation_log_lik = metric.log_likelihood(
#     prediction=mu_validation_preds_unnorm,
#     observation=data_processor.validation_data[:, output_col].flatten(),
#     std=std_validation_preds_unnorm,
# )

# # print(f"Optimal epoch       : {model.optimal_epoch}")
# print(f"Validation LL      :{validation_log_lik: 0.4f}")

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
# plot_prediction(
#     data_processor=data_processor,
#     mean_validation_pred=mu_validation_preds,
#     std_validation_pred=std_validation_preds,
#     validation_label=[r"$\mu$", f"$\pm\sigma$"],
#     sub_plot=ax0,
# )
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
