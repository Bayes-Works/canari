import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari.data_process import DataProcess
from canari.baseline_component import LocalTrend
from canari.lstm_component import LstmNetwork
from canari.autoregression_component import Autoregression
from canari.model import Model
from canari.data_visualization import (
    plot_data,
)

# # Read data
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]


# Define parameters
output_col = [0]
num_epoch = 200

data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=0.5,
    validation_split=0.2,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
# Define AR model
AR_process_error_var_prior = 1e4
var_W2bar_prior = 1e4
ar = Autoregression(
    mu_states=[0, 0, 0, 0, 0, AR_process_error_var_prior],
    var_states=[1e-06, 0.01, 0, AR_process_error_var_prior, 0, var_W2bar_prior],
)
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=26,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
    ),
    ar,
)

# index_start = 0
# index_end = 52
# y1 = train_data["y"][index_start:index_end].flatten()
# trend, _, seasonality, _ = DataProcess.decompose_data(y1)
# t_plot = data_processor.data.index[index_start:index_end].to_numpy()
# plt.plot(t_plot, trend, color="b")
# plt.plot(t_plot, seasonality, color="orange")
# plt.scatter(t_plot, y1, color="k")
# plt.plot(
#     data_processor.get_time("train"),
#     data_processor.get_data("train", normalization=True),
#     color="r",
# )
# plt.show()
# model.auto_initialize_baseline_states(train_data["y"][0:52])

# Training
for epoch in range(num_epoch):
    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )

    # Unstandardize the predictions
    mu_validation_preds_unorm = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.norm_const_mean[output_col],
        data_processor.norm_const_std[output_col],
    )
    std_validation_preds_unorm = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.norm_const_std[output_col],
    )

    # Calculate the log-likelihood metric
    validation_obs = data_processor.get_data("validation").flatten()
    validation_log_lik = metric.log_likelihood(
        prediction=mu_validation_preds_unorm,
        observation=validation_obs,
        std=std_validation_preds_unorm,
    )

    # Early-stopping
    model.early_stopping(evaluate_metric=-validation_log_lik, mode="min")
    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
        states_optim = copy.copy(
            states
        )  # If we want to plot the states, plot those from optimal epoch
    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

time_step = len(train_data["y"]) + len(validation_data["y"])
model.set_memory(states=states_optim, time_step=time_step)
syn_data = model.generate_synthetic_data(
    data=test_data,
    num_time_series=1,
)

# model.set_memory(states=states_optim, time_step=0)
# syn_data = model.generate_synthetic_data(
#     data=normalized_data,
#     num_time_series=1,
# )

#  Plot
fig, ax = plt.subplots(figsize=(10, 6))
plot_data(
    data_processor=data_processor,
    normalization=True,
    plot_column=output_col,
    validation_label="y",
)
# plot_prediction(
#     data_processor=data_processor,
#     mean_validation_pred=mu_validation_preds,
#     std_validation_pred=std_validation_preds,
#     validation_label=[r"$\mu$", f"$\pm\sigma$"],
# )
# plt.plot(data_processor.get_time("all"), syn_data, color="b")
plt.plot(data_processor.get_time("test"), syn_data, color="b")
plt.legend()
plt.show()
