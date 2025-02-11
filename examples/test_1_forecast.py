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


# # Read data
# Read data
data_file = "./data/benchmark_data/test_1_data.csv"
df = pd.read_csv(data_file, skiprows=0, delimiter=",")
date_time = pd.to_datetime(df["timestamp"])
df = df.drop("timestamp", axis=1)
df.index = date_time
df.index.name = "date_time"
# Data pre-processing
output_col = [0]
num_epoch = 50
data_processor = DataProcess(
    data=df,
    # time_covariates=["week_of_year"],
    train_start="2011-02-06 00:00:00",
    train_end="2014-02-02 00:00:00",
    validation_start="2014-02-09 00:00:00",
    validation_end="2015-02-01 00:00:00",
    test_start="2015-02-08 00:00:00",
    output_col=output_col,
)
train_data, validation_data, _, all_data = data_processor.get_splits()

# Model
sigma_v = 3e-1
model = Model(
    LocalTrend(var_states=[1e-1, 1e-1]),
    LstmNetwork(
        look_back_len=52,
        num_features=1,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
    ),
    WhiteNoise(std_error=sigma_v),
)
model.auto_initialize_baseline_states(train_data["y"][0:52])

# Training
scheduled_sigma_v = 1
noise_index = model.states_name.index("white noise")
for epoch in range(50):
    if epoch < 5:
        model.process_noise_matrix[noise_index, noise_index] = 9
    else:
        model.process_noise_matrix[noise_index, noise_index] = sigma_v**2

    # Decaying observation's variance
    # scheduled_sigma_v = exponential_scheduler(
    #     curr_v=scheduled_sigma_v, min_v=sigma_v, decaying_factor=0.7, curr_iter=epoch
    # )
    # noise_index = model.states_name.index("white noise")
    # model.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

    (mu_validation_preds, std_validation_preds, smoother_states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )

    # Unstandardize the predictions
    mu_validation_preds_unnorm = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.norm_const_mean[output_col],
        data_processor.norm_const_std[output_col],
    )
    std_validation_preds_unorm = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.norm_const_std[output_col],
    )

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data(
        data_processor=data_processor,
        normalization=True,
        plot_test_data=False,
        plot_column=output_col,
        validation_label="y",
    )
    plot_prediction(
        data_processor=data_processor,
        mean_validation_pred=mu_validation_preds,
        std_validation_pred=std_validation_preds,
        validation_label=[r"$\mu$", f"$\pm\sigma$"],
    )
    plot_states(
        data_processor=data_processor,
        states=smoother_states,
        states_to_plot=["local level"],
        sub_plot=ax,
    )
    plt.legend()
    plt.show()

    # Calculate the log-likelihood metric
    mse = metric.mse(
        mu_validation_preds_unnorm,
        data_processor.validation_data[:, output_col].flatten(),
    )

    # Early-stopping
    model.early_stopping(evaluate_metric=mse, mode="min")
    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

# #  Plot
# fig, ax = plt.subplots(figsize=(10, 6))
# plot_data(
#     data_processor=data_processor,
#     normalization=False,
#     plot_column=output_col,
#     validation_label="y",
# )
# plot_prediction(
#     data_processor=data_processor,
#     mean_validation_pred=mu_validation_preds,
#     std_validation_pred=std_validation_preds,
#     validation_label=[r"$\mu$", f"$\pm\sigma$"],
# )
# plt.legend()
# plt.show()
