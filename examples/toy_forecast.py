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
)
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
import copy


# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
linear_space = np.linspace(0, 2, num=len(df_raw))
df_raw = df_raw.add(linear_space, axis=0)

data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Resampling data
df = df_raw.resample("H").mean()

# Define parameters
output_col = [0]
num_epoch = 50

data_processor = DataProcess(
    data=df,
    train_split=0.8,
    validation_split=0.2,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
sigma_v = 0.0032322250444898116
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=19,
        num_features=1,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
    ),
    WhiteNoise(std_error=sigma_v),
)
# model.auto_initialize_baseline_states(train_data["y"][1:24])
index_start = 0
index_end = 52
y1 = train_data["y"][index_start:index_end].flatten()
decomposer = TimeSeriesDecomposition(y1)
trend, seasonality, residual, slope = decomposer.decompose()
t_plot = data_processor.data.index[index_start:index_end].to_numpy()
plt.plot(t_plot, trend, color="b")
plt.plot(t_plot, seasonality, color="k")
plt.plot(
    data_processor.get_time("train"),
    data_processor.get_data("train", normalization=True),
    color="r",
)
plt.show()

# Training
scheduled_sigma_v = 1
for epoch in range(num_epoch):
    # Decaying observation's variance
    scheduled_sigma_v = exponential_scheduler(
        curr_v=scheduled_sigma_v, min_v=sigma_v, decaying_factor=0.99, curr_iter=epoch
    )
    noise_index = model.states_name.index("white noise")
    model.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )

    # Unstandardize the predictions
    mu_validation_preds = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.norm_const_mean[output_col],
        data_processor.norm_const_std[output_col],
    )
    std_validation_preds = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.norm_const_std[output_col],
    )

    # Calculate the log-likelihood metric
    validation_obs = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_validation_preds, validation_obs)

    # Early-stopping
    model.early_stopping(evaluate_metric=mse, mode="min")
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

#  Plot
fig, ax = plt.subplots(figsize=(10, 6))
plot_data(
    data_processor=data_processor,
    normalization=False,
    plot_column=output_col,
    validation_label="y",
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds,
    std_validation_pred=std_validation_preds,
    validation_label=[r"$\mu$", f"$\pm\sigma$"],
)
plt.legend()
plt.show()
