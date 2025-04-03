import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari.data_process import DataProcess
from canari.baseline_component import LocalTrend
from canari.lstm_component import LstmNetwork
from canari.white_noise_component import WhiteNoise
from canari.model import Model
from canari.data_visualization import (
    plot_data,
)


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
    time_covariates=["hour_of_day"],
    train_split=0.5,
    validation_split=0.1,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
sigma_v = 0.0032322250444898116
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=19,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
    ),
    WhiteNoise(std_error=sigma_v),
)
model.auto_initialize_baseline_states(train_data["y"][1:24])

# Training
for epoch in range(num_epoch):
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
        states_optim = copy.copy(states)
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
