import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend

# Read data
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

# Build data processor
data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day"],
    train_split=0.8,
    validation_split=0.1,
    output_col=output_col,
)

# split data
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
sigma_v = 0.003
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=12,
        num_features=2,
        infer_len=24 * 3,
        num_layer=1,
        num_hidden_unit=40,
        device="cpu",
        manual_seed=1,
        # smoother=False,
    ),
    WhiteNoise(std_error=sigma_v),
)

model.auto_initialize_baseline_states(train_data["y"][0:24])

if model.lstm_net.smooth:
    model.lstm_net.num_samples = model.lstm_net.lstm_infer_len + len(train_data["y"])

# Training
for epoch in range(num_epoch):

    # set white noise decay
    model.white_noise_decay(epoch, white_noise_max_std=5, white_noise_decay_factor=0.9)

    # warm-up for infer_len steps
    if model.lstm_net.smooth:
        model.pretraining_filter(train_data)

    model.filter(train_data)

    # forecast on the validation set
    mu_validation_preds, std_validation_preds, _ = model.forecast(validation_data)

    # Unstandardize the predictions
    mu_validation_preds = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.scale_const_mean[output_col],
        data_processor.scale_const_std[output_col],
    )
    std_validation_preds = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.scale_const_std[output_col],
    )

    # Calculate the log-likelihood metric
    validation_obs = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_validation_preds, validation_obs)

    # Early-stopping
    model.early_stopping(evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch)
    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
        states_optim = copy.copy(
            model.states
        )  # If we want to plot the states, plot those from optimal epoch

    # smooth on train data
    model.smoother()

    model.set_memory(time_step=0)
    model._current_epoch += 1

    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

# set memory and parameters to optimal epoch
model.set_memory(
    time_step=data_processor.test_start - 1,
)

# forecat on the test set
mu_test_preds, std_test_preds, test_states = model.forecast(
    data=test_data,
)

# Unstandardize the predictions
mu_test_preds = normalizer.unstandardize(
    mu_test_preds,
    data_processor.scale_const_mean[output_col],
    data_processor.scale_const_std[output_col],
)
std_test_preds = normalizer.unstandardize_std(
    std_test_preds,
    data_processor.scale_const_std[output_col],
)

# calculate the test metrics
test_obs = data_processor.get_data("test").flatten()
mse = metric.mse(mu_test_preds, test_obs)
log_lik = metric.log_likelihood(mu_test_preds, test_obs, std_test_preds)

print(f"Test MSE            :{mse: 0.4f}")
print(f"Test Log-Lik        :{log_lik: 0.2f}")

# plot the test data
fig, ax = plt.subplots(figsize=(10, 6))
plot_data(
    data_processor=data_processor,
    standardization=False,
    plot_column=output_col,
    validation_label="y",
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds_optim,
    std_validation_pred=std_validation_preds_optim,
    validation_label=[r"$\mu$", r"$\pm\sigma$"],
)
plot_prediction(
    data_processor=data_processor,
    mean_test_pred=mu_test_preds,
    std_test_pred=std_test_preds,
    test_label=[r"$\mu^{\prime}$", r"$\pm\sigma^{\prime}$"],
    color="purple",
)
plt.legend(loc=(0.1, 1.01), ncol=6, fontsize=12)
plt.tight_layout()
plt.show()
