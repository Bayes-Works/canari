import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend
from examples.plot_smooth_states import main


# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
linear_space = np.linspace(0, 2, num=len(df_raw))
df_raw = df_raw.add(linear_space, axis=0)

# remove first N rows
N = 0
df_raw = df_raw.iloc[N:]

data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)

time_series = time_series.iloc[N:]

time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Resampling data
df = df_raw.resample("H").mean()

# Define parameters
output_col = [0]
num_epoch = 50
covariate = [
    "hour_of_day"
]  # Add covariates if needed, e.g., 'hour_of_day', 'day_of_week'

data_processor = DataProcess(
    data=df,
    time_covariates=covariate,
    train_split=0.8,
    validation_split=0.1,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
sigma_v = 0.01
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=12,
        infer_len=24,  # corresponding to one cycle
        num_features=2,
        num_layer=2,
        num_hidden_unit=40,
        device="cpu",
        # manual_seed=1,
    ),
    WhiteNoise(std_error=sigma_v),
)
model.auto_initialize_baseline_states(train_data["y"][0:24])

# Training
if covariate:
    time_covariate_scale_const_mean = data_processor.scale_const_mean[
        data_processor.covariates_col
    ]
    time_covariate_scale_const_std = data_processor.scale_const_std[
        data_processor.covariates_col
    ]
    train_index, val_index, test_index = data_processor.get_split_indices()
    initial_time_covariate = data_processor.data.values[
        train_index[0], data_processor.covariates_col
    ]
    start_time = data_processor.get_time("train")[
        0
    ]  # Ensure start_time is a single timestamp
    infer_len_start = start_time - pd.DateOffset(hours=model.lstm_net.lstm_infer_len)

    # generate look-back covariates
    look_back_cov = pd.DataFrame(
        index=pd.date_range(start=infer_len_start, end=start_time, freq="H")
    )
    look_back_cov["hour_of_day"] = look_back_cov.index.hour
    # look_back_cov['day_of_week'] = look_back_cov.index.dayofweek

    # store in numpy array
    look_back_cov = np.array(look_back_cov).reshape(-1, 1)

    look_back_cov = normalizer.standardize(
        look_back_cov, time_covariate_scale_const_mean, time_covariate_scale_const_std
    )


for epoch in range(num_epoch):
    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
        lookback_covariates=look_back_cov,
    )

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
        model_optim_dict = model.get_dict()

    if model.stop_training:
        break
    else:
        model.set_memory(states=model.states, time_step=0)

    fig, ax = plot_states(
        data_processor=data_processor,
        states=model.states,
        states_type="prior",
    )
    filename = f"saved_results/smoother#{epoch}.png"
    plt.savefig(filename)
    plt.close()

# set memory and parameters to optimal epoch
model.load_dict(model_optim_dict)
model.lstm_net.set_lstm_states(model.lstm_states)
model.set_memory(
    states=states_optim,
    time_step=data_processor.test_start,
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
