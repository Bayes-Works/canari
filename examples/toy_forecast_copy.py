import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Param
look_back_len = 23
start_offset = 12
num_cycle = 5

# Resampling data
df = df_raw.resample("H").mean()
df = df.iloc[start_offset:]

# Define parameters
output_col = [0]
num_epoch = 50

data_processor = DataProcess(
    data=df,
    train_split=0.8,
    validation_split=0.2,
    time_covariates=["hour_of_day"],
    output_col=output_col,
)

train_data, validation_data, test_data, standardized_data = data_processor.get_splits()

# Model
sigma_v = 3e-2
model = Model(
    LstmNetwork(
        look_back_len=look_back_len,
        num_features=2,
        infer_len=24 * num_cycle,
        num_layer=1,
        num_hidden_unit=40,
        device="cpu",
        manual_seed=1,
    ),
    WhiteNoise(std_error=sigma_v),
)

# Training
for epoch in range(num_epoch):
    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
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
            states
        )  # If we want to plot the states, plot those from optimal epoch

    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

# Get first state and observation
prior_states_mu = states_optim.get_mean("lstm", "prior", True)
first_state = prior_states_mu[0]
first_observation = train_data["y"][0]

#  Plot
# fig, ax = plt.subplots(figsize=(10, 6))
# plot_data(
#     data_processor=data_processor,
#     standardization=False,
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
mu_infer = model.lstm_output_history.mu
real_data = data_processor.get_data(split="all", standardization=True).flatten()
obs_pretrain = real_data[: 24 * num_cycle]
mu_lstm = states_optim.get_mean("lstm", "prior", True)

t = np.arange(len(obs_pretrain))
t1 = np.arange(len(mu_infer)) + len(obs_pretrain) - len(mu_infer)
t2 = np.arange(len(real_data)) + len(obs_pretrain)

plt.plot(t, obs_pretrain, color="green")
plt.plot(t1, mu_infer, color="magenta")
plt.scatter(t2, real_data, color="r")
plt.scatter(t2, mu_lstm, color="b")
plt.show()
