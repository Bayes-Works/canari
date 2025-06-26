import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

# # Read data
data_file = "./data/toy_time_series/synthetic_simple_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

data_file_time = "./data/toy_time_series/synthetic_simple_autoregression_periodic_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Add synthetic anomaly to data
anm_mag = 0.004
trend = np.linspace(0, 0, num=len(df_raw))
changepoint1_index, changepoint2_index, changepoint3_index = int(len(df_raw) * 0.25), int(len(df_raw) * 0.5), int(len(df_raw) * 0.75)
# Apply slope = anm_mag between changepoint1 and changepoint2
for i in range(changepoint1_index, changepoint2_index):
    trend[i] = anm_mag * (i - changepoint1_index)

# Hold the last value constant between changepoint2 and changepoint3
trend[changepoint2_index:changepoint3_index] = trend[changepoint2_index - 1]

# Apply another increasing trend from changepoint3 to end
for i in range(changepoint3_index, len(df_raw)):
    trend[i] = trend[changepoint3_index - 1] + anm_mag * (i - changepoint3_index)
df_raw = df_raw.add(trend, axis=0)


# Define parameters
output_col = [0]
num_epoch = 200

data_processor = DataProcess(
    data=df_raw,
    train_split=0.4,
    validation_split=0.1,
    output_col=output_col,
)

train_data, validation_data, test_data, standardized_data = data_processor.get_splits()

# Model
sigma_v = 0.19378828180966362
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
model._mu_local_level = 0
# model.auto_initialize_baseline_states(train_data["y"][0:24])

# Training
for epoch in range(num_epoch):
    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )
    model.set_memory(states=states, time_step=0)

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
    model.early_stopping(evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch, skip_epoch=150)
    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
        states_optim = copy.copy(
            states
        )  # If we want to plot the states, plot those from optimal epoch

    if model.stop_training:
        break


# save model
model_dict = model.get_dict()

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

#  Plot
fig, ax = plt.subplots(figsize=(10, 6))
plot_data(
    data_processor=data_processor,
    standardization=False,
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


# # Define pretrained model:
pretrained_model = Model(    
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
pretrained_model.lstm_net.load_state_dict(model.lstm_net.state_dict())

# filter and smoother
mu_obs_preds, std_obs_preds, _ = pretrained_model.filter(standardized_data, train_lstm=False)
pretrained_model.smoother()

state_type = "prior"
# # Plotting results from pre-trained model
fig, ax = plot_states(
    data_processor=data_processor,
    states=pretrained_model.states,
    states_type=state_type,
    standardization=True,
    states_to_plot=[
        "level",
        "trend",
        "lstm",
        "white noise",
    ],
)
plot_data(
    data_processor=data_processor,
    standardization=True,
    plot_column=output_col,
    validation_label="y",
    sub_plot=ax[0],
    plot_test_data=True,
)
time = data_processor.get_time(split="all")
ax[0].plot(time, mu_obs_preds, label="Predicted mean", color="tab:blue")
ax[0].fill_between(
    time,
    mu_obs_preds - std_obs_preds,
    mu_obs_preds + std_obs_preds,
    alpha=0.2,
    label="Predicted std", 
    color="tab:blue",
)
# plot_prediction(
#     data_processor=data_processor,
#     mean_validation_pred=optimal_mu_val_preds,
#     std_validation_pred=optimal_std_val_preds,
#     sub_plot=ax[0],
# )
fig.suptitle("Hidden states estimated by the pre-trained model", fontsize=10, y=1)
plt.show()
