import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LocalTrend, LstmNetwork, WhiteNoise
from datetime import datetime, timedelta

# # Read data
data_file = "./data/traffic.npy"
df = np.load(data_file)
ts = 2
cut_off = 24 * 7 * 5
df = pd.DataFrame(df[:cut_off, ts])
start_date = "2008-01-01 00:00:00"
time_idx = pd.date_range(start=start_date, periods=len(df), freq="H")
df.index = time_idx
df.index.name = "date_time"
df.columns = ["values"]

# Define parameters
output_col = [0]
num_epoch = 50
data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day", "day_of_week"],
    train_split=0.8,
    validation_split=0.2,
    output_col=output_col,
)

train_data, validation_data, test_data, standardized_data = data_processor.get_splits()

# Model
model = Model(
    LstmNetwork(
        look_back_len=24,
        num_features=3,
        num_layer=2,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
        model_noise=True,
    ),
    # WhiteNoise(std_error=0.2),
)

# Training
for epoch in range(num_epoch):
    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )
    model.set_memory(states=states, time_step=0)

    # # # Unstandardize the predictions
    # mu_validation_preds = normalizer.unstandardize(
    #     mu_validation_preds,
    #     data_processor.scale_const_mean[data_processor.output_col],
    #     data_processor.scale_const_std[data_processor.output_col],
    # )

    # std_validation_preds = normalizer.unstandardize_std(
    #     std_validation_preds,
    #     data_processor.scale_const_std[data_processor.output_col],
    # )

    # validation_obs = data_processor.get_data("validation").flatten()
    # validation_log_lik = metric.log_likelihood(
    #     prediction=mu_validation_preds,
    #     observation=validation_obs,
    #     std=std_validation_preds,
    # )

    # # Early-stopping
    # model.early_stopping(
    #     evaluate_metric=-validation_log_lik,
    #     current_epoch=epoch,
    #     max_epoch=num_epoch,
    #     skip_epoch=5,
    # )

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
    model.early_stopping(
        evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch, skip_epoch=5
    )

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
    standardization=False,
    plot_train_data=False,
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

plt.plot(model.early_stop_metric_history)
plt.show()


# plot_states(data_processor=data_processor, states=states_optim, states_type="posterior")
# plt.show()
