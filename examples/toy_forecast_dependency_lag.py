import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import (
    DataProcess,
    Model,
    ModelEnsemble,
    plot_data,
    plot_prediction,
    plot_states,
)
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

# # Read data
data_file = "./data/benchmark_data/test_5_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["y", "water_level", "temp_min", "temp_max"]
# lags = [0, 2, 2, 0]
# df_raw = DataProcess.add_lagged_columns(df_raw, lags)
# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=0.289,
    validation_split=0.0693,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Independent model
model_target = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=10,
        num_features=5,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
    ),
    WhiteNoise(std_error=1e-2),
)
model_target.model_type = "target"
model_target.auto_initialize_baseline_states(train_data["y"][0:24])

# Dependent model
model_covar = Model(
    LstmNetwork(
        look_back_len=19,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
    ),
    WhiteNoise(std_error=1e-2),
)
model_covar.model_type = "covariate"
model_covar.output_col = [1]
model_covar.input_col = [4]
# Ensemble Model
model = ModelEnsemble(model_target, model_covar)


# Training
num_epoch = 50
for epoch in range(num_epoch):
    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )
    model.set_memory(time_step=0)

    # Unstandardize the predictions
    mu_validation_preds = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.scale_const_mean[output_col],
        data_processor.scale_const_std[output_col],
    )

    # Calculate the log-likelihood metric
    validation_obs = data_processor.get_data("validation")
    mse = metric.mse(mu_validation_preds, validation_obs)

    # # Early-stopping
    # model.early_stopping(
    #     evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch, skip_epoch=0
    # )
    # if epoch == model.optimal_epoch:
    #     mu_validation_preds_optim = mu_validation_preds
    #     std_validation_preds_optim = std_validation_preds
    #     states_optim = copy.copy(
    #         states
    #     )  # If we want to plot the states, plot those from optimal epoch

    # if model.stop_training:
    #     break

# print(f"Optimal epoch       : {model.optimal_epoch}")
# print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

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


# plot_states(data_processor=data_processor, states=states_optim, states_type="posterior")
# plt.show()
