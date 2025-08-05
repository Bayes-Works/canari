import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import (
    DataProcess,
    Model,
    ModelAssemble,
    plot_data,
    plot_prediction,
    plot_states,
)
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

# # Read data
data_file = "./data/toy_time_series/exp_sine_dependency.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
df_raw.columns = ["exp_sine", "sine"]
lags = [0, 9]
df_raw = DataProcess.add_lagged_columns(df_raw, lags)

data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    train_split=0.8,
    validation_split=0.2,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Independent model
model_target = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=1,
        num_features=11,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=2,
        # model_noise=True,
    ),
    WhiteNoise(std_error=1e-1),
)
model_target.auto_initialize_baseline_states(train_data["y"][0:24])

# Dependent model
model_covar = Model(
    LstmNetwork(
        look_back_len=10,
        num_features=1,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=2,
        # model_noise=True,
    ),
    WhiteNoise(std_error=0.003),
)
model_covar.output_col = [1]

# Assemble models
model = ModelAssemble(target_model=model_target, covariate_model=model_covar)

# Training
num_epoch = 50
for epoch in range(num_epoch):

    (mu_validation_preds, std_validation_preds) = model.lstm_train(
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
    std_validation_preds = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.scale_const_std[output_col],
    )
    # Calculate the log-likelihood metric
    validation_obs = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_validation_preds, validation_obs)

    # Early-stopping
    model_target.early_stopping(
        evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch, skip_epoch=10
    )

    if epoch == model_target.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
        states_optim = copy.copy(model_target.states)

    if model_target.stop_training:
        break

print(f"Optimal epoch target model     : {model_target.optimal_epoch}")
print(f"Validation metric      :{model_target.early_stop_metric: 0.4f}")

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
    mean_validation_pred=mu_validation_preds_optim,
    std_validation_pred=std_validation_preds_optim,
)
plt.legend()
plt.show()
