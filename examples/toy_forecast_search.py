import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import (
    LocalTrend,
    LstmNetwork,
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

data_processor = DataProcess(
    data=df,
    train_split=0.8,
    validation_split=0.2,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()


# Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    sigma_v = trial.suggest_loguniform("sigma_v", 1e-3, 1e-1)
    look_back_len = trial.suggest_int("look_back_len", 12, 48)

    # Define model with suggested parameters
    model = Model(
        LocalTrend(),
        LstmNetwork(
            look_back_len=look_back_len,
            num_features=1,
            num_layer=1,
            num_hidden_unit=50,
            device="cpu",
        ),
        Autoregression(),
        WhiteNoise(std_error=sigma_v),
    )
    model.auto_initialize_baseline_states(train_data["y"][1:24])

    # Training
    scheduled_sigma_v = 1
    for epoch in range(num_epoch):
        scheduled_sigma_v = exponential_scheduler(
            curr_v=scheduled_sigma_v,
            min_v=sigma_v,
            decaying_factor=0.99,
            curr_iter=epoch,
        )
        noise_index = model.states_name.index("white noise")
        model.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

        mu_validation_preds, std_validation_preds, _ = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
        )

        # Unstandardize predictions
        mu_validation_preds = normalizer.unstandardize(
            mu_validation_preds,
            data_processor.norm_const_mean[output_col],
            data_processor.norm_const_std[output_col],
        )

        mse = metric.mse(
            mu_validation_preds, data_processor.validation_data[:, output_col].flatten()
        )

        model.early_stopping(evaluate_metric=mse, mode="min")
        if model.stop_training:
            break

    return model.early_stop_metric


# Create and run the study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Print the best parameters and results
print("Best parameters:", study.best_params)
print("Best validation MSE:", study.best_value)
