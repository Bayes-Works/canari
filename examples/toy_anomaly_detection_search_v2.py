import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import (
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    WhiteNoise,
    Model,
    SKF,
    plot_data,
    plot_prediction,
    plot_skf_states,
)
from examples import DataProcess
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer


# # Read data
data_file = "./data/toy_time_series/sine.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/sine_datetime.csv"
time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(time_series[0])
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]

# Add synthetic anomaly to data
trend = np.linspace(0, 0, num=len(df_raw))
time_anomaly = 200
new_trend = np.linspace(0, 2, num=len(df_raw) - time_anomaly)
trend[time_anomaly:] = trend[time_anomaly:] + new_trend
df_raw = df_raw.add(trend, axis=0)

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["hour_of_day"],
    train_start="2000-01-01 00:00:00",
    train_end="2000-01-09 23:00:00",
    validation_start="2000-01-10 00:00:00",
    validation_end="2000-01-10 23:00:00",
    test_start="2000-01-11 00:00:00",
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()


# #
def model_optimizer(trial=None, best_params=None):
    if trial is not None:
        sigma_v = trial.suggest_loguniform("sigma_v", 1e-3, 1e-1)
        look_back_len = trial.suggest_int("look_back_len", 12, 48)
    elif best_params is not None:
        sigma_v = best_params["sigma_v"]
        look_back_len = best_params["look_back_len"]
    else:
        raise ValueError("Either `trial` or `best_params` must be provided.")

    # Initialize the model
    model = Model(
        LocalTrend(),
        LstmNetwork(
            look_back_len=look_back_len,
            num_features=2,
            num_layer=1,
            num_hidden_unit=50,
            device="cpu",
        ),
        WhiteNoise(std_error=sigma_v),
    )
    model.auto_initialize_baseline_states(train_data["y"][1:24])

    # Training process
    scheduled_sigma_v = 1
    num_epoch = 50
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
            mu_validation_preds,
            data_processor.validation_data[:, output_col].flatten(),
        )

        model.early_stopping(evaluate_metric=mse, mode="min")
        if model.stop_training:
            break

    if trial is not None:
        return model.early_stop_metric
    else:
        print(f"Final validation MSE: {model.early_stop_metric}")
        return model


# Use the model_optimizer function with Optuna for tuning
study = optuna.create_study(direction="minimize")
study.optimize(model_optimizer, n_trials=20)

# Print the best parameters and results
print("Best parameters:", study.best_params)
print("Best validation MSE:", study.best_value)

# Use the model_optimizer function to train with the best parameters
best_model = model_optimizer(best_params=study.best_params)

# #
optimal_sigma_v = study.best_params["sigma_v"]
optimal_look_back_len = study.best_params["look_back_len"]
ab_model = Model(
    LocalAcceleration(),
    LstmNetwork(
        look_back_len=optimal_look_back_len,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        manual_seed=1,
    ),
    WhiteNoise(std_error=optimal_sigma_v),
)


def objective_SKF(trial=None, best_params=None):
    if trial is not None:
        std_transition_error = trial.suggest_loguniform(
            "std_transition_error", 1e-6, 1e-3
        )
        norm_to_abnorm_prob = trial.suggest_loguniform(
            "norm_to_abnorm_prob", 1e-6, 1e-3
        )
        slope = trial.suggest_uniform("slope", 0.01, 0.1)
    elif best_params is not None:
        std_transition_error = best_params["std_transition_error"]
        norm_to_abnorm_prob = best_params["norm_to_abnorm_prob"]
        slope = best_params["slope"]
    else:
        raise ValueError("Either `trial` or `best_params` must be provided.")

    # Initialize SKF with parameters
    best_model.lstm_net.reset_lstm_states()
    skf = SKF(
        norm_model=best_model,
        abnorm_model=ab_model,
        std_transition_error=std_transition_error,
        norm_to_abnorm_prob=norm_to_abnorm_prob,
        abnorm_to_norm_prob=1e-1,  # Fixed value
        norm_model_prior_prob=0.99,  # Fixed value
    )

    if trial is not None:
        # Generate synthetic training data
        num_synthetic_anomaly = 50
        synthetic_train_data, _ = DataProcess.add_synthetic_anomaly(
            train_data, num_samples=num_synthetic_anomaly, slope=slope
        )

        # Compute detection rate
        detection_rate = skf.detect_synthetic_anomaly(data=synthetic_train_data)

        if detection_rate > 0.5:
            detection_rate = 0

        # Define objective: minimize detection rate and slope
        score = detection_rate - slope  # Add slope to detection rate
        return score
    else:
        # If used for fixed parameter training or evaluation, return the SKF instance
        filter_marginal_abnorm_prob, _ = skf.filter(data=all_data)
        smooth_marginal_abnorm_prob, states = skf.smoother(data=all_data)

        fig, ax = plot_skf_states(
            data_processor=data_processor,
            states=states,
            states_to_plot=["local level", "local trend", "lstm", "white noise"],
            model_prob=filter_marginal_abnorm_prob,
            color="b",
            legend_location="upper left",
        )
        fig.suptitle("SKF hidden states", fontsize=10, y=1)
        plt.show()


# Use the objective_SKF function with Optuna
study_SKF = optuna.create_study(direction="minimize")
study_SKF.optimize(objective_SKF, n_trials=50)

# Print the best parameters and results
print("Best parameters:", study_SKF.best_params)

# Use the objective_SKF function with the best parameters
objective_SKF(best_params=study_SKF.best_params)
