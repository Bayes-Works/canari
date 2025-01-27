import optuna
from optuna.samplers import GridSampler
import pandas as pd
from typing import Optional, Tuple
from pytagi import Normalizer as normalizer
from pytagi import exponential_scheduler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from examples import DataProcess
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
import pytagi.metric as metric

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


# Build model
def build_model(sigma_v: float, look_back_len: int) -> Model:
    """ """

    model = Model(
        LocalTrend(),
        LstmNetwork(
            look_back_len=look_back_len,
            num_features=2,
            num_layer=1,
            num_hidden_unit=50,
            device="cpu",
            manual_seed=1,
        ),
        WhiteNoise(std_error=sigma_v),
    )
    return model


# Training
def train_model(model: Model, sigma_v: float):
    """ """

    max_num_epoch = 50
    model.auto_initialize_baseline_states(train_data["y"][1:24])
    scheduled_sigma_v = 1

    for epoch in range(max_num_epoch):
        # Schedule noise
        scheduled_sigma_v = exponential_scheduler(
            curr_v=scheduled_sigma_v,
            min_v=sigma_v,
            decaying_factor=0.9,
            curr_iter=epoch,
        )
        noise_index = model.states_name.index("white noise")
        model.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

        # Train and validate
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
        std_validation_preds = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor.norm_const_std[output_col],
        )

        # Evaluate
        mse = metric.mse(
            mu_validation_preds, data_processor.validation_data[:, output_col].flatten()
        )

        model.early_stopping(evaluate_metric=mse, mode="min")
        if model.stop_training:
            break

    return model.early_stop_metric, mu_validation_preds, std_validation_preds


# Objective function for optimization
def objective_model(trial):
    """
    Optuna objective function to tune hyperparameters.
    """
    sigma_v = trial.suggest_loguniform("sigma_v", 1e-3, 1e-1)
    look_back_len = trial.suggest_int("look_back_len", 12, 48)

    model = build_model(sigma_v, look_back_len)
    validation_metric, _, _ = train_model(model, sigma_v)

    return validation_metric


# Run optimization for model
model_optimizer = optuna.create_study(direction="minimize")
model_optimizer.optimize(objective_model, n_trials=50)
print("Best model parameters:", model_optimizer.best_params)

# Re-train with optimal parameters
optimal_sigma_v = model_optimizer.best_params["sigma_v"]
optimal_look_back_len = model_optimizer.best_params["look_back_len"]

optimal_model = build_model(optimal_sigma_v, optimal_look_back_len)
_, mu_validation_preds, std_validation_preds = train_model(
    optimal_model, optimal_sigma_v
)

fig, ax = plt.subplots(figsize=(10, 6))
plot_data(
    data_processor=data_processor,
    plot_column=output_col,
    plot_test_data=False,
    sub_plot=ax,
    validation_label="y",
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds,
    std_validation_pred=std_validation_preds,
    sub_plot=ax,
    validation_label=[r"$\mu$", f"$\pm\sigma$"],
)
ax.set_xlabel("Time")
plt.title("Validation predictions")
plt.tight_layout()
plt.legend()
plt.show()


# # # Anomaly detection
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


search_space = {
    "std_transition_error": [1e-6, 1e-5, 1e-4],
    "norm_to_abnorm_prob": [1e-6, 1e-5, 1e-4],
    "slope": [0.01, 0.03, 0.05],
}


def objective_SKF(trial):
    # Sample parameters from the search space
    # std_transition_error = trial.suggest_loguniform("std_transition_error", 1e-6, 1e-3)
    # norm_to_abnorm_prob = trial.suggest_loguniform("norm_to_abnorm_prob", 1e-6, 1e-3)
    # slope = trial.suggest_uniform("slope", 0.01, 0.1)

    std_transition_error = trial.suggest_categorical(
        "std_transition_error", search_space["std_transition_error"]
    )
    norm_to_abnorm_prob = trial.suggest_categorical(
        "norm_to_abnorm_prob", search_space["norm_to_abnorm_prob"]
    )
    slope = trial.suggest_categorical("slope", search_space["slope"])

    # Initialize SKF with sampled parameters
    optimal_model.lstm_net.reset_lstm_states()
    skf = SKF(
        norm_model=optimal_model,
        abnorm_model=ab_model,
        std_transition_error=std_transition_error,
        norm_to_abnorm_prob=norm_to_abnorm_prob,
        abnorm_to_norm_prob=1e-1,  # Fixed value
        norm_model_prior_prob=0.99,  # Fixed value
    )

    # Generate synthetic training data
    num_synthetic_anomaly = 50
    synthetic_train_data, _ = DataProcess.add_synthetic_anomaly(
        train_data, num_samples=num_synthetic_anomaly, slope=slope
    )

    # Compute detection rate
    detection_rate = skf.detect_synthetic_anomaly(data=synthetic_train_data)

    if detection_rate > 0.6:
        detection_rate = 0

    # Define objective: minimize detection rate and slope
    score = detection_rate - slope  # Add slope to detection rate

    return score  # Lower scores are better


# Run optimization for SKF
# SKF_optimizer = optuna.create_study(direction="maximize")
# SKF_optimizer.optimize(objective_SKF, n_trials=50)
# print("Best SKF parameters:", SKF_optimizer.best_params)

# Use the GridSampler
sampler = GridSampler(search_space)
SKF_optimizer = optuna.create_study(direction="maximize", sampler=sampler)
SKF_optimizer.optimize(objective_SKF)
print("Best SKF parameters:", SKF_optimizer.best_params)

# Re-run with optimal SKF parameters
optimal_std_transition_error = SKF_optimizer.best_params["std_transition_error"]
optimal_norm_to_abnorm_prob = SKF_optimizer.best_params["norm_to_abnorm_prob"]

optimal_model.lstm_net.reset_lstm_states()
skf_optimal = SKF(
    norm_model=optimal_model,
    abnorm_model=ab_model,
    std_transition_error=optimal_std_transition_error,
    norm_to_abnorm_prob=optimal_norm_to_abnorm_prob,
    # std_transition_error=0.00041951244650633985,
    # norm_to_abnorm_prob=1.5082032194417e-06,
    # std_transition_error=1e-4,
    # norm_to_abnorm_prob=1e-4,
    abnorm_to_norm_prob=1e-1,
    norm_model_prior_prob=0.99,
)

filter_marginal_abnorm_prob, _ = skf_optimal.filter(data=all_data)
smooth_marginal_abnorm_prob, states = skf_optimal.smoother(data=all_data)

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
