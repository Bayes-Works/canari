import optuna
import pandas as pd
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

# Components
sigma_v = 1e-2
local_trend = LocalTrend(var_states=[1e-2, 1e-2])
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=12,
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
    manual_seed=1,
)
noise = WhiteNoise(std_error=sigma_v)

# Normal model
model = Model(
    local_trend,
    lstm_network,
    noise,
)
model.auto_initialize_baseline_states(train_data["y"][0:23])

#  Training
num_epoch = 50
scheduled_sigma_v = 1
for epoch in tqdm(range(num_epoch), desc="Training Progress", unit="epoch"):
    # # Decaying observation's variance
    scheduled_sigma_v = exponential_scheduler(
        curr_v=scheduled_sigma_v, min_v=sigma_v, decaying_factor=0.9, curr_iter=epoch
    )
    noise_index = model.states_name.index("white noise")
    model.process_noise_matrix[noise_index, noise_index] = scheduled_sigma_v**2

    # Train the model
    (mu_validation_preds, std_validation_preds, train_states) = model.lstm_train(
        train_data=train_data, validation_data=validation_data
    )

    # # Unstandardize the predictions
    mu_validation_preds = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.norm_const_mean[output_col],
        data_processor.norm_const_std[output_col],
    )
    std_validation_preds = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.norm_const_std[output_col],
    )

    # Calculate the log-likelihood metric
    validation_log_lik = metric.log_likelihood(
        prediction=mu_validation_preds,
        observation=data_processor.validation_data[:, output_col].flatten(),
        std=std_validation_preds,
    )

    # Early-stopping
    model.early_stopping(evaluate_metric=validation_log_lik, mode="max")
    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation log-likelihood  :{model.early_stop_metric: 0.4f}")

# # Anomaly Detection
ab_model = Model(
    local_acceleration,
    lstm_network,
    noise,
)


def objective(trial):
    # Sample parameters from the search space
    std_transition_error = trial.suggest_loguniform("std_transition_error", 1e-6, 1e-3)
    norm_to_abnorm_prob = trial.suggest_loguniform("norm_to_abnorm_prob", 1e-6, 1e-3)
    slope = trial.suggest_uniform("slope", 0.01, 0.1)

    # Initialize SKF with sampled parameters
    model.lstm_net.reset_lstm_states()
    skf = SKF(
        norm_model=model,
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

    if detection_rate > 0.5:
        detection_rate = 0

    # Define objective: minimize detection rate and slope
    score = detection_rate - slope  # Add slope to detection rate

    return score  # Lower scores are better


study = optuna.create_study(direction="maximize")  # Minimizing the combined score
study.optimize(objective, n_trials=50)
# Print the best parameters
print("Best parameters:", study.best_params)
print("Best combined score:", study.best_value)

# # # Switching Kalman filter
model.lstm_net.reset_lstm_states()
skf_optimal = SKF(
    norm_model=model,
    abnorm_model=ab_model,
    std_transition_error=study.best_params["std_transition_error"],
    norm_to_abnorm_prob=study.best_params["norm_to_abnorm_prob"],
    # std_transition_error=0.00041951244650633985,
    # norm_to_abnorm_prob=1.5082032194417e-06,
    # std_transition_error=1e-4,
    # norm_to_abnorm_prob=1e-4,
    # std_transition_error=3.0478629661685305e-06,
    # norm_to_abnorm_prob=5.484594028583117e-05,
    abnorm_to_norm_prob=1e-1,
    norm_model_prior_prob=0.99,
)

# # Anomaly Detection with optimal parameters
filter_marginal_abnorm_prob, _ = skf_optimal.filter(data=all_data)
smooth_marginal_abnorm_prob, states = skf_optimal.smoother(data=all_data)

# # Plot
marginal_abnorm_prob_plot = filter_marginal_abnorm_prob
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


fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=states,
    states_to_plot=["local level", "local trend", "lstm", "white noise"],
    model_prob=marginal_abnorm_prob_plot,
    color="b",
    legend_location="upper left",
)
fig.suptitle("SKF hidden states", fontsize=10, y=1)
plt.show()
