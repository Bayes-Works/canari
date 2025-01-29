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
data_file = "./data/test_data.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=";", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 4])
df_raw = df_raw.iloc[:, 6].to_frame()
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]
df = df_raw.resample("W").mean()
df = df.iloc[30:, :]
# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df,
    time_covariates=["week_of_year"],
    train_split=0.25,
    validation_split=0.08,
    test_split=0.67,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# fig, ax = plt.subplots(figsize=(10, 6))
# plot_data(
#     data_processor=data_processor,
#     plot_column=output_col,
#     plot_test_data=True,
#     sub_plot=ax,
#     validation_label="y",
# )
# ax.set_xlabel("Time")
# plt.tight_layout()
# plt.legend()
# plt.show()


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
            manual_seed=1,
        ),
        WhiteNoise(std_error=sigma_v),
    )
    model.auto_initialize_baseline_states(train_data["y"][0:52])

    # Training process
    scheduled_sigma_v = 1
    num_epoch = 50
    for epoch in range(num_epoch):
        scheduled_sigma_v = exponential_scheduler(
            curr_v=scheduled_sigma_v,
            min_v=sigma_v,
            decaying_factor=0.9,
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
        return model


# Use the model_optimizer function with Optuna for tuning
study = optuna.create_study(direction="minimize")
study.optimize(model_optimizer, n_trials=5)

# Print the best parameters and results
print("Best parameters:", study.best_params)
print("Best validation MSE:", study.best_value)

# Use the model_optimizer function to train with the best parameters
best_model = model_optimizer(best_params=study.best_params)

# #
# optimal_sigma_v = study.best_params["sigma_v"]
# optimal_look_back_len = study.best_params["look_back_len"]

optimal_sigma_v = 0.001387094942864236
optimal_look_back_len = 13
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

# Initialize SKF with parameters
best_model.lstm_net.reset_lstm_states()
skf = SKF(
    norm_model=best_model,
    abnorm_model=ab_model,
    std_transition_error=1e-4,
    norm_to_abnorm_prob=1e-4,
    abnorm_to_norm_prob=1e-1,  # Fixed value
    norm_model_prior_prob=0.99,  # Fixed value
)
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


# def objective_SKF(trial=None, best_params=None):
#     if trial is not None:
#         std_transition_error = trial.suggest_loguniform(
#             "std_transition_error", 1e-6, 1e-3
#         )
#         norm_to_abnorm_prob = trial.suggest_loguniform(
#             "norm_to_abnorm_prob", 1e-6, 1e-3
#         )
#         slope = trial.suggest_uniform("slope", 0.01, 0.1)
#     elif best_params is not None:
#         std_transition_error = best_params["std_transition_error"]
#         norm_to_abnorm_prob = best_params["norm_to_abnorm_prob"]
#         slope = best_params["slope"]
#     else:
#         raise ValueError("Either `trial` or `best_params` must be provided.")

#     # Initialize SKF with parameters
#     best_model.lstm_net.reset_lstm_states()
#     skf = SKF(
#         norm_model=best_model,
#         abnorm_model=ab_model,
#         std_transition_error=std_transition_error,
#         norm_to_abnorm_prob=norm_to_abnorm_prob,
#         abnorm_to_norm_prob=1e-1,  # Fixed value
#         norm_model_prior_prob=0.99,  # Fixed value
#     )

#     if trial is not None:
#         # Generate synthetic training data
#         num_synthetic_anomaly = 50
#         synthetic_train_data, _ = DataProcess.add_synthetic_anomaly(
#             train_data, num_samples=num_synthetic_anomaly, slope=slope
#         )

#         # Compute detection rate
#         detection_rate = skf.detect_synthetic_anomaly(data=synthetic_train_data)

#         if detection_rate > 0.5:
#             detection_rate = 0

#         # Define objective: minimize detection rate and slope
#         score = detection_rate - slope  # Add slope to detection rate
#         return score
#     else:
#         # If used for fixed parameter training or evaluation, return the SKF instance
#         filter_marginal_abnorm_prob, _ = skf.filter(data=all_data)
#         smooth_marginal_abnorm_prob, states = skf.smoother(data=all_data)

#         fig, ax = plot_skf_states(
#             data_processor=data_processor,
#             states=states,
#             states_to_plot=["local level", "local trend", "lstm", "white noise"],
#             model_prob=filter_marginal_abnorm_prob,
#             color="b",
#             legend_location="upper left",
#         )
#         fig.suptitle("SKF hidden states", fontsize=10, y=1)
#         plt.show()


# # Use the objective_SKF function with Optuna
# study_SKF = optuna.create_study(direction="minimize")
# study_SKF.optimize(objective_SKF, n_trials=50)

# # Print the best parameters and results
# print("Best parameters:", study_SKF.best_params)

# # Use the objective_SKF function with the best parameters
# objective_SKF(best_params=study_SKF.best_params)
