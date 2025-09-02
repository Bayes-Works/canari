import copy
import pandas as pd
from pytagi import Normalizer as normalizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytagi.metric as metric
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_data,
    plot_prediction,
    plot_skf_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise

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
time_anomaly = 120
new_trend = np.linspace(0, 1, num=len(df_raw) - time_anomaly)
trend[time_anomaly:] = trend[time_anomaly:] + new_trend
df_raw = df_raw.add(trend, axis=0)

# Data pre-processing
output_col = [0]
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["hour_of_day"],
    train_split=0.4,
    validation_split=0.1,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Components
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=10,
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
    manual_seed=2,
    # model_noise=True,
)
noise = WhiteNoise(std_error=1e-2)

# Normal model
model = Model(
    local_trend,
    lstm_network,
    noise,
)

#  Abnormal model
ab_model = Model(
    local_acceleration,
    lstm_network,
    noise,
)

# Switching Kalman filter
skf = SKF(
    norm_model=model,
    abnorm_model=ab_model,
    std_transition_error=1e-4,
    norm_to_abnorm_prob=1e-4,
)
skf.auto_initialize_baseline_states(train_data["y"][0:24])

#  Training
num_epoch = 50
states_optim = None
mu_validation_preds_optim = None
std_validation_preds_optim = None

for epoch in tqdm(range(num_epoch), desc="Training Progress", unit="epoch"):
    # Train the model
    (mu_validation_preds, std_validation_preds, states) = skf.lstm_train(
        train_data=train_data, validation_data=validation_data
    )
    skf.model["norm_norm"].set_memory(states=states, time_step=0)

    # # Unstandardize the predictions
    mu_validation_preds_unnorm = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.scale_const_mean[data_processor.output_col],
        data_processor.scale_const_std[data_processor.output_col],
    )

    std_validation_preds_unnorm = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.scale_const_std[data_processor.output_col],
    )

    validation_obs = data_processor.get_data("validation").flatten()
    validation_log_lik = metric.log_likelihood(
        prediction=mu_validation_preds_unnorm,
        observation=validation_obs,
        std=std_validation_preds_unnorm,
    )

    # Early-stopping
    skf.early_stopping(
        evaluate_metric=-validation_log_lik, current_epoch=epoch, max_epoch=num_epoch
    )
    if epoch == skf.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds.copy()
        std_validation_preds_optim = std_validation_preds.copy()
        states_optim = copy.copy(states)

    if skf.stop_training:
        break

print(f"Optimal epoch       : {skf.optimal_epoch}")
print(f"Validation log-likelihood  :{skf.early_stop_metric: 0.4f}")

# # Anomaly Detection
filter_marginal_abnorm_prob, _ = skf.filter(data=all_data)
# smooth_marginal_abnorm_prob, states = skf.smoother()

# smooth_marginal_abnorm_prob, states = skf.smoother(
#     matrix_inversion_tol=1e-5, tol_type="relative"
# )

smooth_marginal_abnorm_prob, states = skf.smoother(
    matrix_inversion_tol=1e-3, tol_type="absolute"
)

# from scipy.io import savemat

# # Norm_norm
# nn_mu_prior = np.stack(skf.model["norm_norm"].states.mu_prior, axis=0)
# nn_mu_prior = np.asarray(nn_mu_prior, dtype=np.float64)
# nn_var_prior = np.stack(skf.model["norm_norm"].states.var_prior, axis=0)
# nn_var_prior = np.asarray(nn_var_prior, dtype=np.float64)
# nn_mu_pos = np.stack(skf.model["norm_norm"].states.mu_posterior, axis=0)
# nn_mu_pos = np.asarray(nn_mu_pos, dtype=np.float64)
# nn_var_pos = np.stack(skf.model["norm_norm"].states.var_posterior, axis=0)
# nn_var_pos = np.asarray(nn_var_pos, dtype=np.float64)
# nn_cov = np.stack(skf.model["norm_norm"].states.cov_states, axis=0)
# nn_cov = np.asarray(nn_cov, dtype=np.float64)
# # Norm_abnorm
# nab_mu_prior = np.stack(skf.model["norm_abnorm"].states.mu_prior, axis=0)
# nab_mu_prior = np.asarray(nab_mu_prior, dtype=np.float64)
# nab_var_prior = np.stack(skf.model["norm_abnorm"].states.var_prior, axis=0)
# nab_var_prior = np.asarray(nab_var_prior, dtype=np.float64)
# nab_mu_pos = np.stack(skf.model["norm_abnorm"].states.mu_posterior, axis=0)
# nab_mu_pos = np.asarray(nab_mu_pos, dtype=np.float64)
# nab_var_pos = np.stack(skf.model["norm_abnorm"].states.var_posterior, axis=0)
# nab_var_pos = np.asarray(nab_var_pos, dtype=np.float64)
# nab_cov = np.stack(skf.model["norm_abnorm"].states.cov_states, axis=0)
# nab_cov = np.asarray(nab_cov, dtype=np.float64)
# # Aborm_norm
# abn_mu_prior = np.stack(skf.model["abnorm_norm"].states.mu_prior, axis=0)
# abn_mu_prior = np.asarray(abn_mu_prior, dtype=np.float64)
# abn_var_prior = np.stack(skf.model["abnorm_norm"].states.var_prior, axis=0)
# abn_var_prior = np.asarray(abn_var_prior, dtype=np.float64)
# abn_mu_pos = np.stack(skf.model["abnorm_norm"].states.mu_posterior, axis=0)
# abn_mu_pos = np.asarray(abn_mu_pos, dtype=np.float64)
# abn_var_pos = np.stack(skf.model["abnorm_norm"].states.var_posterior, axis=0)
# abn_var_pos = np.asarray(abn_var_pos, dtype=np.float64)
# abn_cov = np.stack(skf.model["abnorm_norm"].states.cov_states, axis=0)
# abn_cov = np.asarray(abn_cov, dtype=np.float64)
# # Aborm_abnorm
# abab_mu_prior = np.stack(skf.model["abnorm_abnorm"].states.mu_prior, axis=0)
# abab_mu_prior = np.asarray(abab_mu_prior, dtype=np.float64)
# abab_var_prior = np.stack(skf.model["abnorm_abnorm"].states.var_prior, axis=0)
# abab_var_prior = np.asarray(abab_var_prior, dtype=np.float64)
# abab_mu_pos = np.stack(skf.model["abnorm_abnorm"].states.mu_posterior, axis=0)
# abab_mu_pos = np.asarray(abab_mu_pos, dtype=np.float64)
# abab_var_pos = np.stack(skf.model["abnorm_abnorm"].states.var_posterior, axis=0)
# abab_var_pos = np.asarray(abab_var_pos, dtype=np.float64)
# abab_cov = np.stack(skf.model["abnorm_abnorm"].states.cov_states, axis=0)
# abab_cov = np.asarray(abab_cov, dtype=np.float64)

# savemat(
#     "toy_anomaly.mat",
#     {
#         "nn_mu_prior": nn_mu_prior,
#         "nn_var_prior": nn_var_prior,
#         "nn_mu_pos": nab_mu_pos,
#         "nn_var_pos": nn_var_pos,
#         "nn_cov": nn_cov,
#         "nab_mu_prior": nab_mu_prior,
#         "nab_var_prior": nab_var_prior,
#         "nab_mu_pos": nab_mu_pos,
#         "nab_var_pos": nab_var_pos,
#         "nab_cov": nab_cov,
#         "abn_mu_prior": abn_mu_prior,
#         "abn_var_prior": abn_var_prior,
#         "abn_mu_pos": abn_mu_pos,
#         "abn_var_pos": abn_var_pos,
#         "abn_cov": abn_cov,
#         "abab_mu_prior": abab_mu_prior,
#         "abab_var_prior": abab_var_prior,
#         "abab_mu_pos": abab_mu_pos,
#         "abab_var_pos": abab_var_pos,
#         "abab_cov": abab_cov,
#     },
# )

data_processor.data.index = range(len(data_processor.data))

# # Plot
marginal_abnorm_prob_plot = filter_marginal_abnorm_prob
# fig, ax = plt.subplots(figsize=(10, 6))
# plot_data(
#     data_processor=data_processor,
#     plot_column=output_col,
#     standardization=True,
#     plot_test_data=False,
#     sub_plot=ax,
#     validation_label="y",
# )
# plot_prediction(
#     data_processor=data_processor,
#     mean_validation_pred=mu_validation_preds_optim,
#     std_validation_pred=std_validation_preds_optim,
#     sub_plot=ax,
#     validation_label=[r"$\mu$", f"$\pm\sigma$"],
# )
# ax.set_xlabel("Time")
# plt.title("Validation predictions")
# plt.tight_layout()
# plt.legend()
# plt.show()


# fig, ax = plot_skf_states(
#     data_processor=data_processor,
#     states=states,
#     states_type="posterior",
#     model_prob=marginal_abnorm_prob_plot,
#     # standardization=True,
#     color="b",
#     legend_location="upper left",
# )
# fig.suptitle("SKF hidden states", fontsize=10, y=1)
# plt.show()

fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=states,
    states_type="smooth",
    model_prob=marginal_abnorm_prob_plot,
    # standardization=True,
    color="b",
    legend_location="upper left",
)
fig.suptitle("SKF hidden states", fontsize=10, y=1)
plt.show()


fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=skf.model["norm_norm"].states,
    states_type="smooth",
    model_prob=marginal_abnorm_prob_plot,
    color="b",
    legend_location="upper left",
)
fig.suptitle("norm_norm", fontsize=10, y=1)
plt.show()

fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=skf.model["norm_abnorm"].states,
    states_type="smooth",
    model_prob=marginal_abnorm_prob_plot,
    color="b",
    legend_location="upper left",
)
fig.suptitle("norm_abnorm", fontsize=10, y=1)
plt.show()

fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=skf.model["abnorm_norm"].states,
    states_type="smooth",
    model_prob=marginal_abnorm_prob_plot,
    color="b",
    legend_location="upper left",
)
fig.suptitle("abnorm_norm", fontsize=10, y=1)
plt.show()

fig, ax = plot_skf_states(
    data_processor=data_processor,
    states=skf.model["abnorm_abnorm"].states,
    states_type="smooth",
    model_prob=marginal_abnorm_prob_plot,
    color="b",
    legend_location="upper left",
)
fig.suptitle("abnorm_abnorm", fontsize=10, y=1)
plt.show()
