import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend, ExpSmoothing, LocalLevel
import json

# # Read data
benchmark_no: str = ["1"]

with open("examples/benchmark/BM_metadata.json", "r") as f:
    metadata = json.load(f)
config = metadata[benchmark_no[0]]
######### Data processing #########
# Read data
data_file = config["data_path"]
df = pd.read_csv(data_file, skiprows=0, delimiter=",")
date_time = pd.to_datetime(df["date"])
df = df.drop("date", axis=1)
df.index = date_time
df.index.name = "date_time"
# Data pre-processing
df = DataProcess.add_lagged_columns(df, config["lag_vector"])
output_col = config["output_col"]
data_processor = DataProcess(
    data=df,
    time_covariates=config["time_covariates"],
    train_split=0.8,
    validation_split=0.1,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Model
model = Model(
    # LocalLevel(),
    LocalTrend(),
    ExpSmoothing(mu_states=[0,.5,0], var_states=[0,1e-2,0], es_order=1, activation=None),
    LstmNetwork(
        look_back_len=24,
        num_features=config["num_feature"],
        num_layer=1,
        infer_len=52 * 3,
        num_hidden_unit=50,
        manual_seed=1,
        model_noise=True,
        # smoother=False,
    ),
)

model.auto_initialize_baseline_states(train_data["y"])

# plot_data(
#     data_processor=data_processor,
#     standardization=True,
#     plot_column=output_col,
#     plot_test_data=False,
# )
# plt.show()

# Training
num_epoch = 50
for epoch in range(num_epoch):
    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
        white_noise_decay = True,
    )

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


    # Calculate the metric
    validation_obs = data_processor.get_data("validation").flatten()
    validation_log_lik = metric.log_likelihood(
        prediction=mu_validation_preds,
        observation=validation_obs,
        std=std_validation_preds,
    )


    fig, ax = plot_states(
        data_processor=data_processor,
        states=model.states,
        standardization=True,
        color="b",
        )
    plot_data(
        data_processor=data_processor,
        standardization=True,
        plot_column=output_col,
        plot_test_data=False,
        sub_plot=ax[0],
    )
    fig.suptitle(f"Epoch #{epoch}", fontsize=10, y=1)
    plt.show()

    # Early-stopping
    # model.early_stopping(
    #     evaluate_metric=-validation_log_lik, current_epoch=epoch, max_epoch=num_epoch,
    #     skip_epoch=5,
    # )

    # # Calculate the log-likelihood metric
    # validation_obs = data_processor.get_data("validation").flatten()
    # mse = metric.mse(mu_validation_preds, validation_obs)

    # # Early-stopping
    # model.early_stopping(evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch,
    #                      skip_epoch=5)

    if model.stop_training:
        break


print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation log-likelihood      :{model.early_stop_metric: 0.4f}")

model.set_memory(
    time_step=data_processor.test_start - 1,
)

# forecat on the test set
mu_test_preds, std_test_preds, test_states = model.forecast(
    data=test_data,
)

# Unstandardize the predictions
mu_test_preds = normalizer.unstandardize(
    mu_test_preds,
    data_processor.scale_const_mean[output_col],
    data_processor.scale_const_std[output_col],
)
std_test_preds = normalizer.unstandardize_std(
    std_test_preds,
    data_processor.scale_const_std[output_col],
)

# calculate the test metrics
test_obs = data_processor.get_data("all").flatten()

# # plot the test data
level_sum = model.states.get_mean(states_name="level") + model.states.get_mean(states_name="es")

for i in range(len(model.states.mu_posterior)):
    model.states.mu_posterior[i][0] = level_sum[i]

fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    standardization=True,
    color="b",
)
plot_data(
    data_processor=data_processor,
    standardization=True,
    plot_column=output_col,
    plot_test_data=False,
    sub_plot=ax[0],
)
fig.suptitle("SKF hidden states", fontsize=10, y=1)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
plot_data(
    data_processor=data_processor,
    plot_train_data=True,
    plot_validation_data=True,
    plot_test_data=True,
    standardization=False,
    plot_column=output_col,
)
plot_data(
    data_processor=data_processor,
    standardization=False,
    plot_column=output_col,
)
plot_prediction(
    data_processor=data_processor,
    mean_test_pred=mu_test_preds,
    std_test_pred=std_test_preds,
)
plt.tight_layout()
plt.show()
