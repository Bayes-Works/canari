import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend, ExpSmoothing, LocalLevel

# # Read data
ts = 1
# training set
data_train_file = "./data/tourism/quarterly_in.csv"
df_train = pd.read_csv(data_train_file, skiprows=0, delimiter=",", header=None, usecols=[ts]).dropna()
quarter_to_month = {1: 1, 2: 4, 3: 7, 4: 10}
train_start_time = pd.Timestamp(
    year=int(df_train.iloc[2, 0]),
    month=quarter_to_month[int(df_train.iloc[3, 0])],
    day=1
)
df_train = df_train.iloc[4:,:]
df_train = df_train.astype(float)
# test set
data_test_file = "./data/tourism/quarterly_oos.csv"
df_test = pd.read_csv(data_test_file, skiprows=0, delimiter=",", header=None, usecols=[ts]).dropna()
df_test = df_test.iloc[4:,:]
df_test = df_test.astype(float)

df = pd.concat([df_train, df_test], axis=0)

df.index = pd.date_range(
    start=train_start_time,
    periods=len(df),
    freq="QS"
)

# Define parameters
output_col = [0]
num_epoch = 50
nb_val = 4

# Build data processor
data_processor = DataProcess(
    data=df,
    train_start=df.index[0],
    validation_start=df.index[len(df_train) - nb_val],
    test_start=df.index[len(df_train)],
    time_covariates=["quarter_of_year"],
    output_col=output_col,
)
# split data
train_data, validation_data, test_data, _ = data_processor.get_splits()

# Model
model = Model(
    LocalTrend(),
    # ExpSmoothing(mu_states=[0,-0.5,0], var_states=[0,0.2,0], es_order=1, activation="sigmoid"),
    ExpSmoothing(mu_states=[0,0.2,0], var_states=[0,1e-2,0], es_order=1, activation=None),
    LstmNetwork(
        look_back_len=4,
        num_features=2,
        infer_len=4 *3,
        num_layer=1,
        num_hidden_unit=50,
        manual_seed=1,
        model_noise=True,
        # smoother=False,
    ),
)

model.auto_initialize_baseline_states(train_data["y"])

# Training
for epoch in range(num_epoch):
    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
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

    # Early-stopping
    model.early_stopping(
        evaluate_metric=-validation_log_lik, current_epoch=epoch, max_epoch=num_epoch,
        # skip_epoch=20,
    )

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

# plot the test data
level_sum = states.get_mean(states_name="level") + states.get_mean(states_name="es")

for i in range(len(states.mu_posterior)):
    states.mu_posterior[i][0] = level_sum[i]

fig, ax = plot_states(
    data_processor=data_processor,
    states=states,
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
    test_label=[r"$\mu^{\prime}$", r"$\pm\sigma^{\prime}$"],
    color="purple",
)
plt.legend(loc=(0.1, 1.01), ncol=6, fontsize=12)
plt.tight_layout()
plt.show()
