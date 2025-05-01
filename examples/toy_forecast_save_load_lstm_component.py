import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
from canari.component import LocalTrend, LstmNetwork, WhiteNoise


# # Read data
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

# Model
sigma_v = 1e-2
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=12,
        num_features=1,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        # manual_seed=3,
    ),
    WhiteNoise(std_error=sigma_v),
)
model.auto_initialize_baseline_states(train_data["y"][1:24])

# Training
for epoch in range(num_epoch):
    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )

    # Unstandardize the predictions
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
    validation_obs = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_validation_preds, validation_obs)

    # Early-stopping
    model.early_stopping(evaluate_metric=mse, mode="min")
    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
        states_optim = copy.copy(states)
    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")


# Saved the trained lstm network
params_path = "saved_params/lstm_net_test.pth"
model.lstm_net.load_state_dict(model.early_stop_lstm_param)
model.lstm_net.save(filename=params_path)

# Define a new model that takes the trained lstm network
model2 = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=12,
        num_features=1,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
        load_lstm_net=params_path,  # Load the pre-trained LSTM network
    ),
    WhiteNoise(std_error=sigma_v),
)

# Provide model #2 the initial values from where model #1 stop
model2.mu_states = model.early_stop_init_mu_states
model2.var_states = model.early_stop_init_var_states
lstm_params_before_filter = copy.deepcopy(model2.lstm_net.state_dict())

# # #
model2.filter(data=train_data, train_lstm=False)
model2.smoother(data=train_data)
mu_validation_preds2, std_validation_preds2, _ = model2.forecast(validation_data)

lstm_params_after_filter = copy.deepcopy(model2.lstm_net.state_dict())
assert lstm_params_before_filter == lstm_params_after_filter
print(
    "LSTM network parameters are the same before and after filtering, smoothing, and forecasting."
)

# Unstandardize the predictions
mu_validation_preds2 = normalizer.unstandardize(
    mu_validation_preds2,
    data_processor.norm_const_mean[output_col],
    data_processor.norm_const_std[output_col],
)
std_validation_preds2 = normalizer.unstandardize_std(
    std_validation_preds2,
    data_processor.norm_const_std[output_col],
)

#  Plot
plot_data(
    data_processor=data_processor,
    normalization=False,
    plot_column=output_col,
    validation_label="y",
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds2,
    std_validation_pred=std_validation_preds2,
    validation_label=[r"$\mu$", f"$\pm\sigma$"],
)
plt.legend(loc="upper left")  # Change "upper right" to your desired location

plot_states(
    data_processor=data_processor,
    states=model2.states,
    states_type="prior",
)
plt.show()
