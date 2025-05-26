import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend
import canari.common as common
from pytagi.nn import OutputUpdater


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
num_epoch = 10

data_processor = DataProcess(
    data=df,
    time_covariates=["hour_of_day"],
    train_split=0.7,
    validation_split=0.1,
    output_col=output_col,
)

train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Model
sigma_v = 0.001
model = Model(
    LocalTrend(),
    LstmNetwork(
        look_back_len=19,
        num_features=2,
        num_layer=1,
        num_hidden_unit=40,
        device="cpu",
        manual_seed=1,
        smoother=True,
    ),
    WhiteNoise(std_error=sigma_v),
)
model.auto_initialize_baseline_states(train_data["y"][0:24])

if model.lstm_net.smooth:
    model.lstm_net.num_samples = (
        model.lstm_net.lstm_look_back_len
        + len(train_data["y"])
        # + len(validation_data["y"])
    )

# Training
for epoch in range(num_epoch):
    model.lstm_net.train()

    # set white noise decay
    model._white_noise_decay(epoch, white_noise_max_std=3, white_noise_decay_factor=0.9)

    if model.lstm_net.smooth:
        out_updater = OutputUpdater(model.lstm_net.device)
        for _ in range(0, model.lstm_net.lstm_look_back_len):
            input_covariates = [0.0]  # dummy input covariates
            mu_lstm_input, var_lstm_input = common.prepare_lstm_input(
                model.lstm_output_history, input_covariates
            )
            mu_lstm_pred, var_lstm_pred = model.lstm_net.forward(
                mu_x=np.float32(mu_lstm_input), var_x=np.float32(var_lstm_input)
            )
            out_updater.update(
                output_states=model.lstm_net.output_z_buffer,
                mu_obs=np.array([np.nan], dtype=np.float32),
                var_obs=np.array([0.0], dtype=np.float32),
                delta_states=model.lstm_net.input_delta_z_buffer,
            )
            # Feed backward
            model.lstm_net.backward()
            model.lstm_net.step()

            model.lstm_output_history.update(
                mu_lstm_pred,
                var_lstm_pred,
            )

    # filter on train data
    mu_preds_train, std_preds_train, states = model.filter(train_data, train_lstm=True)

    # smooth on train data
    model.smoother()

    model.lstm_net.eval()
    # forecast on the validation set
    mu_validation_preds, std_validation_preds, _ = model.forecast(validation_data)

    # Unstandardize the predictions
    mu_validation_preds = normalizer.unstandardize(
        mu_validation_preds,
        data_processor.std_const_mean[output_col],
        data_processor.std_const_std[output_col],
    )
    std_validation_preds = normalizer.unstandardize_std(
        std_validation_preds,
        data_processor.std_const_std[output_col],
    )

    # Calculate the log-likelihood metric
    validation_obs = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_validation_preds, validation_obs)

    lstm_optim_states = model.lstm_net.get_lstm_states()
    # smooth lstm states
    if model.lstm_net.smooth:
        mu_zo_smooth, var_zo_smooth = model.lstm_net.smoother()
        zo_smooth_std = np.array(var_zo_smooth) ** 0.5
        mu_sequence = mu_zo_smooth[: model.lstm_net.lstm_look_back_len]
        var_sequence = var_zo_smooth[: model.lstm_net.lstm_look_back_len]
        model.lstm_output_history.mu = mu_sequence
        model.lstm_output_history.var = var_sequence

        import matplotlib.cm as cm

        if "lstm_smooth_fig" not in globals():
            global lstm_smooth_fig, lstm_smooth_ax, lstm_smooth_colors
            lstm_smooth_fig, lstm_smooth_ax = plt.subplots(figsize=(10, 5))
            lstm_smooth_colors = cm.get_cmap("viridis", num_epoch)

        color = lstm_smooth_colors(epoch)
        lstm_smooth_ax.plot(
            mu_sequence, label=f"LSTM smoothed mu (epoch {epoch})", color=color
        )
        lstm_smooth_ax.fill_between(
            range(len(mu_sequence)),
            mu_sequence - 2 * var_sequence,
            mu_sequence + 2 * var_sequence,
            alpha=0.3,
            color=color,
            label=f"LSTM smoothed std (epoch {epoch})",
        )
        lstm_smooth_ax.set_title("LSTM smoothed states")
        lstm_smooth_fig.canvas.draw()
        plt.pause(0.01)

    # Early-stopping
    model.early_stopping(evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch)
    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
        states_optim = copy.copy(
            model.states
        )  # If we want to plot the states, plot those from optimal epoch
        model_optim_dict = model.get_dict()

    if model.stop_training:
        break
    else:
        # reset memory
        model.set_memory(states=model.states, time_step=0)

    fig, ax = plot_states(
        data_processor=data_processor,
        states=model.states,
        states_type="smooth",
    )
    filename = f"saved_results/smoother#{epoch}.png"
    plt.savefig(filename)
    plt.close()

# set memory and parameters to optimal epoch
model.load_dict(model_optim_dict)
model.lstm_net.set_lstm_states(lstm_optim_states)
model.set_memory(
    states=states_optim,
    time_step=data_processor.test_start,
)

# forecat on the test set
mu_test_preds, std_test_preds, test_states = model.forecast(
    data=test_data,
)

# Unstandardize the predictions
mu_test_preds = normalizer.unstandardize(
    mu_test_preds,
    data_processor.std_const_mean[output_col],
    data_processor.std_const_std[output_col],
)
std_test_preds = normalizer.unstandardize_std(
    std_test_preds,
    data_processor.std_const_std[output_col],
)

# calculate the test metrics
test_obs = data_processor.get_data("test").flatten()
mse = metric.mse(mu_test_preds, test_obs)
log_lik = metric.log_likelihood(mu_test_preds, test_obs, std_test_preds)

print(f"Test MSE            :{mse: 0.4f}")
print(f"Test Log-Lik        :{log_lik: 0.2f}")

# plot the test data
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
    validation_label=[r"$\mu$", r"$\pm\sigma$"],
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
