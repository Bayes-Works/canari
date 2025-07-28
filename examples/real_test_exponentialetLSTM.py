import copy
import pandas as pd
from pytagi import Normalizer as normalizer
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pytagi.metric as metric
from statsmodels.tsa.seasonal import seasonal_decompose
from canari import (
    DataProcess,
    Model,
    SKF,
    plot_states,
    plot_data,
    plot_prediction,
    plot_skf_states,
)

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
    }
)

from canari.component import (
    Exponential,
    WhiteNoise,
    Periodic,
    LocalTrend,
    Autoregression,
    LocalLevel,
    LstmNetwork,
)

## Read data
data_file = "/Users/michelwu/Desktop/Exponential component/2650F162.CSV"
df_raw = pd.read_csv(
    data_file,
    sep=";",  # Semicolon as delimiter
    quotechar='"',  # Double quotes as text qualifier
    engine="python",  # Python engine for complex cases
    na_values=[""],  # Treat empty strings as NaN
    skipinitialspace=True,  # Skip spaces after delimiter
    encoding="ISO-8859-1",
    parse_dates=["Date"],
    index_col="Date",
)

df = df_raw[["Deplacements cumulatif X (mm)"]]
df = df.iloc[:]
df = df.resample("M").mean()

output_col = [0]
num_epoch = 50

data_processor = DataProcess(
    data=df,
    time_covariates=["month_of_year"],
    train_split=0.3,
    validation_split=0.1,
    output_col=output_col,
)
data_processor.scale_const_mean = np.array([0.86892857, 6.5])
data_processor.scale_const_std = np.array([1.20594462, 3.45205253])
print(data_processor.scale_const_mean)
print(data_processor.scale_const_std)


train_data, validation_data, test_data, all_data = data_processor.get_splits()

# Model
localtrend = LocalTrend(
    mu_states=[2.75, -0.0018], var_states=[0.2**2, 0.0001**2], std_error=0
)
periodic = Periodic(period=12, mu_states=[0, 0], var_states=[2, 2], std_error=0)
periodic2 = Periodic(period=6, mu_states=[0, 0], var_states=[0.5, 0.5], std_error=0)
sigma_v = 0.75
noise = WhiteNoise(std_error=sigma_v)
AR_process_error_var_prior = 0.5
var_W2bar_prior = 1
AR_process_error_var_prior = 1
var_W2bar_prior = 1

ar = Autoregression(
    mu_states=[-0.1, 0.75, 0, 0, 0, AR_process_error_var_prior],
    var_states=[
        6.36e-05,
        10e-8,
        0,
        AR_process_error_var_prior,
        1e-6,
        var_W2bar_prior,
    ],
)
exponential = Exponential(
    # std_error=0.0,
    mu_states=[0, 0.00265, 10.5, 0, 0],
    var_states=[0.1**2, 0.0001**2, 0.5**2, 0, 0],
)
model = Model(
    localtrend,
    noise,
    exponential,
    LstmNetwork(
        look_back_len=12,
        num_features=2,
        num_layer=1,
        num_hidden_unit=40,
        device="cpu",
        manual_seed=1,
    ),
)

# Training
for epoch in range(num_epoch):
    (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        train_data=train_data,
        validation_data=validation_data,
    )
    model.set_memory(states=states, time_step=0)
    # print(model.states.get_mean("level", "smooth"))
    # print(model.states.get_mean("trend", "smooth"))
    # print(model.states.get_std("level", "smooth"))
    # print(model.states.get_std("trend", "smooth"))

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

    # Calculte the log-likelihood metric
    validation_obs = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_validation_preds, validation_obs)

    # Early-stopping
    model.early_stopping(evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch)
    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
        states_optim = copy.copy(states)
        model_optim_dict = model.get_dict()
        lstm_optim_states = model.lstm_net.get_lstm_states()

    if model.stop_training:
        break

print(f"Optimal epoch : {model.optimal_epoch}")
print(f"Validation MSE : {model.early_stop_metric: 0.4f}")

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
    data_processor.scale_const_mean[output_col],
    data_processor.scale_const_std[output_col],
)
std_test_preds = normalizer.unstandardize_std(
    std_test_preds,
    data_processor.scale_const_std[output_col],
)

# calculate the test metrics
test_obs = data_processor.get_data("test").flatten()
mse = metric.mse(mu_test_preds, test_obs)
log_lik = metric.log_likelihood(mu_test_preds, test_obs, std_test_preds)

print(f"Test MSE            :{mse: 0.4f}")
print(f"Test Log-Lik        :{log_lik: 0.2f}")

# plot the test data
fig, ax = plt.subplots(figsize=(6, 3))
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
plt.scatter(
    data_processor.get_time("all"), data_processor.get_data("all"), color="red", s=2.5
)

plt.legend(loc=(0.1, 1.01), ncol=6, fontsize=12)
plt.tight_layout()
plt.savefig(f"ltsm_exp.pgf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()

fig, ax = plot_states(
    data_processor=data_processor,
    states=model.states,
    states_to_plot=(
        "latent level",
        "latent trend",
        "scale",
        # "exp",
        # "scaled exp",
        # "periodic 1",
        # "autoregression",
        # "phi",
        # "W2bar",
        "white noise",
        "level",
        "trend",
        "lstm",
    ),
    states_type="smooth",
    standardization=True,
)

# plot_data(
#     data_processor=data_processor,
#     plot_column=output_col,
#     standardization=True,
#     plot_test_data=False,
#     validation_label="y",
#     # sub_plot=ax[model.get_states_index("scaled exp")],
#     sub_plot=ax[0],
# )
# ax[model.get_states_index("scaled exp")].plot(
#     df.index,
#     model.states.get_mean("scaled exp", "smooth")
#     + model.states.get_mean("level", "smooth"),
#     color="purple",
# )

# ax[model.get_states_index("scaled exp")].scatter(
ax[3].scatter(
    data_processor.get_time("all"), data_processor.get_data("all"), color="red", s=2.5
)
plt.savefig(f"ltsm_exp_smooth.pgf", bbox_inches="tight", pad_inches=0, transparent=True)
plt.show()

plt.plot(
    data_processor.get_time("all"),
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth")
    # + model.states.get_mean("periodic 1", "smooth")
    + model.states.get_mean("lstm", "smooth"),
    color="purple",
)
scaled_exp_index = model.get_states_index("scaled exp")
level_index = model.get_states_index("level")
periodic_index = model.get_states_index("periodic 1")

cov_scaled_exp_level = []
cov_level_periodic = []
cov_scaled_exp_periodic = []

for i in range(len(model.states.get_mean("level", "smooth"))):
    cov_scaled_exp_level.append(
        model.states.var_smooth[i][scaled_exp_index, level_index]
    )
    cov_level_periodic.append(model.states.var_smooth[i][periodic_index, level_index])
    cov_scaled_exp_periodic.append(
        model.states.var_smooth[i][scaled_exp_index, periodic_index]
    )

cov_level_periodic = np.array(cov_level_periodic)
cov_scaled_exp_level = np.array(cov_scaled_exp_level)
cov_scaled_exp_periodic = np.array(cov_scaled_exp_periodic)


plt.fill_between(
    data_processor.get_time("all"),
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth")
    # + model.states.get_mean("periodic 1", "smooth")
    + np.sqrt(
        model.states.get_std("scaled exp", "smooth") ** 2
        + model.states.get_std("level", "smooth") ** 2
        # + model.states.get_std("periodic 1", "smooth") ** 2
        + 2 * (cov_scaled_exp_level + cov_level_periodic + cov_scaled_exp_periodic)
    ),
    model.states.get_mean("scaled exp", "smooth")
    + model.states.get_mean("level", "smooth")
    # + model.states.get_mean("periodic 1", "smooth")
    - np.sqrt(
        model.states.get_std("scaled exp", "smooth") ** 2
        + model.states.get_std("level", "smooth") ** 2
        # + model.states.get_std("periodic 1", "smooth") ** 2
        + 2 * (cov_scaled_exp_level + cov_level_periodic + cov_scaled_exp_periodic)
    ),
    color="purple",
    alpha=0.2,
)


# print(model.states.var_smooth[2])
# print(model.states.get_mean("scaled exp", "smooth").dtype)

plt.scatter(
    data_processor.get_time("all"), data_processor.get_data("all"), color="red", s=2.5
)
plt.show()
