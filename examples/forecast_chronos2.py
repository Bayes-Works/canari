"""
Forecast with the `Auxiliary` component wrapping Amazon's Chronos-2 foundation
model. The external predictor supplies the one-step-ahead (mean, variance) that
would otherwise come from the Bayesian LSTM.

Requires: `pip install chronos-forecasting torch`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from chronos import BaseChronosPipeline

from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.data_visualization import plot_with_uncertainty
from canari.component import Auxiliary, LocalTrend, WhiteNoise


# ---- Data: synthetic AR + periodic (more complex than the sine toy) ----
data_file = "./data/toy_time_series/synthetic_autoregression_periodic.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
time_series = pd.to_datetime(
    pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)[0]
)
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["values"]
# Keep the run tractable on CPU — Chronos-2 is called at every step.
df = df_raw.iloc[:300]

output_col = [0]
data_processor = DataProcess(
    data=df,
    train_split=0.8,
    validation_split=0.1,
    output_col=output_col,
)
train_data, validation_data, test_data, all_data = data_processor.get_splits()


# ---- Chronos-2 predictor ----
LOOK_BACK = 52  # matches the series' periodicity
NUM_FEATURES = 0  # no covariates

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
    dtype=torch.float32,
)


def chronos_predict(mu_input: np.ndarray, var_input: np.ndarray):
    context = torch.tensor(mu_input[:LOOK_BACK], dtype=torch.float32).reshape(1, 1, -1)
    
    # Request quantiles corresponding to -1 std, median, and +1 std
    quantiles, _ = pipeline.predict_quantiles(
        inputs=context,
        prediction_length=1,
        quantile_levels=[0.1587, 0.5, 0.8413], 
    )
    
    # q is now [q_minus_1std, q_median, q_plus_1std]
    q = quantiles[0][0, 0].cpu().numpy()  
    
    mu = np.array([q[1]])
    
    # The distance between +1 std and -1 std is exactly 2 std
    std = (q[2] - q[0]) / 2.0 
    var = np.array([max(std**2, 1e-6)])
    
    return mu, var


# ---- Model ----
sigma_v = 0.2
model = Model(
    LocalTrend(),
    Auxiliary(
        predict_fn=chronos_predict,
        std_error=0.0,
        look_back_len=LOOK_BACK,
        num_features=NUM_FEATURES,
    ),
    WhiteNoise(std_error=sigma_v),
)

model.auto_initialize_baseline_states(train_data["y"][0:52])

# Warm the aux rolling history with true training targets so Chronos sees real
# context from the start, instead of zeros.
warm = np.asarray(train_data["y"][:LOOK_BACK], dtype=float).reshape(-1)
model.lstm_output_history.mu = warm.copy()
model.lstm_output_history.var = np.zeros_like(warm)

# ---- Single filter over the full series (Chronos is frozen) ----
# Doing it in one pass keeps `model.states` populated for the full timeline, so
# we can plot prior/posterior trajectories everywhere.
mu_all, std_all, states = model.filter(data=all_data, train_lstm=False)

# Slice per split for the prediction metrics.
val_start = data_processor.validation_start
test_start = data_processor.test_start

mu_val = normalizer.unstandardize(
    mu_all[val_start:test_start],
    data_processor.scale_const_mean[output_col],
    data_processor.scale_const_std[output_col],
)
std_val = normalizer.unstandardize_std(
    std_all[val_start:test_start], data_processor.scale_const_std[output_col]
)
mu_test = normalizer.unstandardize(
    mu_all[test_start:],
    data_processor.scale_const_mean[output_col],
    data_processor.scale_const_std[output_col],
)
std_test = normalizer.unstandardize_std(
    std_all[test_start:], data_processor.scale_const_std[output_col]
)

val_obs = data_processor.get_data("validation").flatten()
test_obs = data_processor.get_data("test").flatten()
print(f"Validation MSE      :{metric.mse(mu_val, val_obs): 0.4f}")
print(f"Test MSE            :{metric.mse(mu_test, test_obs): 0.4f}")
print(
    f"Test Log-Lik        :{metric.log_likelihood(mu_test, test_obs, std_test): 0.2f}"
)

# ---- Plot: observations + one-step-ahead predictions ----
fig, ax = plt.subplots(figsize=(10, 6))
plot_data(
    data_processor=data_processor,
    standardization=False,
    plot_column=output_col,
    validation_label="y",
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_val,
    std_validation_pred=std_val,
    validation_label=[r"$\mu$", r"$\pm\sigma$"],
)
plot_prediction(
    data_processor=data_processor,
    mean_test_pred=mu_test,
    std_test_pred=std_test,
    test_label=[r"$\mu^{\prime}$", r"$\pm\sigma^{\prime}$"],
    color="purple",
)
plt.legend(loc=(0.1, 1.01), ncol=6, fontsize=12)
plt.tight_layout()
plt.savefig("forecast_chronos2.png", dpi=120)

# ---- Plot: hidden states over the full series, prior + posterior overlaid ----
fig2, axes2 = plot_states(
    data_processor=data_processor,
    states=states,
    states_type="posterior",
    standardization=False,
    color="b",
    legend_location="upper left",
)

# Overlay the prior on each subplot.
for ax, state_name in zip(axes2, states.states_name):
    t_axis = ax.lines[0].get_xdata()
    scale_mean = (
        data_processor.scale_const_mean[output_col] if state_name == "level" else 0
    )
    scale_std = data_processor.scale_const_std[output_col]
    mu_prior = states.get_mean(
        states_name=state_name,
        states_type="prior",
        standardization=False,
        scale_const_mean=scale_mean,
        scale_const_std=scale_std,
    )
    std_prior = states.get_std(
        states_name=state_name,
        states_type="prior",
        standardization=False,
        scale_const_std=scale_std,
    )
    plot_with_uncertainty(
        time=t_axis,
        mu=mu_prior,
        std=std_prior,
        color="orange",
        linestyle="--",
        ax=ax,
    )

axes2[0].legend(
    [r"$\mu_{post}$", r"$\pm\sigma_{post}$", r"$\mu_{prior}$", r"$\pm\sigma_{prior}$"],
    loc="upper left",
    fontsize=8,
    ncol=2,
)
fig2.suptitle("Hidden states — prior vs posterior (full series)", fontsize=10, y=1)
plt.tight_layout()
plt.savefig("forecast_chronos2_states.png", dpi=120)

plt.show()
