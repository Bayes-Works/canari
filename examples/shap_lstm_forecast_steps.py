import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytagi.metric as metric
import shap
from pytagi import Normalizer as normalizer
from tqdm import tqdm

from canari import DataProcess, Model
from canari.component import LstmNetwork, WhiteNoise


# Read data
data_file = "./data/benchmark_data/test_5_data.csv"
df_raw = pd.read_csv(data_file)

time_col = "date_time"
if time_col not in df_raw.columns:
    time_col = "Date" if "Date" in df_raw.columns else "date"

time_series = pd.to_datetime(df_raw[time_col])
df_raw.index = time_series
df_raw.index.name = "date_time"

target_col = "diplacement_y"
covariate_cols = ["water_level", "temp_min", "temp_max"]
df_raw = df_raw[[target_col] + covariate_cols]
df_raw = df_raw.apply(pd.to_numeric, errors="coerce")
df_raw[covariate_cols] = df_raw[covariate_cols].interpolate(
    method="time", limit_direction="both"
)
df_raw = df_raw.dropna()

# Define parameters
output_col = [0]
num_epoch = 50
time_covariates = ["hour_of_day", "day_of_week", "day_of_year", "month_of_year"]
shap_background_size = 20
shap_num_samples = 200
num_forecast_steps = None
output_dir = Path("./saved_results/shap_lstm_forecast_steps")
output_dir.mkdir(parents=True, exist_ok=True)

# Build data processor
data_processor = DataProcess(
    data=df_raw,
    time_covariates=time_covariates,
    train_split=0.7,
    validation_split=0.15,
    output_col=output_col,
)

# Split data
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()
if num_forecast_steps is None:
    num_forecast_steps = len(test_data["x"])
num_forecast_steps = min(num_forecast_steps, len(test_data["x"]))

# Model
sigma_v = 0.003
model = Model(
    LstmNetwork(
        look_back_len=12,
        num_features=train_data["x"].shape[1] + 1,
        infer_len=24,
        num_layer=1,
        num_hidden_unit=40,
        device="cpu",
        manual_seed=1,
        smoother=False,
    ),
    WhiteNoise(std_error=sigma_v),
)

# Training
for epoch in range(num_epoch):
    mu_validation_preds, std_validation_preds, states = model.lstm_train(
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

    # Calculate the validation metric
    validation_obs = data_processor.get_data("validation").flatten()
    mse = metric.mse(mu_validation_preds, validation_obs)

    # Early-stopping
    model.early_stopping(evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch)
    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

# SHAP background data
feature_names = train_data["cov_names"]
rng = np.random.default_rng(1)
background_index = rng.choice(
    len(train_data["x"]),
    size=min(shap_background_size, len(train_data["x"])),
    replace=False,
)
background_data = train_data["x"][np.sort(background_index)]

# Forecast one step at a time and explain each predicted step
model.set_memory(
    time_step=data_processor.test_start - 1,
)
model.initialize_states_history()
model.lstm_net.eval()

mu_test_preds = []
std_test_preds = []
shap_values_by_step = []
expected_values = []

for index, (x, time) in tqdm(
    list(enumerate(zip(test_data["x"], test_data["time"])))[:num_forecast_steps],
    desc="Forecast SHAP",
    unit="step",
):
    step_memory = copy.deepcopy(model.get_memory())

    def predict_step(covariates):
        current_memory = model.get_memory()
        predictions = []
        try:
            model.lstm_net.eval()
            for covariate in np.asarray(covariates):
                model.set_memory(memory=copy.deepcopy(step_memory))
                mu_pred, _, _, _ = model.forward(np.asarray(covariate, dtype=np.float32))
                mu_pred = normalizer.unstandardize(
                    mu_pred.flatten(),
                    data_processor.scale_const_mean[output_col],
                    data_processor.scale_const_std[output_col],
                )
                predictions.append(mu_pred.item())
        finally:
            model.set_memory(memory=current_memory)

        return np.array(predictions)

    explainer = shap.KernelExplainer(predict_step, background_data)
    shap_values = explainer.shap_values(
        x.reshape(1, -1),
        nsamples=shap_num_samples,
    )
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values_by_step.append(np.asarray(shap_values).reshape(-1))

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.asarray(expected_value).flatten()[0]
    expected_values.append(expected_value)

    # Advance the forecast state using the true covariates for this forecast step.
    model.set_memory(memory=copy.deepcopy(step_memory))
    mu_obs_pred, var_obs_pred, mu_states_prior, var_states_prior = model.forward(x)

    model.update_lstm_states_history(index, last_step=num_forecast_steps - 1)
    model.update_lstm_output_history(mu_states_prior, var_states_prior)
    model._set_posterior_states(mu_states_prior, var_states_prior)
    model.save_states_history()
    model.set_states(mu_states_prior, var_states_prior)

    mu_test_preds.append(mu_obs_pred.item())
    std_test_preds.append(var_obs_pred.item() ** 0.5)

mu_test_preds = normalizer.unstandardize(
    np.array(mu_test_preds),
    data_processor.scale_const_mean[output_col],
    data_processor.scale_const_std[output_col],
)
std_test_preds = normalizer.unstandardize_std(
    np.array(std_test_preds),
    data_processor.scale_const_std[output_col],
)
shap_values_by_step = np.array(shap_values_by_step)

# Calculate the test metrics over the explained horizon
test_time = test_data["time"][:num_forecast_steps]
test_obs = data_processor.get_data("test").flatten()[:num_forecast_steps]
mse = metric.mse(mu_test_preds, test_obs)
log_lik = metric.log_likelihood(mu_test_preds, test_obs, std_test_preds)

print(f"Test MSE            :{mse: 0.4f}")
print(f"Test Log-Lik        :{log_lik: 0.2f}")

# Group SHAP values into the physical inputs plus one time-covariates block
group_names = [
    "time_covariates" if name in time_covariates else name for name in feature_names
]
unique_groups = list(dict.fromkeys(group_names))
group_shap_values = np.zeros((len(shap_values_by_step), len(unique_groups)))
group_abs_shap_values = np.zeros_like(group_shap_values)

for group_index, group_name in enumerate(unique_groups):
    feature_index = [i for i, name in enumerate(group_names) if name == group_name]
    group_shap_values[:, group_index] = shap_values_by_step[:, feature_index].sum(axis=1)
    group_abs_shap_values[:, group_index] = np.abs(
        shap_values_by_step[:, feature_index]
    ).sum(axis=1)

top_group_index = np.argmax(group_abs_shap_values, axis=1)
top_group = [unique_groups[i] for i in top_group_index]
top_group_signed_shap = group_shap_values[np.arange(len(top_group_index)), top_group_index]
top_group_abs_shap = group_abs_shap_values[
    np.arange(len(top_group_index)), top_group_index
]

# Save step-level SHAP values and the dominant input at each predicted step
results = pd.DataFrame(
    {
        "time": test_time,
        "observed": test_obs,
        "predicted_mean": mu_test_preds,
        "predicted_std": std_test_preds,
        "top_input": top_group,
        "top_input_signed_shap": top_group_signed_shap,
        "top_input_abs_shap": top_group_abs_shap,
        "expected_value": expected_values,
    }
)
for feature_index, feature_name in enumerate(feature_names):
    results[f"shap_{feature_name}"] = shap_values_by_step[:, feature_index]
for group_index, group_name in enumerate(unique_groups):
    results[f"group_shap_{group_name}"] = group_shap_values[:, group_index]
    results[f"group_abs_shap_{group_name}"] = group_abs_shap_values[:, group_index]
results.to_csv(output_dir / "forecast_step_shap_values.csv", index=False)

# Composite plot: forecast, inputs, SHAP over time, and top input per step
plot_time = pd.DatetimeIndex(test_time)
input_data = data_processor.data.loc[test_time, covariate_cols]
input_std = input_data.std(ddof=0).replace(0, 1)
input_data_norm = (input_data - input_data.mean()) / input_std

palette = plt.get_cmap("tab10").colors
group_colors = {
    group_name: palette[group_index % len(palette)]
    for group_index, group_name in enumerate(unique_groups)
}

fig, ax = plt.subplots(
    4,
    1,
    figsize=(15, 12),
    sharex=True,
    gridspec_kw={"height_ratios": [2.0, 1.25, 1.6, 1.0]},
    constrained_layout=True,
)

ax[0].plot(plot_time, test_obs, color="black", linewidth=1.8, label="observed")
ax[0].plot(plot_time, mu_test_preds, color="purple", linewidth=1.8, label="forecast")
ax[0].fill_between(
    plot_time,
    mu_test_preds - std_test_preds,
    mu_test_preds + std_test_preds,
    color="purple",
    alpha=0.18,
    label="+/- 1 std",
)
ax[0].set_ylabel(target_col)
ax[0].legend(loc="upper right", frameon=True)
ax[0].set_title("Forecast and step-wise SHAP explanations")

for column in input_data_norm.columns:
    ax[1].plot(plot_time, input_data_norm[column], linewidth=1.4, label=column)
ax[1].axhline(0, color="black", linewidth=0.8, alpha=0.4)
ax[1].set_ylabel("standardized\ninput")
ax[1].legend(loc="upper right", ncol=len(covariate_cols), frameon=True)

for group_index, group_name in enumerate(unique_groups):
    ax[2].plot(
        plot_time,
        group_shap_values[:, group_index],
        color=group_colors[group_name],
        linewidth=1.5,
        label=group_name,
    )
ax[2].axhline(0, color="black", linewidth=0.8, alpha=0.4)
ax[2].set_ylabel("signed\nSHAP")
ax[2].legend(loc="upper right", ncol=min(4, len(unique_groups)), frameon=True)

if len(plot_time) > 1:
    bar_width = 0.8 * np.median(np.diff(plot_time.asi8)) / 1e9 / 86400
else:
    bar_width = 0.8

top_bar_colors = [group_colors[group_name] for group_name in top_group]
ax[3].bar(
    plot_time,
    top_group_abs_shap,
    width=bar_width,
    color=top_bar_colors,
    edgecolor="none",
    alpha=0.9,
)
ax[3].set_ylabel("top\n|SHAP|")
ax[3].set_xlabel("forecast time")
legend_handles = [
    plt.Line2D([0], [0], color=group_colors[group_name], lw=6, label=group_name)
    for group_name in unique_groups
]
ax[3].legend(
    handles=legend_handles,
    loc="upper right",
    ncol=min(4, len(unique_groups)),
    frameon=True,
)

for subplot in ax:
    subplot.grid(True, axis="x", alpha=0.18)
    subplot.margins(x=0)

for label in ax[-1].get_xticklabels():
    label.set_rotation(35)
    label.set_ha("right")
plt.savefig(output_dir / "forecast_inputs_step_shap.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Heatmap of grouped SHAP magnitudes over forecast time
fig, ax = plt.subplots(figsize=(14, 4))
heatmap = ax.imshow(group_abs_shap_values.T, aspect="auto", cmap="magma")
ax.set_yticks(np.arange(len(unique_groups)))
ax.set_yticklabels(unique_groups)
tick_index = np.linspace(0, num_forecast_steps - 1, min(8, num_forecast_steps), dtype=int)
ax.set_xticks(tick_index)
ax.set_xticklabels(pd.Index(test_time[tick_index]).strftime("%Y-%m-%d"), rotation=45)
ax.set_xlabel("forecast time")
ax.set_title("Step-wise input impact magnitude")
fig.colorbar(heatmap, ax=ax, label="|SHAP|")
plt.tight_layout()
plt.savefig(output_dir / "forecast_step_shap_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Feature-level SHAP heatmap, including individual time covariates
fig, ax = plt.subplots(figsize=(14, 5))
feature_heatmap = ax.imshow(np.abs(shap_values_by_step).T, aspect="auto", cmap="magma")
ax.set_yticks(np.arange(len(feature_names)))
ax.set_yticklabels(feature_names)
ax.set_xticks(tick_index)
ax.set_xticklabels(pd.Index(test_time[tick_index]).strftime("%Y-%m-%d"), rotation=45)
ax.set_xlabel("forecast time")
ax.set_title("Feature-level step-wise SHAP magnitude")
fig.colorbar(feature_heatmap, ax=ax, label="|SHAP|")
plt.tight_layout()
plt.savefig(
    output_dir / "forecast_step_feature_shap_heatmap.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

print(f"Step SHAP values CSV : {output_dir / 'forecast_step_shap_values.csv'}")
print(f"Composite plot       : {output_dir / 'forecast_inputs_step_shap.png'}")
print(f"Grouped SHAP heatmap : {output_dir / 'forecast_step_shap_heatmap.png'}")
print(
    "Feature SHAP heatmap : "
    f"{output_dir / 'forecast_step_feature_shap_heatmap.png'}"
)
