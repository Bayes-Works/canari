import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytagi.metric as metric
import shap
from pytagi import Normalizer as normalizer

from canari import DataProcess, Model, plot_data, plot_prediction
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
shap_explain_size = 50
shap_num_samples = 1000
output_dir = Path("./saved_results/shap_lstm")
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
    if epoch == model.optimal_epoch:
        mu_validation_preds_optim = mu_validation_preds
        std_validation_preds_optim = std_validation_preds
        states_optim = copy.copy(states)

    if model.stop_training:
        break

print(f"Optimal epoch       : {model.optimal_epoch}")
print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

# Set memory and parameters to optimal epoch
model.set_memory(
    time_step=data_processor.test_start - 1,
)

# Forecast on the test set
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

# Calculate the test metrics
test_obs = data_processor.get_data("test").flatten()
mse = metric.mse(mu_test_preds, test_obs)
log_lik = metric.log_likelihood(mu_test_preds, test_obs, std_test_preds)

print(f"Test MSE            :{mse: 0.4f}")
print(f"Test Log-Lik        :{log_lik: 0.2f}")

# Plot the test data
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
plt.savefig(output_dir / "forecast.png", dpi=200)
plt.close(fig)

# SHAP explainability
model.set_memory(
    time_step=data_processor.test_start - 1,
)
shap_memory = model.get_memory()
feature_names = train_data["cov_names"]
rng = np.random.default_rng(1)

background_index = rng.choice(
    len(train_data["x"]),
    size=min(shap_background_size, len(train_data["x"])),
    replace=False,
)
explain_index = rng.choice(
    len(test_data["x"]),
    size=min(shap_explain_size, len(test_data["x"])),
    replace=False,
)
background_data = train_data["x"][np.sort(background_index)]
explain_data = test_data["x"][np.sort(explain_index)]


def predict_diplacement_y(covariates):
    current_memory = model.get_memory()
    predictions = []
    try:
        model.lstm_net.eval()
        for covariate in np.asarray(covariates):
            model.set_memory(memory=copy.deepcopy(shap_memory))
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


explainer = shap.KernelExplainer(predict_diplacement_y, background_data)
shap_values = explainer.shap_values(explain_data, nsamples=shap_num_samples)
if isinstance(shap_values, list):
    shap_values = shap_values[0]
shap_values = np.asarray(shap_values)

expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)):
    expected_value = np.asarray(expected_value).flatten()[0]

shap_explanation = shap.Explanation(
    values=shap_values,
    base_values=np.repeat(expected_value, len(explain_data)),
    data=explain_data,
    feature_names=feature_names,
)

# Save feature-level SHAP plots
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, explain_data, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(output_dir / "shap_summary.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    explain_data,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
)
plt.tight_layout()
plt.savefig(output_dir / "shap_feature_importance.png", dpi=200, bbox_inches="tight")
plt.close()

# Save additional SHAP plots
plt.figure(figsize=(10, 6))
shap.plots.waterfall(
    shap_explanation[0],
    max_display=len(feature_names),
    show=False,
)
plt.tight_layout()
plt.savefig(output_dir / "shap_waterfall_first_sample.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
shap.plots.heatmap(
    shap_explanation,
    max_display=len(feature_names),
    show=False,
)
plt.tight_layout()
plt.savefig(output_dir / "shap_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
shap.decision_plot(
    expected_value,
    shap_values,
    explain_data,
    feature_names=feature_names,
    show=False,
)
plt.tight_layout()
plt.savefig(output_dir / "shap_decision.png", dpi=200, bbox_inches="tight")
plt.close()

dependence_dir = output_dir / "shap_dependence"
dependence_dir.mkdir(parents=True, exist_ok=True)
for feature_name in feature_names:
    safe_feature_name = feature_name.replace(" ", "_").replace("/", "_").lower()
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        feature_name,
        shap_values,
        explain_data,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        dependence_dir / f"{safe_feature_name}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

# Save grouped SHAP importance so time covariates are read as one explanatory block
group_names = [
    "time_covariates" if name in time_covariates else name for name in feature_names
]
shap_importance = pd.DataFrame(
    {
        "feature_group": group_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }
)
shap_importance = (
    shap_importance.groupby("feature_group", as_index=False)["mean_abs_shap"]
    .sum()
    .sort_values("mean_abs_shap", ascending=False)
)
shap_importance.to_csv(output_dir / "shap_group_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(shap_importance["feature_group"], shap_importance["mean_abs_shap"])
ax.invert_yaxis()
ax.set_xlabel("mean(|SHAP value|)")
plt.tight_layout()
plt.savefig(output_dir / "shap_group_importance.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Forecast plot        : {output_dir / 'forecast.png'}")
print(f"SHAP summary plot    : {output_dir / 'shap_summary.png'}")
print(f"SHAP importance plot : {output_dir / 'shap_feature_importance.png'}")
print(f"SHAP waterfall plot  : {output_dir / 'shap_waterfall_first_sample.png'}")
print(f"SHAP heatmap         : {output_dir / 'shap_heatmap.png'}")
print(f"SHAP decision plot   : {output_dir / 'shap_decision.png'}")
print(f"SHAP dependence plots: {dependence_dir}")
print(f"SHAP group importance: {output_dir / 'shap_group_importance.csv'}")
