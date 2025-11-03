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
    ModelOptimizer,
    SKFOptimizer,
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise


# # Read data
data_file = "/Users/vuongdai/Desktop/20251107_canari_tutorial_HQ/data/LGA001EFAPRG910.DAT"
df_raw = pd.read_csv(data_file,
                    sep=";",  # Semicolon as delimiter
                    quotechar='"',
                    engine="python",
                    na_values=[""],  # Treat empty strings as NaN
                    skipinitialspace=True,
                    encoding="ISO-8859-1",
                    )
df = df_raw[["Ext/Contraction (mm)"]]
df.columns = ["values"]
df.index = pd.to_datetime(df_raw["Date"])

df =df.resample("W").mean()
mask = df["values"].isna()

# Detrending data
df_detrend = df.copy()
df_detrend = df_detrend.interpolate()
decomposition = seasonal_decompose(df_detrend["values"], model="additive", period=52)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
residual = residual.fillna(0)
df_detrend["values"] = seasonal + residual + np.mean(trend)
df_detrend.loc[mask,"values"] = np.nan
# decomposition.plot()
# plt.plot(seasonal + residual)
# fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=False)
# axs[0].plot(df["values"])
# axs[1].plot(df_detrend["values"])
# axs[2].plot(seasonal)
# axs[3].plot(residual)
# plt.show()

# # Data pre-processing
output_col = [0]
data_processor_detrend = DataProcess(
    data=df_detrend,
    time_covariates=["week_of_year"],
    train_split=0.6,
    validation_split=0.1,
    output_col=output_col,
)
train_data, validation_data, _, _ = data_processor_detrend.get_splits()

data_processor_ = DataProcess(
    data=df,
    time_covariates=["week_of_year"],
    train_split=0.18,
    validation_split=0.1,
    output_col=output_col,
    scale_const_mean=data_processor_detrend.scale_const_mean,
    scale_const_std=data_processor_detrend.scale_const_std
)
train_data_, _, _, _ = data_processor_.get_splits()

data_processor = DataProcess(
    data=df,
    time_covariates=["week_of_year"],
    train_split=0.6,
    validation_split=0.1,
    output_col=output_col,
    scale_const_mean=data_processor_detrend.scale_const_mean,
    scale_const_std=data_processor_detrend.scale_const_std
)
train_data_original, _, _, all_data = data_processor.get_splits()

plot_data(
     data_processor=data_processor_,
     plot_test_data=False,
     standardization=True,
)
plt.show()

######### Define model with parameters #########
def model_with_parameters(param, train_data, validation_data):
    model = Model(
        LocalTrend(),
        LstmNetwork(
            look_back_len=int(param["look_back_len"]),
            num_features=2,
            num_layer=1,
            infer_len=52 * 3,
            num_hidden_unit=50,
            manual_seed=1,
        ),
        WhiteNoise(std_error=param["sigma_v"]),
    )

    model.auto_initialize_baseline_states(train_data["y"][0 : 52 * 3])
    mu_validation_preds_optim = None
    std_validation_preds_optim = None
    num_epoch = 50
    for epoch in range(num_epoch):
        mu_validation_preds, std_validation_preds, states = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
        )

        mu_validation_preds_unnorm = normalizer.unstandardize(
            mu_validation_preds,
            data_processor_detrend.scale_const_mean[data_processor_detrend.output_col],
            data_processor_detrend.scale_const_std[data_processor_detrend.output_col],
        )

        std_validation_preds_unnorm = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor_detrend.scale_const_std[data_processor_detrend.output_col],
        )

        validation_obs = data_processor_detrend.get_data("validation").flatten()
        validation_log_lik = metric.log_likelihood(
            prediction=mu_validation_preds_unnorm,
            observation=validation_obs,
            std=std_validation_preds_unnorm,
        )

        model.early_stopping(
            evaluate_metric=-validation_log_lik,
            current_epoch=epoch,
            max_epoch=num_epoch,
        )
        model.metric_optim = model.early_stop_metric

        if epoch == model.optimal_epoch:
            mu_validation_preds_optim = mu_validation_preds.copy()
            std_validation_preds_optim = std_validation_preds.copy()

        if model.stop_training:
            break

    return (
        model,
        mu_validation_preds_optim,
        std_validation_preds_optim,
    )

# Define optimizer

param= {
    "look_back_len": 52,
    "sigma_v": 0.1,
}

# Train best model
model_optim, mu_validation_preds, std_validation_preds = model_with_parameters(
    param, train_data, validation_data
)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
plot_data(
    data_processor=data_processor_detrend,
    standardization=True,
    plot_test_data=False,
    plot_column=output_col,
    validation_label="y",
)
plot_prediction(
    data_processor=data_processor_detrend,
    mean_validation_pred=mu_validation_preds,
    std_validation_pred=std_validation_preds,
    validation_label=["mean", "std"],
)
plot_states(
    data_processor=data_processor_detrend,
    states=model_optim.states,
    standardization=True,
    states_to_plot=["level"],
    sub_plot=ax,
)
plt.legend()
plt.title("Validation predictions")
plt.show()

# Save best model for SKF analysis later
model_optim_dict = model_optim.get_dict(time_step=0)

def skf_with_parameters(skf_param_space, model_param: dict, train_data):
        norm_model = Model.load_dict(model_param)
        norm_model.auto_initialize_baseline_states(all_data["y"][0 : 52 * 3])

        abnorm_model = Model(
            LocalAcceleration(),
            LstmNetwork(),
            WhiteNoise(),
        )
        skf = SKF(
            norm_model=norm_model,
            abnorm_model=abnorm_model,
            std_transition_error=skf_param_space["std_transition_error"],
            norm_to_abnorm_prob=skf_param_space["norm_to_abnorm_prob"],
        )
        skf.save_initial_states()

        # detection_rate, false_rate, false_alarm_train = skf.detect_synthetic_anomaly(
        #     data=train_data,
        #     num_anomaly=50,
        #     slope_anomaly=skf_param_space["slope"],
        # )
        # skf.metric_optim["detection_rate"] = detection_rate
        # skf.metric_optim["false_rate"] = false_rate
        # skf.metric_optim["false_alarm_train"] = false_alarm_train

        return skf

skf_param = {
    "std_transition_error": 1e-5,
    "norm_to_abnorm_prob": 1e-6,
    "slope": 5e-3,
}

skf_optim = skf_with_parameters(skf_param, model_optim_dict, train_data_)

skf_optim_dict = skf_optim.get_dict()
skf_optim_dict["model_param"] = param
skf_optim_dict["skf_param"] = skf_param
skf_optim_dict["cov_names"] = train_data["cov_names"]
print("Model parameters used:", skf_optim_dict["model_param"])
print("SKF model parameters used:", skf_optim_dict["skf_param"])


synthetic_anomaly_data = DataProcess.add_synthetic_anomaly(
    train_data_,
    num_samples=1,
    slope=[0],
)

filter_marginal_abnorm_prob, states = skf_optim.filter(data=synthetic_anomaly_data[0])
smooth_marginal_abnorm_prob, states = skf_optim.smoother()

fig, ax = plot_skf_states(
    data_processor=data_processor_,
    states=states,
    model_prob=filter_marginal_abnorm_prob,
    standardization=True,
)
fig.suptitle("SKF hidden states", fontsize=10, y=1)
plt.savefig("./saved_results/BM4.png")
plt.show()
