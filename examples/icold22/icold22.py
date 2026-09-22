import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from ray import tune
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import (
    DataProcess,
    Model,
    ModelAssemble,
    Optimizer,
    plot_data,
    plot_prediction,
    plot_states,
)
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

# # Read data
data_file = "/Users/vuongdai/Desktop/backup_canari/data/icold_2022.csv"
df_raw = pd.read_csv(data_file, delimiter=",", header=0,
                     names=["datetime","cb2","cb3","waterlevel","temp_a","temp_b"],
                     usecols=["datetime","cb3","waterlevel"],
                     index_col="datetime", parse_dates=True)

# Resample
df_raw = df_raw.loc['2000-01-30':'2012-12-31']
df = df_raw.resample("W").last()

# Data pre-processing
output_col = [0]
covar_col = [1]

# 
NUM_EPOCH = 50
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
OUTPUT_COL = [0] # Target

# seed
rng = np.random.default_rng(1)
seed = rng.integers(0, 1000)        

# 
def build_data(covar_num_lag):
    df_lagged = DataProcess.add_lagged_columns(df, [0, covar_num_lag])
    data_processor = DataProcess(
        data=df_lagged,
        time_covariates=["week_of_year"],
        train_split=TRAIN_SPLIT,
        validation_split=VAL_SPLIT,
        output_col=output_col,
    )
    num_features = covar_num_lag + 3

    return data_processor, num_features

# 
def model_with_parameters(param):
    (
        data_processor,
        num_features,
    ) = build_data(param["covar_num_lag"])

    train_data, validation_data, test_data, _ = data_processor.get_splits()

    # Target model
    model_target = Model(
        LstmNetwork(
            # look_back_len=param["target_look_back_len"],
            look_back_len=1,
            num_features=num_features,
            num_layer=1,
            num_hidden_unit=50,
            manual_seed=seed,
            smoother=False,
        ),
        WhiteNoise(std_error=param["target_sigma_v"]),
    )

    # Covariate model
    model_covar = Model(
        LstmNetwork(
            look_back_len=param["covar_look_back_len"],
            num_features=2,
            num_layer=1,
            num_hidden_unit=50,
            manual_seed=seed,
            smoother=False,
        ),
        WhiteNoise(std_error=param["covar_sigma_v"]),
    )
    model_covar.output_col = covar_col.copy()
    model_covar.input_col = [4]

    # Assemble models
    model = ModelAssemble(target_model=model_target, covariate_model=model_covar)

    # 
    validation_obs = data_processor.get_data("validation").flatten()

    mu_validation_preds_optim = None
    std_validation_preds_optim = None
    mu_covar_validation_preds_optim = None
    std_covar_validation_preds_optim = None
    mu_test_preds_optim = None
    std_test_preds_optim = None
    mu_covar_test_preds_optim = None
    std_covar_test_preds_optim = None
    num_test = len(test_data["y"])
    num_val = len(validation_data["y"])
    
    for epoch in range(NUM_EPOCH):
        (mu_validation_preds, std_validation_preds) = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
        )

        # Validation
        # Tartget predictions
        mu_validation_preds = normalizer.unstandardize(
            mu_validation_preds,
            data_processor.scale_const_mean[output_col],
            data_processor.scale_const_std[output_col],
        )
        std_validation_preds = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor.scale_const_std[output_col],
        )

        mse = metric.mse(mu_validation_preds, validation_obs)
        validation_log_lik = metric.log_likelihood(
            prediction=mu_validation_preds,
            observation=validation_obs,
            std=std_validation_preds,
        )
        
        # Covariate predictions
        mu_covar_validation_preds = np.array(
            model.covariate_model[0].output_history.mu[-num_val:]
        )
        std_covar_validation_preds = (
            np.array(model.covariate_model[0].output_history.var[-num_val:])
            + model.covariate_model[0].sched_sigma_v**2
        ) ** 0.5
        mu_covar_validation_preds = normalizer.unstandardize(
            mu_covar_validation_preds,
            data_processor.scale_const_mean[covar_col],
            data_processor.scale_const_std[covar_col],
        )
        std_covar_validation_preds = normalizer.unstandardize_std(
            std_covar_validation_preds,
            data_processor.scale_const_std[covar_col],
        )

        # Test set
        model.covariate_model[0].set_memory(time_step=data_processor.test_start - 1)
        model.target_model.set_memory(time_step=data_processor.test_start - 1)

        # Tartget predictions
        mu_test_preds, std_test_preds = model.forecast(data=test_data)
        
        mu_test_preds = normalizer.unstandardize(
            mu_test_preds,
            data_processor.scale_const_mean[output_col],
            data_processor.scale_const_std[output_col],
        )
        std_test_preds = normalizer.unstandardize_std(
            std_test_preds,
            data_processor.scale_const_std[output_col],
        )

        # Covariate predictions
        mu_covar_test_preds = np.array(
            model.covariate_model[0].output_history.mu[-num_test:]
        )
        std_covar_test_preds = (
            np.array(model.covariate_model[0].output_history.var[-num_test:]) + model.covariate_model[0].sched_sigma_v**2
        ) ** 0.5
        mu_covar_test_preds = normalizer.unstandardize(
            mu_covar_test_preds,
            data_processor.scale_const_mean[covar_col],
            data_processor.scale_const_std[covar_col],
        )
        std_covar_test_preds = normalizer.unstandardize_std(
            std_covar_test_preds,
            data_processor.scale_const_std[covar_col],
        )

        # Early-stopping on the validation set
        model.target_model.early_stopping(
            evaluate_metric=mse,
            current_epoch=epoch,
            max_epoch=NUM_EPOCH,
            # skip_epoch=5,
        )

        model.covariate_model[0].set_memory(time_step=0)
        model.target_model.set_memory(time_step=0)

        # Metric used by the Optimizer (minimized)
        model.metric_optim = model.target_model.early_stop_metric

        if epoch == model.target_model.optimal_epoch:
            mu_validation_preds_optim = mu_validation_preds
            std_validation_preds_optim = std_validation_preds
            mu_covar_validation_preds_optim = mu_covar_validation_preds
            std_covar_validation_preds_optim = std_covar_validation_preds
            mu_test_preds_optim = mu_test_preds
            std_test_preds_optim = std_test_preds
            mu_covar_test_preds_optim = mu_covar_test_preds
            std_covar_test_preds_optim = std_covar_test_preds

        
        if model.target_model.stop_training:
            break

    return (
        data_processor,
        mu_validation_preds_optim,
        std_validation_preds_optim,
        mu_covar_validation_preds_optim,
        std_covar_validation_preds_optim,
        mu_test_preds_optim,
        std_test_preds_optim,
        mu_covar_test_preds_optim,
        std_covar_test_preds_optim,
        )


# # Hyperparameter optimization with TPE sampling
# param_space = {
#     "target_sigma_v": tune.loguniform(1e-2, 2e-1),
#     "covar_look_back_len": [26, 62],
#     "covar_sigma_v": tune.loguniform(1e-2, 2e-1),
#     "covar_num_lag": [0, 26], 
# }

# model_optimizer = Optimizer(
#     model=model_with_parameters,
#     param=param_space,
#     num_optimization_trial=120,
#     num_startup_trials=60,
#     mode="min",
#     max_concurrent=6,
# )
# model_optimizer.optimize()


param_optim = {
    'target_sigma_v': 0.011071429111750946,
    'covar_look_back_len': 14,
    'covar_sigma_v': 0.03821106587480968,
    'covar_num_lag': 2,
}
(
    data_processor,
    mu_validation_preds_optim,
    std_validation_preds_optim,
    mu_covar_validation_preds_optim,
    std_covar_validation_preds_optim,
    mu_test_preds_optim,
    std_test_preds_optim,
    mu_covar_test_preds_optim,
    std_covar_test_preds_optim,
) = model_with_parameters(param_optim)


print(f"Seed            : {seed}")
print(f"Optimal parameters            : {param_optim}")

#  Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Target model
plot_data(
    data_processor=data_processor,
    standardization=False,
    plot_column=output_col,
    sub_plot=axs[0],
    plot_nan=False,
    validation_label="obs.",
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_validation_preds_optim,
    std_validation_pred=std_validation_preds_optim,
    sub_plot=axs[0],
    validation_label=["pred.", "$\\pm\\sigma$"],
    num_std = 2,
)
plot_prediction(
    data_processor=data_processor,
    mean_test_pred=mu_test_preds_optim,
    std_test_pred=std_test_preds_optim,
    sub_plot=axs[0],
    test_label=["test pred.", "$\\pm\\sigma$"],
    color="purple",
    num_std = 2,
)
axs[0].set_title("Target model")
axs[0].legend()

# Covariate model
plot_data(
    data_processor=data_processor,
    standardization=False,
    plot_column=covar_col,
    sub_plot=axs[1],
    plot_nan=False,
    validation_label="obs.",
)
plot_prediction(
    data_processor=data_processor,
    mean_validation_pred=mu_covar_validation_preds_optim,
    std_validation_pred=std_covar_validation_preds_optim,
    sub_plot=axs[1],
    validation_label=["pred.", "$\\pm\\sigma$"],
    num_std = 2,
)
plot_prediction(
    data_processor=data_processor,
    mean_test_pred=mu_covar_test_preds_optim,
    std_test_pred=std_covar_test_preds_optim,
    sub_plot=axs[1],
    test_label=["test pred.", "$\\pm\\sigma$"],
    color="purple",
    num_std = 2,
)
axs[1].set_title("Covariate model")
axs[1].legend()

plt.tight_layout()
plt.savefig("saved_results/icold22.png", dpi=300, bbox_inches="tight")
