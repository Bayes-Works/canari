import fire
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ray import tune
from pytagi import metric
from pytagi import Normalizer as normalizer
from canari import (
    DataProcess,
    Model,
    Optimizer,
    SKF,
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from canari.component import LocalTrend, LocalAcceleration, KernelRegression, WhiteNoise

def main(
    param_optim: bool = True,
):
    ######### Data processing #########
    # Read data
    data_file = "/Users/vuongdai/Desktop/backup_canari/LTU0014/LTU014PIAEVA920.DAT"
    df_raw = pd.read_csv(data_file,
                     sep=";",  # Semicolon as delimiter
                     quotechar='"',
                     engine="python",
                     na_values=[""],  # Treat empty strings as NaN
                     skipinitialspace=True,
                     encoding="ISO-8859-1",
                     )
    df = df_raw[["Deplacements cumulatif X (mm)"]]
    df.columns = ["ext"]
    df.index = pd.to_datetime(df_raw["Date"])
    df = df.resample("D").last()

    data_processor = DataProcess(
        data=df,
        train_split=0.2,
        validation_split=0.08,
        test_split=0.67,
        output_col=[0],
        standardization=False,
    )
    train_data, validation_data, test_data, all_data = data_processor.get_splits()

    period = 365
    num_kernel = 10
    np.random.seed(42)
    mu_cp = np.random.uniform(low=-3, high=3, size=num_kernel)
    # mu_cp = 1
    var_cp = 2
    ######### Define model with parameters #########
    def model_with_parameters(param):
        model = Model(
            LocalTrend(),
            KernelRegression(period=period,
                            kernel_length=param["kernel_length"],
                            num_control_point=num_kernel,
                            mu_control_point = mu_cp,
                            var_control_point = var_cp,
                            ),
            WhiteNoise(std_error=param["sigma_v"])
        )

        model.auto_initialize_baseline_states(train_data["y"])
        mu_preds, std_preds,_ = model.filter(data=train_data)

        obs = data_processor.get_data("train").flatten()
        log_lik = metric.log_likelihood(
            prediction=mu_preds,
            observation=obs,
            std=std_preds,
        )
        mse = metric.mse(mu_preds, obs)
        model.metric_optim = -log_lik

        # mu_preds, std_preds,_ = model.forecast(data=validation_data)

        # mu_preds = normalizer.unstandardize(
        #     mu_preds,
        #     data_processor.scale_const_mean[data_processor.output_col],
        #     data_processor.scale_const_std[data_processor.output_col],
        # )

        # std_preds = normalizer.unstandardize_std(
        #     std_preds,
        #     data_processor.scale_const_std[data_processor.output_col],
        # )
        
        # obs = data_processor.get_data("validation").flatten()
        # log_lik = metric.log_likelihood(
        #     prediction=mu_preds,
        #     observation=obs,
        #     std=std_preds,
        # )
        # mse = metric.mse(mu_preds, obs)
        # model.metric_optim = mse

        return model

    ######### Define SKF model with parameters #########
    def skf_with_parameters(skf_param_space, param):
        norm_model = Model(
            LocalTrend(),
            KernelRegression(period=period,
                            kernel_length=param["kernel_length"],
                            num_control_point=num_kernel,
                            mu_control_point = mu_cp,
                            var_control_point = var_cp,
                            ),
            WhiteNoise(std_error=param["sigma_v"])
        )
        norm_model.auto_initialize_baseline_states(train_data["y"])

        abnorm_model = Model(
            LocalAcceleration(),
            KernelRegression(period=period,
                            kernel_length=param["kernel_length"],
                            num_control_point=num_kernel,
                            mu_control_point = mu_cp,
                            var_control_point = var_cp,
                            ),
            WhiteNoise(std_error=param["sigma_v"])
        )

        skf = SKF(
            norm_model=norm_model,
            abnorm_model=abnorm_model,
            std_transition_error=skf_param_space["std_transition_error"],
            norm_to_abnorm_prob=skf_param_space["norm_to_abnorm_prob"],
        )
        skf.save_initial_states()

        skf.filter(data=all_data)
        log_lik_all = np.nanmean(skf.ll_history)
        skf.metric_optim = -log_lik_all

        skf.load_initial_states()

        return skf

    ######### Parameter optimization #########
    if param_optim:
        # Define parameter search space
        # param_space = {
        #     "kernel_length": [0.1, 0.99],
        #     "sigma_v": [1e-1, 5e-1],
        #     # "std_error": [1e-1, 5e-1],
        #     # "std_error_cp": [1e-5, 1e-3],
        # }
        # # Define optimizer
        # model_optimizer = Optimizer(
        #     model=model_with_parameters,
        #     param=param_space,
        #     num_optimization_trial=50,
        #     mode="min",
        #     num_startup_trials=20,
        # )
        # model_optimizer.optimize()
        # # Get best model
        # param = model_optimizer.get_best_param()
        # model_with_parameters(param)

        param = {
            "kernel_length": 0.95,
            "sigma_v": 3e-1,
            # "std_error": 0.3,
            # "std_error_cp": 1e-3
        }
        # Train best model
        model_with_parameters(param)

        # # Optimize for skf
        # skf_param_space = {
        #     "std_transition_error": [1e-6, 1e-4],
        #     "norm_to_abnorm_prob": [1e-6, 1e-4],
        # }
        # skf_optimizer = Optimizer(
        #     model=skf_with_parameters,
        #     param=skf_param_space,
        #     model_input=param,
        #     num_optimization_trial=30,
        #     num_startup_trials=10,
        #     mode="min",
        # )
        # skf_optimizer.optimize()
        # # Get parameters
        # skf_param = skf_optimizer.get_best_param()

        skf_param={
            "std_transition_error": 1e-4,
            "norm_to_abnorm_prob": 1e-5,
        }
        skf_optim = skf_with_parameters(skf_param, param)
        skf_optim_dict = skf_optim.get_dict()
        skf_optim_dict["model_param"] = param
        skf_optim_dict["skf_param"] = skf_param
        skf_optim_dict["cov_names"] = train_data["cov_names"]
        with open("saved_params/ltu14_1.pkl", "wb") as f:
            pickle.dump(skf_optim_dict, f)
    else:
        # # Load saved skf model
        with open("saved_params/ltu14_1.pkl", "rb") as f:
            skf_optim_dict = pickle.load(f)
        skf_optim = SKF.load_dict(skf_optim_dict)

    ######### Detect anomaly #########
    print("Model parameters used:", skf_optim_dict["model_param"])
    print("SKF model parameters used:", skf_optim_dict["skf_param"])

    skf_optim.auto_initialize_baseline_states(train_data["y"])
    filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        model_prob=filter_marginal_abnorm_prob,
        states_to_plot=["level", "trend", "kernel regression", "white noise"],
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
