import fire
import copy
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytagi import exponential_scheduler
from pytagi import metric
from pytagi import Normalizer as normalizer
from src import (
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    WhiteNoise,
    Autoregression,
    Model,
    ModelOptimizer,
    SKF,
    SKFOptimizer,
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from examples import DataProcess
import pickle


# Fix parameters grid search
SKF_std_transition_error_fix = 1e-4
SKF_norm_to_abnorm_prob_fix = 1e-4


def main(
    num_trial_optimization: int = 20,
    param_tune: bool = True,
    grid_search: bool = False,
):
    # Read data
    data_file = "./data/toy_time_series/synthetic_simple_autoregression_periodic.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

    anm_start_index = 52*10

    # LT anomaly
    # anm_mag = 0.010416667/10
    anm_mag = 0.3/52
    # anm_baseline = np.linspace(0, 3, num=len(df_raw))
    anm_baseline = np.arange(len(df_raw)) * anm_mag
    # Set the first 52*12 values in anm_baseline to be 0
    anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
    anm_baseline[:anm_start_index] = 0

    df_raw = df_raw.add(anm_baseline, axis=0)

    data_file_time = "./data/toy_time_series/synthetic_simple_autoregression_periodic_datetime.csv"
    time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(time_series[0])
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["values"]

    # Data pre-processing
    output_col = [0]
    train_split=0.289
    validation_split=0.0693*2

    # Remove the last 52*5 rows in df_raw
    train_split = train_split * len(df_raw) / len(df_raw[:-52*5])
    validation_split = validation_split * len(df_raw) / len(df_raw[:-52*5])
    df_raw = df_raw[:-52*5]

    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df_raw,
        time_covariates=["week_of_year"],
        train_split=train_split,
        validation_split=validation_split,
        output_col=output_col,
    )
    (
        data_processor.train_data,
        data_processor.validation_data,
        data_processor.test_data,
        data_processor.all_data,
    ) = data_processor.get_splits()

    # Load model_dict from local
    with open("saved_params/toy_simple_model.pkl", "rb") as f:
        model_dict = pickle.load(f)

    LSTM = LstmNetwork(
        look_back_len=19,
        num_features=2,
        num_layer=1,
        num_hidden_unit=50,
        device="cpu",
    )

    # # Define model
    # def initialize_model(param):

    #     print("phi_AR =", model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item())
    #     print("sigma_AR =", np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()))

    #     model = Model(
    #         LocalTrend(mu_states=model_dict['early_stop_init_mu_states'][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
    #         LSTM,
    #         Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()), 
    #                phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(), 
    #                mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()], 
    #                var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
    #     )
    #     model.lstm_net.load_state_dict(model_dict["lstm_network_params"])
    #     return model

    # # Define parameter search space
    # if param_tune:
    #     param = {
    #         "look_back_len": [10, 30],
    #         "sigma_v": [1e-3, 2e-1],
    #     }
    #     # Define optimizer
    #     model_optimizer = ModelOptimizer(
    #         initialize_model=initialize_model,
    #         train=training,
    #         param_space=param,
    #         data_processor=data_processor,
    #         num_optimization_trial=num_trial_optimization,
    #     )
    #     model_optimizer.optimize()
    #     # Get best model
    #     model_optim = model_optimizer.get_best_model()
    # else:

    # param = {
    #     "look_back_len": look_back_len_fix,
    #     "sigma_v": sigma_v_fix,
    # }
    # model_optim = initialize_model(param)

    # # Train best model
    # model_optim, states_optim, mu_validation_preds, std_validation_preds = training(
    #     model=model_optim, data_processor=data_processor
    # )

    # # Save best model for SKF analysis later
    # model_optim_dict = model_optim.get_dict()

    # # Plot
    # fig, ax = plt.subplots(figsize=(10, 6))
    # plot_data(
    #     data_processor=data_processor,
    #     normalization=True,
    #     plot_test_data=False,
    #     plot_column=output_col,
    #     test_label="y",
    # )
    # plot_prediction(
    #     data_processor=data_processor,
    #     mean_validation_pred=mu_validation_preds,
    #     std_validation_pred=std_validation_preds,
    #     validation_label=[r"$\mu$", f"$\pm\sigma$"],
    # )
    # plot_states(
    #     data_processor=data_processor,
    #     states=states_optim,
    #     normalization=True,
    #     states_to_plot=["local level"],
    #     sub_plot=ax,
    # )
    # plt.legend()
    # plt.title("Validation predictions")
    # plt.show()

    norm_model = Model(
        LocalTrend(mu_states=model_dict['early_stop_init_mu_states'][0:2].reshape(-1), var_states=[1e-12, 1e-12]),
        LSTM,
        Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()), 
                phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(), 
                mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()], 
                var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
    )
    norm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

    # Define SKF model
    def initialize_skf(skf_param):

        abnorm_model = Model(
            LocalAcceleration(mu_states=[model_dict['early_stop_init_mu_states'][0].item(), model_dict['early_stop_init_mu_states'][1].item(), 0], var_states=[1e-12, 1e-12, 1e-4]),
            LSTM,
            Autoregression(std_error=np.sqrt(model_dict['states_optimal'].mu_prior[-1][model_dict['W2bar_index']].item()),
                   phi=model_dict['states_optimal'].mu_prior[-1][model_dict['phi_index']].item(),
                   mu_states=[model_dict['early_stop_init_mu_states'][model_dict['autoregression_index']].item()],
                   var_states=[model_dict['early_stop_init_var_states'][model_dict['autoregression_index'], model_dict['autoregression_index']].item()]),
        )
        abnorm_model.lstm_net.load_state_dict(model_dict["lstm_network_params"])

        skf = SKF(
            norm_model=norm_model,
            abnorm_model=abnorm_model,
            std_transition_error=skf_param["std_transition_error"],
            norm_to_abnorm_prob=skf_param["norm_to_abnorm_prob"],
            abnorm_to_norm_prob=1e-1,
            norm_model_prior_prob=0.99,
        )
        return skf
    
    model_optim_dict = norm_model.get_dict()

    # Define parameter search space
    slope_upper_bound = 5e-2
    slope_lower_bound = 1e-3

    # Plot synthetic anomaly
    synthetic_anomaly_data = DataProcess.add_synthetic_anomaly(
        data_processor.train_data,
        num_samples=1,
        slope=[slope_lower_bound, slope_upper_bound],
    )
    plot_data(
        data_processor=data_processor,
        normalization=True,
        plot_validation_data=False,
        plot_test_data=False,
        plot_column=output_col,
    )
    train_time = data_processor.get_time("train")
    for ts in synthetic_anomaly_data:
        plt.plot(train_time, ts["y"])
    plt.legend(
        [
            "data without anomaly",
            "",
            "smallest anomaly tested",
            "largest anomaly tested",
        ]
    )
    plt.title("Train data with added synthetic anomalies")
    plt.show()

    if param_tune:
        if grid_search:
            skf_param = {
                "std_transition_error": [1e-5, 1e-4, 1e-3],
                "norm_to_abnorm_prob": [1e-5, 1e-4, 1e-3],
                "slope": [0.006, 0.008, 0.01, 0.02],
            }
        else:
            skf_param = {
                "std_transition_error": [1e-6, 1e-2],
                "norm_to_abnorm_prob": [1e-6, 1e-2],
                "slope": [slope_lower_bound, slope_upper_bound],
            }
        # Define optimizer
        skf_optimizer = SKFOptimizer(
            initialize_skf=initialize_skf,
            model_param=model_optim_dict,
            param_space=skf_param,
            data=data_processor.train_data,
            num_synthetic_anomaly=50,
            num_optimization_trial=num_trial_optimization * 2,
            grid_search=grid_search,
        )
        skf_optimizer.optimize()
        # Get best model
        skf_optim = skf_optimizer.get_best_model()
    else:
        skf_param = {
            "std_transition_error": SKF_std_transition_error_fix,
            "norm_to_abnorm_prob": SKF_norm_to_abnorm_prob_fix,
        }
        skf_optim = initialize_skf(skf_param)

    # Detect anomaly
    filter_marginal_abnorm_prob, states = skf_optim.filter(data=data_processor.all_data)

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        states_to_plot=["local level", "local trend", "lstm", "autoregression"],
        model_prob=filter_marginal_abnorm_prob,
        normalization=False,
        color="b",
        legend_location="upper left",
    )
    ax[0].axvline(
        x=data_processor.data.index[anm_start_index],
        color="r",
        linestyle="--",
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.show()

    if param_tune:
        print("SKF model parameters used:", skf_optimizer.param_optim)
    else:
        # print("Model parameters used:", param)
        print("SKF model parameters used:", skf_param)
    print("-----")


def training(model, data_processor, num_epoch: int = 50):
    """ """
    # index_start = 0
    # index_end = 24 * 3 + 1
    # y1 = data_processor.train_data["y"][:-1].flatten()
    # trend, _, seasonality, _ = DataProcess.decompose_data(y1)
    # t_plot = data_processor.data.index[index_start:index_end].to_numpy()
    # plt.plot(t_plot, trend, color="b")
    # plt.plot(t_plot, seasonality, color="orange")
    # plt.scatter(t_plot, y1, color="k")
    # plt.plot(
    #     data_processor.get_time("train"),
    #     data_processor.get_data("train", normalization=True),
    #     color="r",
    # )
    # plt.show()

    model.auto_initialize_baseline_states(data_processor.train_data["y"][0:24])
    states_optim = None
    mu_validation_preds_optim = None
    std_validation_preds_optim = None

    for epoch in range(num_epoch):
        mu_validation_preds, std_validation_preds, states = model.lstm_train(
            train_data=data_processor.train_data,
            validation_data=data_processor.validation_data,
        )

        mu_validation_preds_unnorm = normalizer.unstandardize(
            mu_validation_preds,
            data_processor.norm_const_mean[data_processor.output_col],
            data_processor.norm_const_std[data_processor.output_col],
        )

        std_validation_preds_unnorm = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor.norm_const_std[data_processor.output_col],
        )

        validation_obs = data_processor.get_data("validation").flatten()
        validation_log_lik = metric.log_likelihood(
            prediction=mu_validation_preds_unnorm,
            observation=validation_obs,
            std=std_validation_preds_unnorm,
        )

        model.early_stopping(evaluate_metric=-validation_log_lik, mode="min")

        if epoch == model.optimal_epoch:
            mu_validation_preds_optim = mu_validation_preds.copy()
            std_validation_preds_optim = std_validation_preds.copy()
            states_optim = copy.copy(states)
        if model.stop_training:
            break

    return (
        model,
        states_optim,
        mu_validation_preds_optim,
        std_validation_preds_optim,
    )


if __name__ == "__main__":
    fire.Fire(main)
