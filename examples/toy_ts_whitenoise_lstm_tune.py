import fire
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytagi import metric
from pytagi import Normalizer
from canari import (
    DataProcess,
    Model,
    ModelOptimizer,
    SKF,
    SKFOptimizer,
    plot_data,
    plot_prediction,
    plot_skf_states,
    plot_states,
)
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise, Autoregression

sigma_v_fix = None
look_back_len_fix = None

true_sigma_AR = 0.1
true_phi_AR = 0.7
stationary_std_AR = true_sigma_AR / np.sqrt(1 - true_phi_AR**2)


def main(
    num_trial_optimization: int = 50,
    param_optimization: bool = True,
):
    #  Read data
    data_file = "./data/toy_time_series/simple_syn_ar_std01_phi07_periodic.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

    data_file_time = "./data/toy_time_series/synthetic_autoregression_periodic_datetime.csv"
    time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(time_series[0])
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["values"]

    # Data processor initialization
    output_col = [0]
    data_processor = DataProcess(
        data=df_raw,
        time_covariates=["week_of_year"],
        train_split=0.6,
        validation_split=0.2,
        output_col=output_col,
        # standardization=False,
    )
    data_processor.scale_const_mean = np.array([0, 2.6068333e+01])
    data_processor.scale_const_std = np.array([1, 15.090957])

    train_data, validation_data, test_data, all_data = data_processor.get_splits()

    ########################################
    ########################################

    # Parameter optimization for model
    num_epoch = 100

    def initialize_model(param, train_data, validation_data):
        model = Model(
            LocalTrend(mu_states=[0, 0], std_error=0),
            LstmNetwork(
                look_back_len=param["look_back_len"],
                num_features=2,
                num_layer=1,
                num_hidden_unit=50,
                manual_seed=1,
            ),
            WhiteNoise(std_error=param["sigma_v"]),
        )
        model._mu_local_level = 0
        # model.auto_initialize_baseline_states(train_data["y"][0:104])

        states_optim = None
        mu_validation_preds_optim = None
        std_validation_preds_optim = None
        # Training
        for epoch in range(num_epoch):
            (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
                train_data=train_data,
                validation_data=validation_data,
            )

            # Unstandardize the predictions
            mu_validation_preds_unnorm = Normalizer.unstandardize(
                mu_validation_preds,
                data_processor.scale_const_mean[data_processor.output_col],
                data_processor.scale_const_std[data_processor.output_col],
            )

            std_validation_preds_unnorm = Normalizer.unstandardize_std(
                std_validation_preds,
                data_processor.scale_const_std[data_processor.output_col],
            )

            validation_obs = data_processor.get_data("validation").flatten()
            validation_log_lik = metric.log_likelihood(
                prediction=mu_validation_preds_unnorm,
                observation=validation_obs,
                std=std_validation_preds_unnorm,
            )

            model.early_stopping(
                evaluate_metric=-validation_log_lik,
                current_epoch=epoch,
                max_epoch=num_epoch,
                # skip_epoch = 50,
            )
            model.metric_optim = model.early_stop_metric

            if epoch == model.optimal_epoch:
                mu_validation_preds_optim = mu_validation_preds.copy()
                std_validation_preds_optim = std_validation_preds.copy()
                states_optim = copy.copy(states)

            model.set_memory(states=states, time_step=0)
            if model.stop_training:
                break

        return (
            model,
            states_optim,
            mu_validation_preds_optim,
            std_validation_preds_optim,
        )

    # Define parameter search space
    if param_optimization:
        param_space = {
            "look_back_len": [10, 30],
            "sigma_v": [1e-3, 2e-1],
        }
        # Define optimizer
        model_optimizer = ModelOptimizer(
            model=initialize_model,
            param_space=param_space,
            train_data=train_data,
            validation_data=validation_data,
            num_optimization_trial=num_trial_optimization,
        )
        model_optimizer.optimize()
        # Get best model
        param = model_optimizer.get_best_param()
    else:
        param = {
            "look_back_len": look_back_len_fix,
            "sigma_v": sigma_v_fix,
        }

    # Train best model
    print("Model parameters used:", param)
    model_optim, states_optim, mu_validation_preds, std_validation_preds = (
        initialize_model(param, train_data, validation_data)
    )
    print('----------------------------------------------')
    print("Stationary std AR =", stationary_std_AR)

    # Save best model for SKF analysis later
    model_optim_dict = model_optim.get_dict()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data(
        data_processor=data_processor,
        standardization=True,
        plot_test_data=False,
        plot_column=output_col,
        test_label="y",
    )
    plot_prediction(
        data_processor=data_processor,
        mean_validation_pred=mu_validation_preds,
        std_validation_pred=std_validation_preds,
        validation_label=[r"$\mu$", f"$\pm\sigma$"],
    )
    plot_states(
        data_processor=data_processor,
        states=states_optim,
        standardization=True,
        states_to_plot=["level"],
        sub_plot=ax,
    )
    plt.legend()
    plt.title("Validation predictions")
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)