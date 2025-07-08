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
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise
import pickle

sigma_v_fix = None
look_back_len_fix = None
SKF_std_transition_error_fix = None
SKF_norm_to_abnorm_prob_fix = None


def main(
    num_trial_optimization: int = 20,
    param_optimization: bool = True,
):
    # Read data
    data_file = "./data/benchmark_data/detrended_data/test_10_data_detrended.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(df_raw.iloc[:, 0])
    df_raw = df_raw.iloc[:, 1:]
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["values"]

    # Add synthetic anomaly to data
    # LT anomaly
    # anm_mag = 0.010416667/10
    anm_start_index = 52*8
    anm_mag = 0.1/52
    # anm_baseline = np.linspace(0, 3, num=len(df_raw))
    anm_baseline = np.arange(len(df_raw)) * anm_mag
    # Set the first 52*12 values in anm_baseline to be 0
    anm_baseline[anm_start_index:] -= anm_baseline[anm_start_index]
    anm_baseline[:anm_start_index] = 0
    df_raw = df_raw.add(anm_baseline, axis=0)

    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df_raw,
        time_covariates=["week_of_year"],
        train_split=0.3,
        validation_split=0.1,
        output_col=output_col,
    )
    train_data, validation_data, test_data, all_data = data_processor.get_splits()

    ########################################
    ########################################

    # Parameter optimization for model
    num_epoch = 50

    def initialize_model(param, train_data, validation_data):
        model = Model(
            LocalTrend(),
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
        # model.auto_initialize_baseline_states(train_data["y"][0:24])

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

    ########################################
    ########################################

    # Parameter optimization for SKF
    def initialize_skf(skf_param_space, model_param: dict):
        norm_model = Model.load_dict(model_param)
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
        return skf

    # Define parameter search space
    slope_upper_bound = 5e-2
    slope_lower_bound = 1e-3

    # Plot synthetic anomaly
    synthetic_anomaly_data = DataProcess.add_synthetic_anomaly(
        train_data,
        num_samples=1,
        slope=[slope_lower_bound, slope_upper_bound],
    )
    plot_data(
        data_processor=data_processor,
        standardization=True,
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

    if param_optimization:
        skf_param_space = {
            "std_transition_error": [1e-6, 1e-2],
            "norm_to_abnorm_prob": [1e-6, 1e-2],
            "slope": [slope_lower_bound, slope_upper_bound],
            "threshold_anm_prob": [1e-2, 1.],
        }
        # Define optimizer
        skf_optimizer = SKFOptimizer(
            initialize_skf=initialize_skf,
            model_param=model_optim_dict,
            param_space=skf_param_space,
            data=train_data,
            num_synthetic_anomaly=50,
            num_optimization_trial=num_trial_optimization * 2,
        )
        skf_optimizer.optimize()

        # Get parameters
        skf_param = skf_optimizer.get_best_param()
    else:
        skf_param = {
            "std_transition_error": SKF_std_transition_error_fix,
            "norm_to_abnorm_prob": SKF_norm_to_abnorm_prob_fix,
        }

    print("SKF model parameters used:", skf_param)
    skf_optim = initialize_skf(skf_param, model_optim_dict)

    # Save the dictionaries
    with open('saved_params/ts10_whitenoise_models/skf_param10.pkl', 'wb') as f:
        pickle.dump(skf_param, f)

    with open('saved_params/ts10_whitenoise_models/model_optim_dict10.pkl', 'wb') as f:
        pickle.dump(model_optim_dict, f)

    # Detect anomaly
    filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)
    filter_marginal_abnorm_prob, states = skf_optim.smoother()

    # Plotting SKF results
    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        states_type="smooth",
        states_to_plot=["level", "trend", "lstm", "white noise"],
        model_prob=filter_marginal_abnorm_prob,
        standardization=False,
    )
    ax[0].axvline(
        x=data_processor.data.index[anm_start_index],
        color="r",
        linestyle="--",
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)