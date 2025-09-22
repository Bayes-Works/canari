import copy
import fire
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
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


def main(
    num_trial_optimization: int = 100,
    param_optimization: bool = False,
    param_grid_search: bool = False,
):
    ########################################
    # Read data & data processing
    data_file = "./data/benchmark_data/test_1_data.csv"
    df = pd.read_csv(data_file, skiprows=0, delimiter=",")
    date_time = pd.to_datetime(df["timestamp"])
    df = df.drop("timestamp", axis=1)
    df.index = date_time
    df.index.name = "date_time"
    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df,
        time_covariates=["week_of_year"],
        train_start="2011-02-06 00:00:00",
        train_end="2014-02-02 00:00:00",
        validation_start="2014-02-09 00:00:00",
        validation_end="2015-02-01 00:00:00",
        test_start="2015-02-08 00:00:00",
        output_col=output_col,
    )
    train_data, validation_data, test_data, all_data = data_processor.get_splits()

    ########################################
    # Parameter optimization
    if param_optimization or param_grid_search:
        ##################
        # Optimize for model
        def initialize_model(param, train_data, validation_data):
            model = Model(
                LocalTrend(),
                LstmNetwork(
                    look_back_len=param["look_back_len"],
                    num_features=2,
                    num_layer=1,
                    infer_len=52 * 3,
                    num_hidden_unit=50,
                    device="cpu",
                    manual_seed=1,
                ),
                WhiteNoise(std_error=param["sigma_v"]),
            )

            # index_start = 0
            # index_end = 52 * 3
            # y1 = data_processor.train_data["y"][index_start:index_end].flatten()
            # trend, _, seasonality, _ = DataProcess.decompose_data(y1)
            # t_plot = data_processor.data.index[index_start:index_end].to_numpy()
            # plt.plot(t_plot, trend, color="b")
            # plt.plot(t_plot, seasonality, color="orange")
            # plt.scatter(t_plot, y1, color="k")
            # plt.plot(
            #     data_processor.get_time("train"),
            #     data_processor.get_data("train", standardization=True),
            #     color="r",
            # )
            # plt.show()

            model.auto_initialize_baseline_states(train_data["y"][0 : 52 * 3])
            states_optim = None
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
                    data_processor.scale_const_mean[data_processor.output_col],
                    data_processor.scale_const_std[data_processor.output_col],
                )

                std_validation_preds_unnorm = normalizer.unstandardize_std(
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
                "look_back_len": [12, 52],
                "sigma_v": [1e-1, 4e-1],
            }
        elif param_grid_search:
            param_space = {
                "look_back_len": [12, 26, 52],
                "sigma_v": [1e-1, 2e-1, 3e-1, 4e-1],
            }
        # Define optimizer
        model_optimizer = ModelOptimizer(
            model=initialize_model,
            param_space=param_space,
            train_data=train_data,
            validation_data=validation_data,
            num_optimization_trial=num_trial_optimization,
            grid_search=param_grid_search,
        )
        model_optimizer.optimize()
        # Get best model
        param = model_optimizer.get_best_param()
        # Train best model
        model_optim, states_optim, mu_validation_preds, std_validation_preds = (
            initialize_model(param, train_data, validation_data)
        )
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_data(
            data_processor=data_processor,
            standardization=True,
            plot_test_data=False,
            plot_column=output_col,
            validation_label="y",
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
        # Save best model for SKF analysis later
        model_optim_dict = model_optim.get_dict(time_step=0)

        ##################
        # Optimize for skf
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
            skf.save_initial_states()
            return skf

        # Define parameter search space
        slope_upper_bound = 5e-2
        slope_lower_bound = 1e-3
        # # Plot synthetic anomaly
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
            train_label="data without anomaly",
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

        if param_grid_search:
            skf_param_space = {
                "std_transition_error": [1e-6, 1e-5, 1e-4, 1e-3],
                "norm_to_abnorm_prob": [1e-6, 1e-5, 1e-4, 1e-3],
                "slope": [0.002, 0.004, 0.006, 0.008, 0.01, 0.03, 0.05, 0.07, 0.09],
            }
        elif param_optimization:
            skf_param_space = {
                "std_transition_error": [1e-6, 1e-3],
                "norm_to_abnorm_prob": [1e-6, 1e-3],
                "slope": [slope_lower_bound, slope_upper_bound],
            }
        skf_optimizer = SKFOptimizer(
            initialize_skf=initialize_skf,
            model_param=model_optim_dict,
            param_space=skf_param_space,
            data=train_data,
            num_synthetic_anomaly=50,
            num_optimization_trial=num_trial_optimization * 2,
            grid_search=param_grid_search,
        )
        skf_optimizer.optimize()
        # Get parameters
        skf_param = skf_optimizer.get_best_param()
        skf_optim = initialize_skf(skf_param, model_optim_dict)
        skf_optim_dict = skf_optim.get_dict()
        skf_optim_dict["model_param"] = param
        skf_optim_dict["skf_param"] = skf_param
        with open("saved_params/benchmark_1.pkl", "wb") as f:
            pickle.dump(skf_optim_dict, f)
    else:
        ########################################
        # Load saved skf model
        with open("saved_params/benchmark_1.pkl", "rb") as f:
            skf_optim_dict = pickle.load(f)
        skf_optim = SKF.load_dict(skf_optim_dict)

    ########################################
    # Detect anomaly
    print("Model parameters used:", skf_optim_dict["model_param"])
    print("SKF model parameters used:", skf_optim_dict["skf_param"])

    filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)
    filter_marginal_abnorm_prob, states = skf_optim.smoother()

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states_type="smooth",
        states=states,
        model_prob=filter_marginal_abnorm_prob,
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
