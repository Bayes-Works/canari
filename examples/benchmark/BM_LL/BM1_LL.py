import fire
import copy
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytagi import metric
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
    num_trial_optim_model: int = 100,
    param_optimization: bool = True,
    smoother: bool = True,
    plot: bool = False,
):
    ######### Data processing #########
    # Read data
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
    seed = np.random.randint(0, 100)

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
                manual_seed=seed,
                smoother=smoother,
            ),
            WhiteNoise(std_error=param["sigma_v"]),
        )

        model.auto_initialize_baseline_states(train_data["y"][0 : 52 * 3])
        num_epoch = 50
        for epoch in range(num_epoch):
            mu_validation_preds, std_validation_preds, _ = model.lstm_train(
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

            if model.stop_training:
                break

        #### Define SKF model with parameters #########

        abnorm_model = Model(
            LocalAcceleration(),
            LstmNetwork(),
            WhiteNoise(),
        )
        skf = SKF(
            norm_model=model,
            abnorm_model=abnorm_model,
            std_transition_error=param["std_transition_error"],
            norm_to_abnorm_prob=param["norm_to_abnorm_prob"],
        )

        skf.save_initial_states()

        mu_preds, std_preds, _, _ = skf.filter(data=all_data)

        mu_preds_unnorm = normalizer.unstandardize(
            mu_preds,
            data_processor.scale_const_mean[data_processor.output_col],
            data_processor.scale_const_std[data_processor.output_col],
        )

        std_preds_unnorm = normalizer.unstandardize_std(
            std_preds,
            data_processor.scale_const_std[data_processor.output_col],
        )

        obs_all = data_processor.get_data("all").flatten()
        log_lik_all = metric.log_likelihood(
            prediction=mu_preds_unnorm,
            observation=obs_all,
            std=std_preds_unnorm,
        )
        skf.metric_optim = -log_lik_all

        skf.load_initial_states()

        return skf

    if param_optimization:
        param_space = {
            "look_back_len": [12, 52],
            "sigma_v": [1e-3, 2e-1],
            "std_transition_error": [1e-6, 1e-4],
            "norm_to_abnorm_prob": [1e-6, 1e-4],
        }
        # Define optimizer
        model_optimizer = ModelOptimizer(
            model=model_with_parameters,
            param_space=param_space,
            train_data=train_data,
            validation_data=validation_data,
            num_optimization_trial=num_trial_optim_model,
            num_startup_trials=50,
        )
        model_optimizer.optimize()
        # Get best model
        param = model_optimizer.get_best_param()
        skf_optim = model_with_parameters(param, train_data, validation_data)

        skf_optim_dict = skf_optim.get_dict()
        skf_optim_dict["model_param"] = param
        skf_optim_dict["cov_names"] = train_data["cov_names"]
        with open("saved_params/benchmark_1_LL_1.pkl", "wb") as f:
            pickle.dump(skf_optim_dict, f)
    else:
        with open("saved_params/benchmark_1_LL_1.pkl", "rb") as f:
            skf_optim_dict = pickle.load(f)
        skf_optim = SKF.load_dict(skf_optim_dict)

    _, _, states, filter_marginal_abnorm_prob = skf_optim.filter(data=all_data)

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        model_prob=filter_marginal_abnorm_prob,
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.savefig("./saved_results/BM1_1.png")
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
