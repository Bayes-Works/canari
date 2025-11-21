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
from scipy.optimize import minimize
from scipy.stats import norm, lognorm


def main(smoother: bool = True):
    ######### Data processing #########
    # Read data
    data_file = "./data/benchmark_data/test_5_data.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(df_raw.iloc[:, 0])
    df_raw = df_raw.iloc[:, 1:]
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["y", "water_level", "temp_min", "temp_max"]
    lags = [0, 4, 4, 4]
    df_raw = DataProcess.add_lagged_columns(df_raw, lags)
    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df_raw,
        time_covariates=["week_of_year"],
        train_split=0.289,
        validation_split=0.0693,
        output_col=output_col,
    )
    train_data, validation_data, test_data, all_data = data_processor.get_splits()
    seed = np.random.randint(0, 100)

    ######### Define model with parameters #########
    def model_obj(param):
        metric, _ = model_with_parameters(param, train_data, validation_data)
        return metric

    model_history = []

    def model_obj_logged(x):
        """Wraps skf_obj to record and print every evaluation for ANY number of parameters."""
        val = model_obj(x)

        # convert log-space vector to linear for readability
        # x_lin = np.exp(np.asarray(x_log))
        x_lin = np.asarray(x)

        # store
        model_history.append((x_lin.copy(), float(val)))

        # pretty print all parameters
        params_str = ", ".join([f"x{i}={p:.6g}" for i, p in enumerate(x_lin)])

        print(f"Eval #{len(model_history):4d} | {params_str} | objective={val:.6f}")

        return val

    def model_with_parameters(param, train_data, validation_data):
        model = Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=12,
                num_features=17,
                num_layer=1,
                infer_len=52 * 3,
                num_hidden_unit=50,
                manual_seed=seed,
                smoother=smoother,
            ),
            WhiteNoise(std_error=param[0]),
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
            std_transition_error=param[1],
            norm_to_abnorm_prob=param[2],
        )

        skf.save_initial_states()

        filter_marginal_abnorm_prob, states, mu_preds, std_preds = skf.filter(
            data=all_data
        )

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

        skf.load_initial_states()

        return -log_lik_all, skf

    x0 = [
        0.01,
        1e-5,
        1e-5,
    ]

    bounds = [(1e-3, 0.4), (1e-6, 1e-4), (1e-6, 1e-4)]
    res = minimize(model_obj_logged, x0, bounds=bounds, method="L-BFGS-B")
    print("Optimal parameters:", res.x)
    print("Function value:", res.fun)

    # res = minimize(
    #     model_obj_logged,
    #     x0,
    #     bounds=bounds,
    #     method="Powell",
    #     # options={"maxiter": 50, "disp": True},
    #     options={
    #         "maxiter": 50,
    #         "maxfev": 500,
    #         "xtol": 1e-3,
    #         "ftol": 1e-3,  # absolute AND relative
    #         "disp": True,
    #     },
    # )

    _, skf_optim = model_with_parameters(res.x, train_data, validation_data)

    skf_optim_dict = skf_optim.get_dict()
    skf_optim_dict["cov_names"] = train_data["cov_names"]
    with open("saved_params/benchmark_5_LL_scipy.pkl", "wb") as f:
        pickle.dump(skf_optim_dict, f)

    filter_marginal_abnorm_prob, states, *_ = skf_optim.filter(data=all_data)

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        model_prob=filter_marginal_abnorm_prob,
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.savefig("./saved_results/BM5_LL_scipy.png")
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
