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


def main(
    num_trial_optim_model: int = 20,
    num_trial_optim_skf: int = 30,
    param_optimization: bool = True,
    param_grid_search: bool = False,
    smoother: bool = True,
    plot: bool = False,
):
    ######### Data processing #########
    # Read data
    data_file = "./data/benchmark_data/test_2_data.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=";", header=None)
    time_series = pd.to_datetime(df_raw.iloc[:, 4])
    df_raw = df_raw.iloc[:, 6].to_frame()
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["values"]
    df = df_raw.resample("W").mean()
    df = df.iloc[30:, :]
    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df,
        time_covariates=["week_of_year"],
        train_split=0.25,
        validation_split=0.08,
        test_split=0.67,
        output_col=output_col,
    )
    train_data, validation_data, test_data, all_data = data_processor.get_splits()
    seed = np.random.randint(0, 100)

    ######### Define model with parameters #########
    def model_obj(param):
        model, *_ = model_with_parameters(param, train_data, validation_data)
        return model.metric_optim

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
            LocalTrend(var_states=[1e-1, 1e-1]),
            LstmNetwork(
                look_back_len=int(param[0]),
                num_features=2,
                num_layer=1,
                infer_len=52 * 3,
                num_hidden_unit=50,
                manual_seed=seed,
                smoother=smoother,
            ),
            WhiteNoise(std_error=param[1]),
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

            if model.stop_training:
                break

        return (
            model,
            mu_validation_preds_optim,
            std_validation_preds_optim,
        )

    param = [22, 0.195]
    model_optim, mu_validation_preds, std_validation_preds = model_with_parameters(
        param, train_data, validation_data
    )
    model_optim_dict = model_optim.get_dict(time_step=0)

    # x0 = [
    #     24,
    #     0.01,
    # ]

    # bounds = [(10, 52), (1e-4, 0.5)]
    # # res = minimize(model_obj, x0, bounds=bounds, method="L-BFGS-B")
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
    # model_optim, *_ = model_with_parameters(res.x, train_data, validation_data)
    # model_optim_dict = model_optim.get_dict(time_step=0)

    # print("Optimal parameters:", res.x)
    # print("Function value:", res.fun)

    def skf_obj(skf_param):
        skf = skf_with_parameters(skf_param, model_optim_dict, train_data)

        #
        detection_rate = skf.metric_optim["detection_rate"]
        false_rate = skf.metric_optim["false_rate"]
        false_alarm_train = skf.metric_optim["false_alarm_train"]
        anm_magnitude = skf.metric_optim["anomaly_magnitude"]
        #
        mean = 0.5  # mean of normal
        std_dev = 0.05  # std deviation of normal
        # metric
        j1 = norm.cdf(detection_rate, loc=mean, scale=std_dev)
        j2 = 1 - lognorm.cdf(false_rate, s=0.2, scale=0.1)
        j3 = 1 - lognorm.cdf(anm_magnitude, s=0.2, scale=0.3)
        # j3 = 1 - lognorm.cdf(anm_magnitude, s=0.3, scale=0.1/4)
        _metric = j1 * j2 * j3

        return -_metric

    ######### Define SKF model with parameters #########
    slope = 0.52

    def skf_with_parameters(skf_param_space, model_param: dict, train_data):
        norm_model = Model.load_dict(model_param)

        abnorm_model = Model(
            LocalAcceleration(),
            LstmNetwork(),
            WhiteNoise(),
        )
        skf = SKF(
            norm_model=norm_model,
            abnorm_model=abnorm_model,
            # std_transition_error=np.exp(skf_param_space[0]),
            # norm_to_abnorm_prob=np.exp(skf_param_space[1]),
            std_transition_error=skf_param_space[0],
            norm_to_abnorm_prob=skf_param_space[1],
            # std_transition_error=1e-5,
            # norm_to_abnorm_prob=1e-6,
        )
        skf.save_initial_states()

        num_anomaly = 50
        detection_rate, false_rate, false_alarm_train = skf.detect_synthetic_anomaly(
            data=train_data,
            num_anomaly=num_anomaly,
            # slope_anomaly=slope / 52,
            slope_anomaly=skf_param_space[2] / 52,
        )

        data_len_year = (
            data_processor.data.index[data_processor.train_end]
            - data_processor.data.index[data_processor.train_start]
        ).days / 365.25
        skf.metric_optim["detection_rate"] = detection_rate
        skf.metric_optim["false_rate"] = false_rate / data_len_year
        if false_alarm_train == "Yes":
            skf.metric_optim["false_alarm_train"] = 1 / data_len_year
        else:
            skf.metric_optim["false_alarm_train"] = 0
        skf.metric_optim["anomaly_magnitude"] = skf_param_space[2]

        return skf

    # ---------- logging wrapper (Option 1) ----------

    # def skf_obj_logged(x_log):
    #     """Wraps skf_obj to record and print every evaluation for ANY number of parameters."""
    #     val = skf_obj(x_log)

    #     # convert log-space vector to linear for readability
    #     # x_lin = np.exp(np.asarray(x_log))
    #     x_lin = np.asarray(x_log)

    #     # store
    #     eval_history.append((x_lin.copy(), float(val)))

    #     # pretty print all parameters
    #     params_str = ", ".join([f"x{i}={p:.6g}" for i, p in enumerate(x_lin)])

    #     print(f"Eval #{len(eval_history):4d} | {params_str} | objective={val:.6f}")

    #     return val

    eval_history = []
    best_val = float("inf")
    best_x = None

    # ------------------------------------------------------------
    # Logging wrapper
    # ------------------------------------------------------------
    def skf_obj_logged(x_log):
        """Wraps skf_obj to record and print every evaluation.
        Tracks global best over all calls.
        """
        nonlocal best_val, best_x

        # evaluate
        val = skf_obj(x_log)

        # choose linear/log representation
        x_lin = np.asarray(x_log, dtype=float)

        # store full history
        eval_history.append((x_lin.copy(), float(val)))

        # update global best
        if val < best_val:
            best_val = float(val)
            best_x = x_lin.copy()

        # pretty-print
        params_str = ", ".join([f"x{i}={p:.6g}" for i, p in enumerate(x_lin)])

        print(
            f"Eval #{len(eval_history):4d} | {params_str} | "
            f"objective={val:.6f} | best={best_val:.6f}"
        )

        return val

    skf_x0 = [5e-5, 1e-5, 1e-1]
    bounds = [(1e-6, 1e-4), (1e-6, 1e-4), (0.052, 2.6)]
    # skf_x0 = [np.log(1e-4), np.log(1e-4)]
    # bounds = [(np.log(1e-6), np.log(1e-4)), (np.log(1e-6), np.log(1e-4))]
    # res = minimize(skf_obj, skf_x0, bounds=bounds, method='L-BFGS-B')
    res = minimize(
        skf_obj_logged,
        skf_x0,
        bounds=bounds,
        # method="L-BFGS-B",
        # options={
        #     "maxiter": 200,
        #     "gtol": 1e-5,
        #     "eps": 5e-1,  # <---- finite difference step size in log-space
        #     "disp": True,  # optional: show iteration log
        # },
        method="Powell",
        options={"maxiter": 50, "disp": True},
    )

    print("\n=== Optimization complete ===")

    # convert all optimized parameters from log-space → linear
    # x_opt_lin = np.exp(res.x)
    x_opt_lin = res.x

    # pretty print all parameters
    params_str = ", ".join([f"x{i}={p:.6g}" for i, p in enumerate(x_opt_lin)])
    print(f"Optimal: {params_str}")

    print(f"Optimal objective value: {res.fun:.6f}")
    print(f"Function evaluations: {res.nfev}, Iterations: {res.nit}")

    skf_optim = skf_with_parameters(res.x, model_optim_dict, train_data)

    filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)
    smooth_marginal_abnorm_prob, states = skf_optim.smoother()

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        model_prob=filter_marginal_abnorm_prob,
    )
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    plt.savefig("./saved_results/BM2.png")
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
