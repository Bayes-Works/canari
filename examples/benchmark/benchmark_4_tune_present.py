import fire
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytagi import exponential_scheduler
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from src import (
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    WhiteNoise,
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

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
    }
)
import matplotlib as mpl

mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}",
    }
)

# Fix parameters:
sigma_v_fix = 0.04168150423508317
look_back_len_fix = 14


def main(
    num_trial_optimization: int = 50,
    param_tune: bool = False,
    grid_search: bool = False,
):
    # Read data
    data_file = "./data/benchmark_data/test_4_data.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(df_raw.iloc[:, 0])
    df_raw = df_raw.iloc[:, 1:]
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["displacement_y", "water_level", "temp_min", "temp_max"]
    # lags = [0, 4, 4, 4]
    # df_raw = DataProcess.add_lagged_columns(df_raw, lags)
    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df_raw,
        time_covariates=["week_of_year"],
        train_split=0.4,
        validation_split=0.07,
        output_col=output_col,
    )
    (
        data_processor.train_data,
        data_processor.validation_data,
        data_processor.test_data,
        data_processor.all_data,
    ) = data_processor.get_splits()

    # Define model
    def initialize_model(param):
        return Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=param["look_back_len"],
                num_features=5,
                num_layer=1,
                num_hidden_unit=50,
                device="cpu",
                manual_seed=1,
            ),
            WhiteNoise(std_error=param["sigma_v"]),
        )

    # Define parameter search space
    if param_tune:
        param = {
            "look_back_len": [12, 52],
            "sigma_v": [1e-3, 2e-1],
        }
        # Define optimizer
        model_optimizer = ModelOptimizer(
            initialize_model=initialize_model,
            train=training,
            param_space=param,
            data_processor=data_processor,
            num_optimization_trial=num_trial_optimization,
        )
        model_optimizer.optimize()
        # Get best model
        model_optim = model_optimizer.get_best_model()
    else:
        param = {
            "look_back_len": look_back_len_fix,
            "sigma_v": sigma_v_fix,
        }
        model_optim = initialize_model(param)

    # Train best model
    model_optim, states_optim, mu_validation_preds, std_validation_preds = training(
        model=model_optim, data_processor=data_processor
    )

    # Save best model for SKF analysis later
    model_optim_dict = model_optim.get_dict()

    # Plot
    # Fig #1
    fig, ax = plt.subplots(3, 1, figsize=(4, 2.4))
    plot_data(
        data_processor=data_processor,
        normalization=True,
        plot_test_data=False,
        plot_column=[0],
        sub_plot=ax[0],
    )
    ax[0].set_ylabel("$y$")
    plot_data(
        data_processor=data_processor,
        normalization=False,
        plot_test_data=False,
        plot_column=[1],
        sub_plot=ax[1],
    )
    ax[1].set_ylabel("water level")
    plot_data(
        data_processor=data_processor,
        normalization=False,
        plot_test_data=False,
        plot_column=[2],
        sub_plot=ax[2],
    )
    ax[2].set_ylabel("temp")
    fig.subplots_adjust(hspace=0.15)
    # plt.show()
    plt.savefig("saved_results/dependencies.pgf", format="pgf", bbox_inches="tight")

    # Fig #1
    fig, ax = plt.subplots(figsize=(4, 1))
    plot_data(
        data_processor=data_processor,
        normalization=True,
        plot_test_data=False,
        sub_plot=ax,
    )
    # ax.set_ylabel("$y$")
    # plt.show()
    plt.savefig("saved_results/data.pgf", format="pgf", bbox_inches="tight")

    # Fig #2
    fig, ax = plot_states(
        data_processor=data_processor,
        states=states_optim,
        normalization=True,
    )

    plot_data(
        data_processor=data_processor,
        normalization=True,
        plot_test_data=False,
        plot_column=output_col,
        # validation_label="y",
        sub_plot=ax[0],
    )

    plot_prediction(
        data_processor=data_processor,
        mean_validation_pred=mu_validation_preds,
        std_validation_pred=std_validation_preds,
        validation_label=[r"$\mu$", f"$\pm\sigma$"],
        sub_plot=ax[0],
    )

    train_time = data_processor.get_time(split="train")

    fig.set_size_inches(3.2, 2.3)
    # fig.subplots_adjust(hspace=0.6)  # 👈 control vertical spacing

    # Set x-limits
    for a in ax:
        a.set_xlim(train_time[0], train_time[-1])

    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[2].set_xticklabels([])
    ax[0].set_ylabel("level")
    ax[1].set_ylabel("trend")
    ax[2].set_ylabel("lstm")
    ax[3].set_ylabel("noise")
    # plt.show()
    for axis in ax:
        for line in axis.get_lines():
            line.set_linewidth(0.8)  # 👈 Set your desired linewidth
    plt.savefig("saved_results/filter.pgf", format="pgf", bbox_inches="tight")

    # Fig #3
    fig, ax = plt.subplots(figsize=(4, 1.2))
    plot_data(
        data_processor=data_processor,
        normalization=True,
        plot_test_data=False,
        sub_plot=ax,
        validation_label="obs",
    )
    # ax.set_ylabel("$y$")
    plot_prediction(
        data_processor=data_processor,
        mean_validation_pred=mu_validation_preds,
        std_validation_pred=std_validation_preds,
        validation_label=[r"$\mu$", f"$\pm\sigma$"],
        sub_plot=ax,
    )
    # ax.legend()
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        ncol=3,
        frameon=False,
    )
    for line in ax.get_lines():
        line.set_linewidth(0.8)  # 👈 Set your desired linewidth
    # plt.show()
    plt.savefig("saved_results/forecast.pgf", format="pgf", bbox_inches="tight")


def training(model, data_processor, num_epoch: int = 50):
    """
    Training procedure
    """

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
    #     data_processor.get_data("train", normalization=True),
    #     color="r",
    # )
    # plt.show()

    model.auto_initialize_baseline_states(data_processor.train_data["y"][0 : 52 * 3])
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
    start_time = time.time()
    fire.Fire(main)
    end_time = time.time()
    print(f"Run time: {end_time-start_time:.2f} seconds")
