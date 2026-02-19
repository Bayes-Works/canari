# # Run tourism monthly 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm
from bm.utils import p50, p90

import fire
import copy
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend, ExpSmoothing, LocalLevel, Autoregression

def main():

    time_start = time.time()

    # Read data
    data_train_file = "./data/m4/Hourly-train.csv"
    df_train = pd.read_csv(data_train_file, skiprows=1, delimiter=",", header=None)
    data_test_file = "./data/m4/Hourly-test.csv"
    df_test = pd.read_csv(data_test_file, skiprows=1, delimiter=",", header=None)

    time_series = np.arange(414)
    # exclude = [57, 127, 132]
    # time_series = np.delete(time_series, exclude)

    # time_series = np.arange(132, 414)
    # exclude = [57, 127, 132, 133]
    # time_series = time_series[~np.isin(time_series, exclude)]

    mu_test_all = np.zeros((48,len(time_series)))
    std_test_all = np.zeros((48,len(time_series)))
    test_obs_all = np.zeros((48,len(time_series)))
    saved_result = {
        "states": {},
        "mu_test": {},
        "std_test": {},
        "test_obs": {},
        "p50": {},
        "p90": {},
    }

    # for ts in tqdm(time_series, desc="Time series"):

    pbar = tqdm(time_series, desc="Time series")
    for ts in pbar:
        pbar.set_postfix(ts=ts)  # shows current ts on the bar line
        mu_test, std_test, states, test_obs = m4_hour(df_train, df_test, ts)
        saved_result["states"][ts] = states
        mu_test_all[:,ts] = mu_test.flatten()
        std_test_all[:,ts] = std_test.flatten()
        test_obs_all[:,ts] = test_obs.flatten()

    # Metrics
    p50_overall = p50(test_obs_all, mu_test_all, std_test_all)
    p90_overall = p90(test_obs_all, mu_test_all, std_test_all)

    saved_result["mu_test"] = mu_test_all
    saved_result["std_test"] = std_test_all
    saved_result["p50"] = p50_overall
    saved_result["p90"] = p90_overall

    with open("saved_results/bm/m4_ar.pkl", "wb") as f:
        pickle.dump(saved_result, f)

    time_end = time.time()
    print(f"Runtime: {time_end - time_start:.2f} seconds")
    print(f"p50: {p50_overall:.4f}")
    print(f"p90: {p90_overall:.4f}")

def _prepare_series(df_train, df_test, ts):
    df_train = df_train.iloc[ts, 1:].to_frame()
    df_train = df_train.dropna()
    df_train = df_train.astype(float)

    df_test = df_test.iloc[ts, 1:].to_frame()
    df_test = df_test.dropna()
    df_test = df_test.astype(float)

    df = pd.concat([df_train, df_test], axis=0)
    train_start_time = pd.Timestamp(
        year=2000,
        month=1,
        day=1,
        hour=12,
    )

    df.index = pd.date_range(
        start=train_start_time,
        periods=len(df),
        freq="H"
    )

    nb_train = len(df_train)
    return df, nb_train


def m4_hour(df_train, df_test, ts):

    df, nb_train = _prepare_series(df_train, df_test, ts)

    # Define parameters
    output_col = [0]
    num_epoch = 50
    nb_val = 12
    # Build data processor
    data_processor = DataProcess(
        data=df,
        train_start=df.index[0],
        validation_start=df.index[nb_train - nb_val],
        test_start=df.index[nb_train],
        time_covariates=["hour_of_day"],
        output_col=output_col,
    )
    # split data
    train_data, validation_data, test_data, _ = data_processor.get_splits()
    trainval = data_processor.get_splits(split="train_val")

    # Model
    lstm_smoother = True
    var_noise = 1e-2
    model = Model(
        LocalTrend(var_states=[1e-2, 1e-4]),
        # ExpSmoothing(mu_states=[0,-3,0], var_states=[0,1e-2,0], es_order=1, activation="sigmoid"),
        ExpSmoothing(mu_states=[0,0.5,0], var_states=[0,1e-2,0], es_order=1, activation=None),
        LstmNetwork(
            look_back_len=12,
            num_features=2,
            infer_len=24 * 3,
            num_layer=1,
            num_hidden_unit=50,
            # manual_seed=1,
            model_noise=True,
            smoother=lstm_smoother,
        ),
        Autoregression(
            mu_states=[0, 0.9, 0, 0, 0, var_noise],
            var_states=[
                1e-5,
                0.25,
                0,
                var_noise,
                1e-6,
                1e-2,
            ],
        ),
    )

    model.auto_initialize_baseline_states(train_data["y"])

    # Training
    for epoch in range(num_epoch):
        # (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
        #     train_data=train_data,
        #     validation_data=validation_data,
        # )

        # # Unstandardize the predictions
        # mu_validation_preds = normalizer.unstandardize(
        #     mu_validation_preds,
        #     data_processor.scale_const_mean[output_col],
        #     data_processor.scale_const_std[output_col],
        # )
        # std_validation_preds = normalizer.unstandardize_std(
        #     std_validation_preds,
        #     data_processor.scale_const_std[output_col],
        # )


        # # Calculate the metric
        # validation_obs = data_processor.get_data("validation").flatten()
        # validation_log_lik = metric.log_likelihood(
        #     prediction=mu_validation_preds,
        #     observation=validation_obs,
        #     std=std_validation_preds,
        # )

        # # Early-stopping
        # model.early_stopping(
        #     evaluate_metric=-validation_log_lik, current_epoch=epoch, max_epoch=num_epoch
        # )

        # if model.stop_training:
        #     break

        model.white_noise_decay(
            epoch,
            white_noise_max_std=3,
            white_noise_decay_factor=0.9,
        )

        if lstm_smoother:
            model.pretraining_filter(trainval)

        model.filter(
            data=trainval,
        )
        model.smoother()
        model.set_memory(time_step=0)
        model._current_epoch += 1

    model.set_memory(
        time_step=data_processor.test_start - 1,
    )

    # forecat on the test set
    mu_test_preds, std_test_preds, _ = model.forecast(
        data=test_data,
    )

    _states_plot = copy.copy(model.states)
    # plot the test data
    level_sum = _states_plot.get_mean(states_name="level") + _states_plot.get_mean(states_name="es")
    for i in range(len(_states_plot.mu_posterior)):
        _states_plot.mu_posterior[i][0] = level_sum[i]

    fig, ax = plot_states(
        data_processor=data_processor,
        states=_states_plot,
        standardization=True,
        states_to_plot=["level", "trend", "es", "es coeff", "es prod", "lstm", "autoregression", "AR_error"],
        color="k",
    )
    plot_data(
        data_processor=data_processor,
        standardization=True,
        plot_column=output_col,
        plot_test_data=True,
        sub_plot=ax[0],
    )
    plot_prediction(
        data_processor=data_processor,
        mean_test_pred=mu_test_preds,
        std_test_pred=std_test_preds,
        sub_plot=ax[0],
    )
    fig.suptitle(f"TS #{ts}", fontsize=10, y=1)
    plt.savefig(f"saved_results/bm/m4/TS_{ts}_ar.png", dpi=200, bbox_inches="tight")
    plt.close() 

    # Unstandardize the predictions
    mu_test_preds = normalizer.unstandardize(
        mu_test_preds,
        data_processor.scale_const_mean[output_col],
        data_processor.scale_const_std[output_col],
    )
    std_test_preds = normalizer.unstandardize_std(
        std_test_preds,
        data_processor.scale_const_std[output_col],
    )

    test_obs = data_processor.get_data(split="test", standardization = False).flatten()

    return mu_test_preds.flatten(), std_test_preds.flatten(), model.states, test_obs

if __name__ == "__main__":
    fire.Fire(main)