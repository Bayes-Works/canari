import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend, ExpSmoothing, LocalLevel, Autoregression


def _prepare_series(df_train, df_test, ts):
    df_train = df_train.iloc[:, ts].to_frame()
    train_start_time = pd.Timestamp(
        year=int(df_train.iloc[2, 0]),
        month=int(df_train.iloc[3, 0]),
        day=1
    )
    df_train = df_train.iloc[4:,:]
    df_train = df_train.astype(float)
    df_train = df_train.dropna()

    df_test = df_test.iloc[:, ts].to_frame()
    df_test = df_test.iloc[4:,:]
    df_test = df_test.astype(float)
    df_test = df_test.dropna()

    df = pd.concat([df_train, df_test], axis=0)

    df.index = pd.date_range(
        start=train_start_time,
        periods=len(df),
        freq="MS"
    )

    nb_train = len(df_train)
    return df, nb_train

def tourism_month(df_train, df_test, ts):

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
        time_covariates=["month_of_year"],
        output_col=output_col,
    )

    # split data
    train_data, validation_data, test_data, _ = data_processor.get_splits()
    trainval = data_processor.get_splits(split="train_val")

    # Model
    lstm_smoother=True
    var_noise = 1e-2
    model = Model(
        LocalTrend(),
        # ExpSmoothing(mu_states=[0,-0.5,0], var_states=[0,0.2,0], es_order=1, activation="sigmoid"),
        # ExpSmoothing(mu_states=[0,0.3,0], var_states=[0,1e-2,0], es_order=1, activation=None),
        ExpSmoothing(mu_states=[0,.3, 0, 0, 1e-3, 0], var_states=[0,1e-2,0,0,1e-5,0], es_order=2, activation=None),
        LstmNetwork(
            look_back_len=12,
            num_features=2,
            infer_len=12 * 3,
            num_layer=1,
            num_hidden_unit=50,
            manual_seed=2,
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

    model.set_memory(
        time_step=data_processor.test_start - 1,
    )

    # forecat on the test set
    mu_test_preds, std_test_preds, states = model.forecast(
        data=test_data,
    )


    _states_plot = copy.copy(model.states)
    # plot the test data
    level_sum = _states_plot.get_mean(states_name="level") + _states_plot.get_mean(states_name="es")
    for i in range(len(states.mu_posterior)):
        _states_plot.mu_posterior[i][0] = level_sum[i]

    fig, ax = plot_states(
        data_processor=data_processor,
        states=_states_plot,
        standardization=True,
        states_to_plot=["1", "2","3","4","5","6"],
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
    plt.savefig(f"saved_results/bm/tourism_month/TS_{ts}.png", dpi=200, bbox_inches="tight")
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


