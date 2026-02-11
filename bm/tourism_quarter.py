import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise, LocalTrend, ExpSmoothing, LocalLevel

def _prepare_series(df_train, df_test, ts):
    quarter_to_month = {1: 1, 2: 4, 3: 7, 4: 10}
    df_train = df_train.iloc[:, ts].to_frame()
    train_start_time = pd.Timestamp(
        year=int(df_train.iloc[2, 0]),
        month=quarter_to_month[int(df_train.iloc[3, 0])],
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
        freq="QS"
    )

    nb_train = len(df_train)
    return df, nb_train


def tourism_quarter(df_train, df_test, ts):

    df, nb_train = _prepare_series(df_train, df_test, ts)

    # Define parameters
    output_col = [0]
    num_epoch = 50
    nb_val = 4

    # Build data processor
    data_processor = DataProcess(
        data=df,
        train_start=df.index[0],
        validation_start=df.index[nb_train - nb_val],
        test_start=df.index[nb_train],
        time_covariates=["quarter_of_year"],
        output_col=output_col,
    )
    # split data
    train_data, validation_data, test_data, _ = data_processor.get_splits()

    # Model
    model = Model(
        LocalTrend(),
        # ExpSmoothing(mu_states=[0,-0.5,0], var_states=[0,0.2,0], es_order=1, activation="sigmoid"),
        ExpSmoothing(mu_states=[0,0.3,0], var_states=[0,1e-2,0], es_order=1, activation=None),
        LstmNetwork(
            look_back_len=4,
            num_features=2,
            infer_len=4 * 3,
            num_layer=1,
            num_hidden_unit=50,
            manual_seed=1,
            model_noise=True,
            smoother=False,
        ),
    )

    model.auto_initialize_baseline_states(train_data["y"])

    # Training
    for epoch in range(num_epoch):
        (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
        )

        # Unstandardize the predictions
        mu_validation_preds = normalizer.unstandardize(
            mu_validation_preds,
            data_processor.scale_const_mean[output_col],
            data_processor.scale_const_std[output_col],
        )
        std_validation_preds = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor.scale_const_std[output_col],
        )


        # Calculate the metric
        validation_obs = data_processor.get_data("validation").flatten()
        validation_log_lik = metric.log_likelihood(
            prediction=mu_validation_preds,
            observation=validation_obs,
            std=std_validation_preds,
        )

        # Early-stopping
        model.early_stopping(
            evaluate_metric=-validation_log_lik, current_epoch=epoch, max_epoch=num_epoch
        )

        if model.stop_training:
            break

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
    for i in range(len(states.mu_posterior)):
        _states_plot.mu_posterior[i][0] = level_sum[i]

    fig, ax = plot_states(
        data_processor=data_processor,
        states=_states_plot,
        standardization=True,
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
    plt.savefig(f"bm/results/tourism_quarter/TS_{ts}.png", dpi=200, bbox_inches="tight")
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
