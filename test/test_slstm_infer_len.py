import os
import pandas as pd
import numpy.testing as npt
import pytagi.metric as metric
from canari import DataProcess, Model
from canari.component import LstmNetwork, WhiteNoise
import numpy as np
import pytest

BASE_DIR = os.path.dirname(__file__)


def create_slstm_model(look_back_len: int, infer_length: int) -> Model:
    sigma_v = 3e-2
    return Model(
        LstmNetwork(
            look_back_len=look_back_len,
            num_features=2,
            infer_len=infer_length,
            num_layer=1,
            num_hidden_unit=40,
            device="cpu",
            manual_seed=2,
        ),
        WhiteNoise(std_error=sigma_v),
    )


@pytest.mark.parametrize(
    "look_back_len,start_offset", [(l, s) for l in [12, 18, 24] for s in [0, 12]]
)
def test_slstm_infer_len_parametrized(look_back_len, start_offset, plot_mode):
    """
    Run training and forecasting for time-series forecasting model for multiple inference lengths.
    """
    output_col = [0]
    infer_length = 24 * 5

    # # Read data
    data_file = os.path.join(BASE_DIR, "../data/toy_time_series/sine.csv")
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    data_file_time = os.path.join(BASE_DIR, "../data/toy_time_series/sine_datetime.csv")
    time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(time_series[0])
    df_raw.index = time_series

    df_raw.index.name = "date_time"
    df_raw.columns = ["values"]

    # Resampling data
    df = df_raw.resample("H").mean()

    # offset the data by a random start time
    df = df.iloc[start_offset:]

    # Data processing
    data_processor = DataProcess(
        data=df,
        train_split=0.8,
        validation_split=0.2,
        time_covariates=["hour_of_day"],
        output_col=output_col,
    )
    train_data, validation_data, _, _ = data_processor.get_splits()

    # Initialize model
    model = create_slstm_model(look_back_len, infer_length)

    # Train model
    num_epoch = 50
    for epoch in range(num_epoch):
        (mu_validation_preds, _, states) = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
        )

        # Calculate the log-likelihood metric
        validation_obs = data_processor.get_data(
            "validation", standardization=True
        ).flatten()
        mse = metric.mse(mu_validation_preds, validation_obs)

        # Early-stopping
        model.early_stopping(
            evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch
        )
        if epoch == model.optimal_epoch:
            model_optim_dict = model.get_dict()
            model_optim_dict["cov_names"] = train_data["cov_names"]

        if model.stop_training:
            break

    # Testing first prediction
    # filter using smoothed look-back from optimal epoch
    _, _, states = model.filter(data=train_data, train_lstm=False)
    model.smoother()

    # Get first state and observation
    prior_states_mu = states.get_mean("lstm", "prior", True)
    prior_states_std = states.get_std("lstm", "prior", True)
    first_state = prior_states_mu[0]
    first_observation = train_data["y"][0]

    # plot (optional)
    if plot_mode:

        look_back_mu = model.early_stop_lstm_output_mu
        look_back_std = model.early_stop_lstm_output_var**0.5

        states_mu = prior_states_mu[: len(train_data["y"])]
        states_std = prior_states_std[: len(train_data["y"])]

        look_back_len_actual = len(look_back_mu)
        x_look_back = np.arange(-look_back_len_actual, 0)
        x_states = np.arange(0, len(states_mu))
        x_obs = np.arange(0, len(train_data["y"]))

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(
            x_look_back,
            look_back_mu,
            label="Look-back (smoothed)",
            color="tab:green",
        )
        plt.fill_between(
            x_look_back,
            look_back_mu - look_back_std,
            look_back_mu + look_back_std,
            color="tab:green",
            alpha=0.3,
        )
        plt.plot(
            x_states,
            states_mu,
            label="Prior states",
            color="tab:blue",
        )
        plt.fill_between(
            x_states,
            states_mu - states_std,
            states_mu + states_std,
            color="tab:blue",
            alpha=0.3,
        )
        plt.plot(x_obs, train_data["y"], label="Observation", color="tab:red")
        plt.axvline(x=-0.5, color="k", ls="--", label="Look-back boundary")

        plt.xlabel("Relative time (0 = first observation)")
        plt.ylabel("Value")
        plt.title(f"SLSTM with look_back_len={look_back_len}")
        plt.legend()
        plt.show()

    print(f"Observation : {first_observation}")
    print(f"Observation : {first_state}")

    npt.assert_allclose(
        first_state,
        first_observation,
        atol=0.09,
        err_msg=f"First state mismatch for look_back_len={look_back_len} with start_offset={start_offset}",
    )

    # Testing mse for the inferred look-back length
    mu_infer = model.lstm_output_history.mu
    real_data = data_processor.get_data(split="all", standardization=True).flatten()
    obs_pretrain = real_data[:infer_length]
    obs_pretrain = obs_pretrain[-look_back_len:]
    mse = metric.mse(mu_infer, obs_pretrain)

    print(f"MSE : {mse}")

    assert mse < 2e-2, f"MSE {mse} is bigger than the saved threshold {2e-2}"
