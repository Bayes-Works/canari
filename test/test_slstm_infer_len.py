import os
import pandas as pd
import numpy.testing as npt
import copy
import pytagi.metric as metric
from canari import DataProcess, Model
from canari.component import LstmNetwork, WhiteNoise
from canari.data_visualization import plot_data, plot_states
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)


def create_slstm_model(look_back_len: int) -> Model:
    sigma_v = 0.001
    return Model(
        LstmNetwork(
            look_back_len=look_back_len,
            num_features=2,
            infer_len=24 * 2,
            num_layer=1,
            num_hidden_unit=40,
            device="cpu",
            manual_seed=1,
        ),
        WhiteNoise(std_error=sigma_v),
    )


def test_slstm_infer_len_parametrized():
    """
    Run training and forecasting for time-series forecasting model for multiple inference lengths.
    """
    look_back_lens = [11, 19, 23]  # different input sequence lengths
    start_offsets = [0, 6, 12]  # creates offsets in cycles

    # create combinations of look_back_len and start_offset
    infer_params = [
        (look_back_len, start_offset)
        for look_back_len in look_back_lens
        for start_offset in start_offsets
    ]

    output_col = [0]

    # Read and prepare data
    data_file = os.path.join(BASE_DIR, "../data/toy_time_series/sine.csv")
    df_raw_orig = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

    data_file_time = os.path.join(BASE_DIR, "../data/toy_time_series/sine_datetime.csv")
    time_series_orig = pd.read_csv(
        data_file_time, skiprows=1, delimiter=",", header=None
    )

    for look_back_len, start_offset in infer_params:

        df_raw = copy.copy(df_raw_orig)

        time_series = pd.to_datetime(time_series_orig[0])
        df_raw.index = time_series

        df_raw = df_raw[start_offset:]

        # Data processing
        data_processor = DataProcess(
            data=df_raw,
            train_split=0.8,
            validation_split=0.2,
            time_covariates=["hour_of_day"],
            output_col=output_col,
        )
        train_data, validation_data, _, _ = data_processor.get_splits()

        # Initialize model
        model = create_slstm_model(look_back_len=look_back_len)

        # Train model
        for epoch in range(num_epoch := 50):
            (mu_validation_preds, _, states) = model.lstm_train(
                train_data=train_data,
                validation_data=validation_data,
            )

            # Calculate the log-likelihood metric
            validation_obs = data_processor.get_data("validation").flatten()
            mse = metric.mse(mu_validation_preds, validation_obs)

            # Early-stopping
            model.early_stopping(
                evaluate_metric=mse, current_epoch=epoch, max_epoch=num_epoch
            )
            if epoch == model.optimal_epoch:
                states_optim = copy.copy(states)

            model.set_memory(states=states, time_step=0)
            if model.stop_training:
                break

        # Get first state and observation
        first_state = states.get_mean("lstm", "prior", True)[0]
        first_observation = train_data["y"][0]

        npt.assert_allclose(
            first_state,
            first_observation,
            atol=0.2,
            err_msg=f"First state mismatch for look_back_len={look_back_len} with start_offset={start_offset}",
        )
