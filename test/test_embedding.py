import os
import pandas as pd
import numpy as np
from canari import DataProcess, Model
from canari.component import LocalTrend, LstmNetwork, WhiteNoise
import pytest
from typing import Any

BASE_DIR = os.path.dirname(__file__)


def model_test_runner(model: Model, run_mode: str) -> Any:
    """
    Run training SSM+LSTM model with embedding functionality
    """

    output_col = [0]

    # Read data
    data_file = os.path.join(BASE_DIR, "../data/toy_time_series/sine.csv")
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
    linear_space = np.linspace(0, 2, num=len(df_raw))
    df_raw = df_raw.add(linear_space, axis=0)
    data_file_time = os.path.join(BASE_DIR, "../data/toy_time_series/sine_datetime.csv")
    time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(time_series[0])
    df_raw.index = time_series

    # Data processing
    data_processor = DataProcess(
        data=df_raw,
        train_split=0.8,
        validation_split=0.2,
        output_col=output_col,
    )
    train_data, validation_data, _, _ = data_processor.get_splits()

    # Initialize model
    model.auto_initialize_baseline_states(train_data["y"][0:24])
    num_epoch = 10
    for epoch in range(num_epoch):
        (mu_validation_preds, std_validation_preds, states) = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
        )

    if run_mode == "fixed_embedding":
        return model.lstm_embedding
    elif run_mode == "update_embedding":
        # train for one more epoch to update embedding
        model.filter(train_data, train_lstm=True, update_embedding=True)
        return model.lstm_embedding


@pytest.mark.parametrize("run_mode", [("fixed_embedding"), ("update_embedding")])
@pytest.mark.parametrize("embed_mode", [("set_embedding"), ("init_embedding")])
@pytest.mark.parametrize("smoother", [(False), (True)], ids=["LSTM", "SLSTM"])
def test_model_embedding(embed_mode, smoother, run_mode):
    "Test embedding functionality in LSTM model"

    if embed_mode == "set_embedding":

        # Define embedding
        embed_len = 15
        embed_mu = (
            np.sin(np.linspace(0, 2 * np.pi, embed_len))
            .reshape(1, -1)
            .astype(np.float32)
        )
        embed_var = (np.ones_like(embed_mu) * 0.1).astype(np.float32)

        model = Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=19,
                num_features=1,
                num_layer=1,
                infer_len=24,
                num_hidden_unit=50,
                device="cpu",
                manual_seed=1,
                smoother=smoother,
                embedding=(embed_mu, embed_var),
            ),
            WhiteNoise(std_error=0.003),
        )
    elif embed_mode == "init_embedding":

        model = Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=19,
                num_features=1,
                num_layer=1,
                infer_len=24,
                num_hidden_unit=50,
                device="cpu",
                manual_seed=1,
                smoother=smoother,
                embed_len=15,
            ),
            WhiteNoise(std_error=0.003),
        )
    else:
        raise ValueError("Invalid embed_mode")

    lstm_embedding = model_test_runner(model, run_mode)

    assert lstm_embedding is not None
    assert lstm_embedding.length == 15

    if embed_mode == "set_embedding":
        if run_mode == "fixed_embedding":
            np.testing.assert_array_almost_equal(
                lstm_embedding.mu.reshape(1, -1),
                embed_mu,
                decimal=5,
                err_msg="LSTM embedding mean does not match the set embedding mean",
            )
            np.testing.assert_array_almost_equal(
                lstm_embedding.var.reshape(1, -1),
                embed_var,
                decimal=5,
                err_msg="LSTM embedding variance does not match the set embedding variance",
            )
        elif run_mode == "update_embedding":
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_almost_equal,
                lstm_embedding.mu.reshape(1, -1),
                embed_mu,
                decimal=5,
                err_msg="LSTM embedding mean should have been updated and not match the set embedding mean",
            )
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_almost_equal,
                lstm_embedding.var.reshape(1, -1),
                embed_var,
                decimal=5,
                err_msg="LSTM embedding variance should have been updated and not match the set embedding variance",
            )
