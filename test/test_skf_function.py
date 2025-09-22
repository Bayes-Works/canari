import pytest
import numpy as np
import numpy.testing as npt

from canari import Model, SKF
from canari.data_struct import LstmOutputHistory
from canari.component import (
    LocalAcceleration,
    LocalTrend,
    LstmNetwork,
    WhiteNoise,
)


# LSTM versoin of SKF
# Components
sigma_v = 5e-2
lstm_look_back_len = 10
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=lstm_look_back_len,
    num_features=2,
    num_layer=1,
    infer_len=24,
    num_hidden_unit=50,
    device="cpu",
    smoother=False,
)
slstm_netwotk = LstmNetwork(
    look_back_len=lstm_look_back_len,
    num_features=2,
    num_layer=1,
    infer_len=24,
    num_hidden_unit=50,
    device="cpu",
    smoother=True,
)
noise = WhiteNoise(std_error=sigma_v)

# Normal model with LSTM
lstm_model = Model(
    local_trend,
    lstm_network,
    noise,
)

# Normal model with SLSTM
slstm_model = Model(
    local_trend,
    slstm_netwotk,
    noise,
)

#  Abnormal model with LSTM
lstm_ab_model = Model(
    local_acceleration,
    lstm_network,
    WhiteNoise(),
)

#  Abnormal model with SLSTM
slstm_ab_model = Model(
    local_acceleration,
    slstm_netwotk,
    WhiteNoise(),
)

# Switching Kalman filter
skf_lstm = SKF(
    norm_model=lstm_model,
    abnorm_model=lstm_ab_model,
    std_transition_error=1e-4,
    norm_to_abnorm_prob=1e-4,
)
skf_lstm.save_initial_states()

skf_slstm = SKF(
    norm_model=slstm_model,
    abnorm_model=slstm_ab_model,
    std_transition_error=1e-4,
    norm_to_abnorm_prob=1e-4,
)
skf_slstm.save_initial_states()


@pytest.mark.parametrize("skf_version", [skf_lstm, skf_slstm], ids=["LSTM", "SLSTM"])
def test_skf_transition_models(skf_version):
    """Test construction of transition models for skf"""

    # Test transition matrices
    npt.assert_allclose(
        skf_version.model["norm_norm"].transition_matrix,
        skf_version.model["abnorm_norm"].transition_matrix,
    )
    npt.assert_allclose(
        skf_version.model["abnorm_abnorm"].transition_matrix,
        skf_version.model["norm_abnorm"].transition_matrix,
    )
    assert not np.allclose(
        skf_version.model["norm_norm"].transition_matrix,
        skf_version.model["abnorm_abnorm"].transition_matrix,
    )
    # Test observation matrices
    npt.assert_allclose(
        skf_version.model["norm_norm"].observation_matrix,
        skf_version.model["norm_norm"].observation_matrix,
    )
    npt.assert_allclose(
        skf_version.model["norm_abnorm"].observation_matrix,
        skf_version.model["abnorm_norm"].observation_matrix,
    )

    # Test process noise matrices
    idx_acc = skf_version.model["norm_norm"].get_states_index(
        states_name="acceleration"
    )
    assert (
        skf_version.model["norm_norm"].process_noise_matrix[idx_acc, idx_acc]
        == skf_version.model["abnorm_abnorm"].process_noise_matrix[idx_acc, idx_acc]
    )
    assert (
        skf_version.model["norm_norm"].process_noise_matrix[idx_acc, idx_acc]
        == skf_version.model["abnorm_norm"].process_noise_matrix[idx_acc, idx_acc]
    )
    assert not (
        skf_version.model["norm_norm"].process_noise_matrix[idx_acc, idx_acc]
        == skf_version.model["norm_abnorm"].process_noise_matrix[idx_acc, idx_acc]
    )

    idx_noise = skf_version.model["norm_norm"].get_states_index(
        states_name="white noise"
    )
    assert (
        skf_version.model["norm_norm"].process_noise_matrix[idx_noise, idx_noise]
        == skf_version.model["abnorm_abnorm"].process_noise_matrix[idx_noise, idx_noise]
    )
    assert (
        skf_version.model["norm_abnorm"].process_noise_matrix[idx_noise, idx_noise]
        == skf_version.model["abnorm_norm"].process_noise_matrix[idx_noise, idx_noise]
    )


@pytest.mark.parametrize("skf_version", [skf_lstm, skf_slstm], ids=["LSTM", "SLSTM"])
def test_skf_filter(skf_version):
    """Test SKF.filter"""

    new_mu_states = 0.1 * np.ones(skf_version.model["norm_norm"].mu_states.shape)
    new_var_states = 0.2 * np.ones(skf_version.model["norm_norm"].var_states.shape)
    skf_version.model["norm_norm"].set_states(new_mu_states, new_var_states)

    data = {}
    data["x"] = np.array([[0.1]])
    data["y"] = np.array([0.1])

    skf_version.filter(data=data)

    assert (
        skf_version.model["norm_norm"].var_obs_predict
        == skf_version.model["abnorm_norm"].var_obs_predict
    )

    # Check if lstm's memory is clear at at end of skf_version.filer
    if not skf_version.lstm_net.smooth:
        lstm_output_history_init = LstmOutputHistory()
        lstm_output_history_init.initialize(lstm_look_back_len)
        npt.assert_allclose(
            skf_version.model["norm_norm"].lstm_output_history.mu,
            lstm_output_history_init.mu,
        )
        npt.assert_allclose(
            skf_version.model["norm_norm"].lstm_output_history.var,
            lstm_output_history_init.var,
        )
        assert skf_version.marginal_prob["norm"] == skf_version.norm_model_prior_prob
        assert (
            skf_version.marginal_prob["abnorm"] == 1 - skf_version.norm_model_prior_prob
        )


@pytest.mark.parametrize("skf_version", [skf_lstm, skf_slstm], ids=["LSTM", "SLSTM"])
def test_detect_synthetic_anomaly(skf_version):
    """Test detect_synthetic_anomaly function"""

    data = {}
    data["x"] = np.array([[0.1]])
    data["x"] = np.tile(data["x"], (10, 1))
    data["y"] = np.array([[0.1]])
    data["y"] = np.tile(data["y"], (10, 1))

    np.random.seed(1)
    skf_version.detect_synthetic_anomaly(
        data=data,
        num_anomaly=1,
        slope_anomaly=0.01,
    )
    mu_1 = skf_version.model["norm_norm"].mu_states.copy()

    np.random.seed(1)
    skf_version.detect_synthetic_anomaly(
        data=data,
        num_anomaly=1,
        slope_anomaly=0.01,
    )
    mu_2 = skf_version.model["norm_norm"].mu_states.copy()

    npt.assert_allclose(
        mu_1,
        mu_2,
    )
