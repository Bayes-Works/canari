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


# Components
sigma_v = 5e-2
lstm_look_back_len = 10
local_trend = LocalTrend()
local_acceleration = LocalAcceleration()
lstm_network = LstmNetwork(
    look_back_len=lstm_look_back_len,
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
)
noise = WhiteNoise(std_error=sigma_v)

# Normal model
model = Model(
    local_trend,
    lstm_network,
    noise,
)

#  Abnormal model
ab_model = Model(
    local_acceleration,
    lstm_network,
    WhiteNoise(),
)

# Switching Kalman filter
skf = SKF(
    norm_model=model,
    abnorm_model=ab_model,
    std_transition_error=1e-4,
    norm_to_abnorm_prob=1e-4,
    abnorm_to_norm_prob=1e-1,
    norm_model_prior_prob=0.99,
)


def test_skf_transition_models():
    """Test construction of transition models for skf"""

    # Test transition matrices
    npt.assert_allclose(
        skf.model["norm_norm"].transition_matrix,
        skf.model["abnorm_norm"].transition_matrix,
    )
    npt.assert_allclose(
        skf.model["abnorm_abnorm"].transition_matrix,
        skf.model["norm_abnorm"].transition_matrix,
    )
    assert not np.allclose(
        skf.model["norm_norm"].transition_matrix,
        skf.model["abnorm_abnorm"].transition_matrix,
    )
    # Test observation matrices
    npt.assert_allclose(
        skf.model["norm_norm"].observation_matrix,
        skf.model["norm_norm"].observation_matrix,
    )
    npt.assert_allclose(
        skf.model["norm_abnorm"].observation_matrix,
        skf.model["abnorm_norm"].observation_matrix,
    )

    # Test process noise matrices
    idx_acc = skf.model["norm_norm"].get_states_index(states_name="local acceleration")
    assert (
        skf.model["norm_norm"].process_noise_matrix[idx_acc, idx_acc]
        == skf.model["abnorm_abnorm"].process_noise_matrix[idx_acc, idx_acc]
    )
    assert (
        skf.model["norm_norm"].process_noise_matrix[idx_acc, idx_acc]
        == skf.model["abnorm_norm"].process_noise_matrix[idx_acc, idx_acc]
    )
    assert not (
        skf.model["norm_norm"].process_noise_matrix[idx_acc, idx_acc]
        == skf.model["norm_abnorm"].process_noise_matrix[idx_acc, idx_acc]
    )

    idx_noise = skf.model["norm_norm"].get_states_index(states_name="white noise")
    assert (
        skf.model["norm_norm"].process_noise_matrix[idx_noise, idx_noise]
        == skf.model["abnorm_abnorm"].process_noise_matrix[idx_noise, idx_noise]
    )
    assert (
        skf.model["norm_abnorm"].process_noise_matrix[idx_noise, idx_noise]
        == skf.model["abnorm_norm"].process_noise_matrix[idx_noise, idx_noise]
    )


def test_skf_filter():
    """Test SKF.filter"""

    # Test same hidden states for each transition model:
    # or: self.set_same_states_transition_models()
    # Check if the first predictions from "norm_norm" and "abnorm_norm" are the same

    new_mu_states = 0.1 * np.ones(skf.model["norm_norm"].mu_states.shape)
    new_var_states = 0.2 * np.ones(skf.model["norm_norm"].var_states.shape)
    skf.model["norm_norm"].set_states(new_mu_states, new_var_states)

    data = {}
    data["x"] = np.array([[0.1]])
    data["y"] = np.array([0.1])

    skf.filter(data=data)

    assert (
        skf.model["norm_norm"].var_obs_predict
        == skf.model["abnorm_norm"].var_obs_predict
    )

    # Check if lstm's memory is clear at at end of SKF.filer
    lstm_output_history_init = LstmOutputHistory()
    lstm_output_history_init.initialize(lstm_look_back_len)
    npt.assert_allclose(skf.lstm_output_history.mu, lstm_output_history_init.mu)
    npt.assert_allclose(skf.lstm_output_history.var, lstm_output_history_init.var)
    assert skf.marginal_prob_current["norm"] == skf.norm_model_prior_prob
    assert skf.marginal_prob_current["abnorm"] == 1 - skf.norm_model_prior_prob


def test_detect_synthetic_anomaly():
    """Test detect_synthetic_anomaly function"""

    data = {}
    data["x"] = np.array([[0.1]])
    data["x"] = np.tile(data["x"], (10, 1))
    data["y"] = np.array([[0.1]])
    data["y"] = np.tile(data["y"], (10, 1))

    np.random.seed(1)
    skf.detect_synthetic_anomaly(
        data=data,
        num_anomaly=1,
        slope_anomaly=0.01,
    )
    mu_1 = skf.model["norm_norm"].mu_states.copy()

    np.random.seed(1)
    skf.detect_synthetic_anomaly(
        data=data,
        num_anomaly=1,
        slope_anomaly=0.01,
    )
    mu_2 = skf.model["norm_norm"].mu_states.copy()

    npt.assert_allclose(
        mu_1,
        mu_2,
    )
