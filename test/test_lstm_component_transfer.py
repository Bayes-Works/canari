import numpy as np
import numpy.testing as npt
import pytest

from canari.component import LstmNetwork


@pytest.fixture
def state_dicts():
    original_params = {
        "LSTM.0": (
            np.array([[1.0]], dtype=np.float32),
            np.array([[0.1]], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            np.array([0.01], dtype=np.float32),
        ),
        "Linear.1": (
            np.array([[2.0]], dtype=np.float32),
            np.array([[0.2]], dtype=np.float32),
            np.array([2.0], dtype=np.float32),
            np.array([0.02], dtype=np.float32),
        ),
    }
    loaded_params = {
        "LSTM.0": (
            np.array([[3.0]], dtype=np.float32),
            np.array([[0.3]], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
            np.array([0.03], dtype=np.float32),
        ),
        "Linear.1": (
            np.array([[4.0]], dtype=np.float32),
            np.array([[0.4]], dtype=np.float32),
            np.array([4.0], dtype=np.float32),
            np.array([0.04], dtype=np.float32),
        ),
    }
    return original_params, loaded_params


def test_transfer_loaded_params_reinitializes_output_layer_for_finetune(state_dicts):
    original_params, loaded_params = state_dicts

    new_params = LstmNetwork._transfer_loaded_params(
        original_params,
        loaded_params,
        finetune=True,
        increase_output_variance=False,
    )

    assert new_params["LSTM.0"] == loaded_params["LSTM.0"]
    assert new_params["Linear.1"] == original_params["Linear.1"]


def test_transfer_loaded_params_increases_output_layer_variance(state_dicts):
    original_params, loaded_params = state_dicts

    new_params = LstmNetwork._transfer_loaded_params(
        original_params,
        loaded_params,
        finetune=False,
        increase_output_variance=True,
    )

    assert new_params["LSTM.0"] == loaded_params["LSTM.0"]
    npt.assert_allclose(new_params["Linear.1"][0], loaded_params["Linear.1"][0])
    npt.assert_allclose(new_params["Linear.1"][1], loaded_params["Linear.1"][1] * 1.5)
    npt.assert_allclose(new_params["Linear.1"][2], loaded_params["Linear.1"][2])
    npt.assert_allclose(new_params["Linear.1"][3], loaded_params["Linear.1"][3] * 1.5)


def test_transfer_loaded_params_rejects_conflicting_transfer_modes(state_dicts):
    original_params, loaded_params = state_dicts

    with pytest.raises(ValueError, match="mutually exclusive"):
        LstmNetwork._transfer_loaded_params(
            original_params,
            loaded_params,
            finetune=True,
            increase_output_variance=True,
        )


def test_transfer_loaded_params_increases_output_layer_variance_for_sequences():
    original_params = {
        "LSTM.0": ([1.0], [0.1], [1.0], [0.01]),
        "Linear.1": ([2.0], [[0.2, 0.3]], [2.0], [0.02, 0.03]),
    }
    loaded_params = {
        "LSTM.0": ([3.0], [0.3], [3.0], [0.03]),
        "Linear.1": ([4.0], [[0.4, 0.5]], [4.0], [0.04, 0.05]),
    }

    new_params = LstmNetwork._transfer_loaded_params(
        original_params,
        loaded_params,
        finetune=False,
        increase_output_variance=True,
    )

    npt.assert_allclose(new_params["Linear.1"][1], [[0.6, 0.75]])
    npt.assert_allclose(new_params["Linear.1"][3], [0.06, 0.075])
