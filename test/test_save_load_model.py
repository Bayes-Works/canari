import numpy.testing as npt
from canari import Model
from canari.component import LocalTrend, LstmNetwork, WhiteNoise

# Components
lstm_look_back_len = 10
lstm_network_1 = LstmNetwork(
    look_back_len=lstm_look_back_len,
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
    manual_seed=1,
)
lstm_network_2 = LstmNetwork(
    look_back_len=lstm_look_back_len,
    num_features=2,
    num_layer=1,
    num_hidden_unit=50,
    device="cpu",
    manual_seed=2,
)

# Model
model1 = Model(
    LocalTrend(),
    lstm_network_1,
    WhiteNoise(),
)

model2 = Model(
    LocalTrend(),
    lstm_network_2,
    WhiteNoise(),
)


def compare_model_dict(model_1_dict, model_2_dict):
    """Test save/load for model.py"""

    for component_1, component_2 in zip(
        model_1_dict["components"].values(), model_2_dict["components"].values()
    ):
        npt.assert_allclose(component_1.mu_states, component_2.mu_states)
        npt.assert_allclose(component_1.var_states, component_2.var_states)
        npt.assert_allclose(
            component_1.transition_matrix, component_2.transition_matrix
        )
        npt.assert_allclose(
            component_1.process_noise_matrix, component_2.process_noise_matrix
        )
        npt.assert_allclose(
            component_1.observation_matrix, component_2.observation_matrix
        )
        assert component_1.states_name == component_2.states_name

    npt.assert_allclose(model_1_dict["mu_states"], model_2_dict["mu_states"])
    npt.assert_allclose(model_1_dict["var_states"], model_2_dict["var_states"])

    if "lstm_network_params" in model_1_dict and "lstm_network_params" in model_2_dict:
        assert (
            model_1_dict["lstm_network_params"] == model_2_dict["lstm_network_params"]
        )


def compare_lstm_dict(model_1_dict, model_2_dict):
    """Test lstm initialization with different seeds"""
    assert (
        not model_1_dict["lstm_network_params"] == model_2_dict["lstm_network_params"]
    )


def test_model_save_load():
    """Test save/load for model.py"""
    model1_dict = model1.get_dict()
    model1_loaded = Model.load_dict(model1_dict)
    model1_loaded_dict = model1_loaded.get_dict()
    compare_model_dict(model1_dict, model1_loaded_dict)

    model2_dict = model2.get_dict()
    compare_lstm_dict(model1_dict, model2_dict)
