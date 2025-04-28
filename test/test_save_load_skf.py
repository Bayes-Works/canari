from canari import Model, SKF
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise
from test.test_save_load_model import compare_model_dict


# SKF model
skf = SKF(
    norm_model=Model(
        LocalTrend(),
        LstmNetwork(look_back_len=10, num_layer=2, num_hidden_unit=10),
        WhiteNoise(),
    ),
    abnorm_model=Model(
        LocalAcceleration(),
        LstmNetwork(look_back_len=10, num_layer=2, num_hidden_unit=10),
        WhiteNoise(),
    ),
    std_transition_error=1e-4,
    norm_to_abnorm_prob=1e-4,
    abnorm_to_norm_prob=1e-1,
    norm_model_prior_prob=0.99,
)


def test_skf_save_load():
    """Test save/load for SKF.py"""
    skf_dict = skf.get_dict()
    skf_loaded = SKF.load_dict(skf_dict)
    skf_loaded_dict = skf_loaded.get_dict()

    compare_model_dict(skf_dict["norm_model"], skf_loaded_dict["norm_model"])
    compare_model_dict(skf_dict["abnorm_model"], skf_loaded_dict["abnorm_model"])

    assert skf_dict["std_transition_error"] == skf_loaded_dict["std_transition_error"]
    assert skf_dict["norm_to_abnorm_prob"] == skf_loaded_dict["norm_to_abnorm_prob"]
    assert skf_dict["abnorm_to_norm_prob"] == skf_loaded_dict["abnorm_to_norm_prob"]
    assert skf_dict["norm_model_prior_prob"] == skf_loaded_dict["norm_model_prior_prob"]
