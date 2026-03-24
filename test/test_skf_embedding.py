import numpy as np
import pytest

from canari import Model, SKF
from canari.component import LocalAcceleration, LocalTrend, LstmNetwork, WhiteNoise


def _build_skf(embed_mode: str, smoother: bool, embed_len: int = 6):
    lstm_kwargs = dict(
        look_back_len=3,
        num_features=2,
        num_layer=1,
        infer_len=3,
        num_hidden_unit=8,
        device="cpu",
        smoother=smoother,
    )

    embed_mu = None
    embed_var = None

    if embed_mode == "set_embedding":
        embed_mu = (
            np.sin(np.linspace(0, 2 * np.pi, embed_len)).reshape(1, -1).astype(np.float32)
        )
        embed_var = (np.ones_like(embed_mu) * 0.1).astype(np.float32)
        lstm_kwargs["embedding"] = (embed_mu, embed_var)
    elif embed_mode == "init_embedding":
        lstm_kwargs["embed_len"] = embed_len
    else:
        raise ValueError("Invalid embed_mode")

    norm_model = Model(
        LocalTrend(),
        LstmNetwork(**lstm_kwargs),
        WhiteNoise(std_error=0.01),
    )
    abnorm_model = Model(
        LocalAcceleration(),
        LstmNetwork(**lstm_kwargs),
        WhiteNoise(std_error=0.01),
    )

    skf = SKF(
        norm_model=norm_model,
        abnorm_model=abnorm_model,
        std_transition_error=1e-4,
        norm_to_abnorm_prob=1e-4,
    )

    return skf, embed_mu, embed_var


def _small_data(n_step: int = 5):
    return {
        "x": np.tile(np.array([[0.1]], dtype=np.float32), (n_step, 1)),
        "y": np.linspace(0.0, 0.2, n_step, dtype=np.float32),
        "time": np.arange(n_step),
    }


@pytest.mark.parametrize("embed_mode", ["set_embedding", "init_embedding"])
@pytest.mark.parametrize("smoother", [False, True], ids=["LSTM", "SLSTM"])
def test_skf_filter_embedding(embed_mode, smoother):
    """Test that SKF.filter supports embedding-enabled LSTM inputs."""

    embed_len = 6
    skf, embed_mu, embed_var = _build_skf(embed_mode, smoother, embed_len=embed_len)

    skf.filter(data=_small_data(), update_embedding=False)

    lstm_embedding = skf.model["norm_norm"].lstm_embedding
    assert lstm_embedding.length == embed_len

    if embed_mode == "set_embedding":
        np.testing.assert_array_almost_equal(
            lstm_embedding.mu.reshape(1, -1),
            embed_mu,
            decimal=5,
            err_msg="SKF LSTM embedding mean changed unexpectedly when update_embedding=False",
        )
        np.testing.assert_array_almost_equal(
            lstm_embedding.var.reshape(1, -1),
            embed_var,
            decimal=5,
            err_msg="SKF LSTM embedding variance changed unexpectedly when update_embedding=False",
        )


@pytest.mark.parametrize("smoother", [False, True], ids=["LSTM", "SLSTM"])
def test_skf_filter_embedding_update_mask_validation(smoother):
    """Test update_mask validation for SKF embedding updates."""

    skf, _, _ = _build_skf("init_embedding", smoother, embed_len=4)

    with pytest.raises(ValueError, match="update_mask must be a 1D list/array"):
        skf.filter(
            data=_small_data(),
            update_embedding=True,
            update_mask=[1, 0],
        )


def test_skf_save_load_with_embedding():
    """Test SKF save/load keeps embedding values for the normal transition model."""

    skf, _, _ = _build_skf("init_embedding", smoother=False, embed_len=4)
    skf.model["norm_norm"].lstm_embedding.update(
        np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
        np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32),
    )
    skf.model["norm_norm"]._sync_lstm_embedding()

    saved = skf.get_dict()
    loaded = SKF.load_dict(saved)

    np.testing.assert_allclose(
        loaded.model["norm_norm"].lstm_embedding.mu,
        skf.model["norm_norm"].lstm_embedding.mu,
    )
    np.testing.assert_allclose(
        loaded.model["norm_norm"].lstm_embedding.var,
        skf.model["norm_norm"].lstm_embedding.var,
    )
