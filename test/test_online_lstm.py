import numpy as np

from canari import Model, SKF
from canari.component import LocalAcceleration, LocalTrend, LstmNetwork, WhiteNoise


def make_data(num_steps=20):
    return {
        "x": np.empty((num_steps, 0), dtype=np.float32),
        "y": np.sin(np.arange(num_steps, dtype=np.float32)).reshape(-1, 1),
    }


def parameters_changed(before, after):
    return any(
        not np.array_equal(np.asarray(old), np.asarray(new))
        for layer in before
        for old, new in zip(before[layer], after[layer])
    )


def make_lstm_model():
    return Model(
        LstmNetwork(
            look_back_len=1,
            num_features=1,
            num_hidden_unit=4,
            manual_seed=1,
        ),
        WhiteNoise(std_error=0.1),
    )


def make_skf():
    return SKF(
        Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=1,
                num_features=1,
                num_hidden_unit=4,
                manual_seed=1,
            ),
            WhiteNoise(std_error=0.1),
        ),
        Model(
            LocalAcceleration(),
            LstmNetwork(),
            WhiteNoise(std_error=0.1),
        ),
        std_transition_error=1e-3,
        norm_to_abnorm_prob=1e-3,
    )


def test_model_online_lstm_filter_updates_parameters():
    model = make_lstm_model()
    parameters_before = model.lstm_net.state_dict()

    mean, std, states = model.online_lstm_filter(
        make_data(), start=10, end=13, window_len=3
    )

    assert len(mean) == len(std) == len(states.mu_prior) == 3
    assert np.all(np.isfinite(mean))
    assert np.all(std > 0)
    assert parameters_changed(parameters_before, model.lstm_net.state_dict())


def test_skf_online_lstm_filter_updates_parameters():
    skf = make_skf()
    parameters_before = skf.lstm_net.state_dict()

    anomaly_probability, states = skf.online_lstm_filter(
        make_data(), end=13, window_len=3
    )

    assert len(anomaly_probability) == len(states.mu_prior) == 13
    assert np.all((0 <= anomaly_probability) & (anomaly_probability <= 1))
    assert parameters_changed(parameters_before, skf.lstm_net.state_dict())
