import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pytagi import metric

from examples import ood_experiment as experiment
from canari import DataProcess, Model
from canari.component import LstmNetwork, WhiteNoise


def make_data(num_steps):
    return {
        "x": np.zeros((num_steps, 1), dtype=np.float32),
        "y": np.arange(num_steps, dtype=np.float32).reshape(-1, 1),
    }


def make_results():
    return [
        experiment.ScenarioResult(
            scenario=scenario,
            mean=np.array([0.5, 1.5]),
            std=np.ones(2),
            mse=0.25,
            mean_log_likelihood=-1.0,
            n_observed=2,
            offline_epochs=100 if scenario.offline_pretraining else 0,
        )
        for scenario in experiment.SCENARIOS
    ]


def test_scenario_matrix_matches_requested_initialization_and_training_modes():
    assert [scenario.initialization for scenario in experiment.SCENARIOS] == [
        "local",
        "global",
        "local",
        "global",
        "local",
        "global",
    ]
    assert [scenario.offline_pretraining for scenario in experiment.SCENARIOS] == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert [scenario.validation_mode for scenario in experiment.SCENARIOS] == [
        "frozen",
        "frozen",
        "online",
        "online",
        "online",
        "online",
    ]
    assert [scenario.online_start for scenario in experiment.SCENARIOS] == [
        None,
        None,
        "validation",
        "validation",
        "beginning",
        "beginning",
    ]


def test_build_data_processor_uses_two_52_week_years_and_no_test_split():
    index = pd.date_range("2020-01-05", periods=120, freq="W-SUN")
    series = pd.DataFrame({experiment.SERIES: np.arange(120)}, index=index)

    processor = experiment.build_data_processor(series)
    train, validation, test, all_data = processor.get_splits()

    assert processor.validation_start == experiment.TRAIN_STEPS == 104
    assert processor.validation_end == len(series)
    assert len(train["y"]) == 104
    assert len(validation["y"]) == 16
    assert len(test["y"]) == 0
    assert len(all_data["y"]) == len(series)


def test_score_validation_uses_one_common_observed_mask():
    observations = np.array([1.0, np.nan, 3.0])
    mean = np.array([0.0, np.nan, 2.0])
    std = np.array([1.0, np.nan, 2.0])

    scores = experiment.score_validation(mean, std, observations)

    observed = np.array([True, False, True])
    assert scores["n_observed"] == 2
    assert scores["mse"] == pytest.approx(
        metric.mse(mean[observed], observations[observed])
    )
    assert scores["mean_log_likelihood"] == pytest.approx(
        metric.log_likelihood(mean[observed], observations[observed], std[observed])
    )


@pytest.mark.parametrize(
    ("mean", "std"),
    [
        (np.array([0.0, np.nan]), np.ones(2)),
        (np.zeros(2), np.array([1.0, 0.0])),
    ],
)
def test_score_validation_rejects_invalid_predictions(mean, std):
    with pytest.raises(ValueError):
        experiment.score_validation(mean, std, np.ones(2))


def test_frozen_validation_replays_train_and_filters_validation_without_training():
    class FakeNetwork:
        def __init__(self):
            self.num_samples = None
            self.evaluating = False

        def eval(self):
            self.evaluating = True

    class FakeModel:
        def __init__(self):
            self.lstm_net = FakeNetwork()
            self.calls = []

        def filter(self, data, train_lstm):
            self.calls.append((len(data["y"]), train_lstm))
            length = len(data["y"])
            return np.arange(length), np.ones(length), None

    model = FakeModel()
    mean, std = experiment.run_frozen_validation(
        model, train_data=make_data(4), validation_data=make_data(3)
    )

    assert model.lstm_net.num_samples == 7
    assert model.lstm_net.evaluating
    assert model.calls == [(4, False), (3, False)]
    np.testing.assert_array_equal(mean, np.arange(3))
    np.testing.assert_array_equal(std, np.ones(3))


@pytest.mark.parametrize(
    ("online_start", "expected_start"),
    [("validation", 104), ("beginning", experiment.ONLINE_WINDOW_LEN)],
)
def test_online_predictions_are_sliced_to_the_validation_span(
    online_start, expected_start
):
    class FakeModel:
        def __init__(self):
            self.call = None

        def online_lstm_filter(self, data, start, window_len, end):
            self.call = (start, window_len, end)
            return np.arange(start, end), np.ones(end - start), None

    model = FakeModel()
    mean, std = experiment.run_online_validation(
        model,
        all_data=make_data(110),
        validation_start=104,
        validation_end=110,
        online_start=online_start,
    )

    assert model.call == (expected_start, experiment.ONLINE_WINDOW_LEN, 110)
    np.testing.assert_array_equal(mean, np.arange(104, 110))
    np.testing.assert_array_equal(std, np.ones(6))


def test_offline_training_never_receives_ood_validation(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.validation_lengths = []
            self.memory_steps = []

        def lstm_train(self, train_data, validation_data, data_processor):
            self.validation_lengths.append(len(validation_data["y"]))
            return np.array([]), np.array([]), object()

        def set_memory(self, states, time_step):
            self.memory_steps.append(time_step)

        def get_dict(self):
            return {"trained": True}

    model = FakeModel()
    monkeypatch.setattr(experiment, "build_model", lambda initialization: model)

    snapshot = experiment.train_offline(
        "local",
        train_data=make_data(104),
        data_processor=SimpleNamespace(),
        num_epochs=2,
    )

    assert model.validation_lengths == [0, 0]
    assert model.memory_steps == [0, 0]
    assert snapshot == {"trained": True}


def test_real_model_snapshot_supports_frozen_and_online_validation(monkeypatch):
    index = pd.date_range("2024-01-07", periods=16, freq="W-SUN")
    series = pd.DataFrame({"y": np.sin(np.arange(16) / 2)}, index=index)
    processor = DataProcess(
        series,
        time_covariates=["week_of_year"],
        train_split=0.5,
        validation_split=0.5,
        output_col=[0],
    )
    train, validation, _, all_data = processor.get_splits()

    def tiny_model(initialization="local"):
        return Model(
            LstmNetwork(
                look_back_len=3,
                num_features=2,
                infer_len=2,
                num_hidden_unit=4,
                num_layer=1,
                manual_seed=1,
            ),
            WhiteNoise(std_error=0.1),
        )

    def parameters_changed(before, after):
        return any(
            not np.array_equal(np.asarray(old), np.asarray(new))
            for layer in before
            for old, new in zip(before[layer], after[layer])
        )

    monkeypatch.setattr(experiment, "build_model", tiny_model)
    monkeypatch.setattr(experiment, "ONLINE_WINDOW_LEN", 3)
    snapshot = experiment.train_offline("local", train, processor, num_epochs=1)

    frozen_model = Model.load_dict(copy.deepcopy(snapshot))
    frozen_before = copy.deepcopy(frozen_model.lstm_net.state_dict())
    frozen_mean, frozen_std = experiment.run_frozen_validation(
        frozen_model, train, validation
    )
    assert not parameters_changed(frozen_before, frozen_model.lstm_net.state_dict())

    online_model = Model.load_dict(copy.deepcopy(snapshot))
    online_before = copy.deepcopy(online_model.lstm_net.state_dict())
    online_mean, online_std = experiment.run_online_validation(
        online_model,
        all_data,
        validation_start=processor.validation_start,
        validation_end=processor.validation_end,
        online_start="validation",
    )
    assert parameters_changed(online_before, online_model.lstm_net.state_dict())

    for mean, std in ((frozen_mean, frozen_std), (online_mean, online_std)):
        assert len(mean) == len(std) == len(validation["y"])
        assert np.all(np.isfinite(mean))
        assert np.all(std > 0)


def test_write_results_routes_each_scenario_to_its_own_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(experiment, "plot_scenario", lambda *args: None)
    monkeypatch.setattr(experiment, "plot_residual_comparison", lambda *args: None)
    dates = pd.date_range("2020-01-05", periods=2, freq="W-SUN")

    summary = experiment.write_results(
        make_results(), dates, observations=np.array([0.0, 1.0])
    )

    assert len(summary) == 6
    for scenario in experiment.SCENARIOS:
        assert (scenario.output_dir / "predictions.csv").is_file()
        assert (scenario.output_dir / "metrics.csv").is_file()
    assert (tmp_path / "summary/metrics.csv").is_file()
    assert (tmp_path / "summary/metrics.md").is_file()
