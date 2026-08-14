"""Train or test an anomaly-detection SKF with a global LSTM.

Set ``TRAIN_AND_OPTIMIZE_SKF`` below to choose the workflow, then run this
example from the repository root:

    python examples/anomaly_detection_global_lstm.py

All experiment settings are constants below; no configuration file is read.
"""

import multiprocessing as mp
import os
import pickle
from pathlib import Path

import pandas as pd
from pytagi import metric
from pytagi import Normalizer as normalizer
from ray import tune

from canari import DataProcess, Model, Optimizer, SKF
from canari.component import LocalAcceleration, LocalTrend, LstmNetwork, WhiteNoise

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/ts_weekly_values.csv"
DATETIME_PATH = ROOT / "data/ts_weekly_datetimes.csv"
GLOBAL_LSTM_PATH = ROOT / "saved_params/global_model.bin"
SAVED_SKF_PATH = ROOT / "saved_params/anomaly_detection_lstm_finetuned.pkl"

# True: fine-tune the LSTM, optimize and save the SKF, then reload and test it.
# False: load the saved SKF and test it directly.
TRAIN_AND_OPTIMIZE_SKF = True

SERIES_NAME = "Sensor01"
VALIDATION_START = "2013-04-21"
TEST_START = "2015-04-19"

LOOK_BACK_LEN = 52
INFER_LEN = 3 * 52
NUM_HIDDEN_UNITS = 256
NUM_EPOCHS = 100
BASELINE_INIT_LEN = 2 * 52
USE_SLSTM = False

SIGMA_V_GRID = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
GRID_SEARCH_N_JOBS = 1  # -1 uses all available CPU cores
SKF_OPTIMIZATION_TRIALS = 200
SKF_STARTUP_TRIALS = 100
NUM_SYNTHETIC_ANOMALIES = 50
NUM_EVALUATION_REALIZATIONS = 25
MAX_DETECTION_STEPS = 3 * 52
ANOMALY_SLOPES = [0.025, 0.05, 0.075, 0.225, 0.5, 0.75, 1.0]
SKF_N_JOBS = 1  # -1 uses all available CPU cores, used for the multiple realizations.


def prepare_data():
    values = pd.read_csv(DATA_PATH)[SERIES_NAME]
    datetimes = pd.to_datetime(pd.read_csv(DATETIME_PATH)[SERIES_NAME])

    last_valid_index = values.last_valid_index()
    values = values.iloc[: last_valid_index + 1]
    datetimes = datetimes.iloc[: last_valid_index + 1]
    dataframe = pd.DataFrame(
        {"values": values.to_numpy()},
        index=pd.DatetimeIndex(datetimes, name="date_time"),
    )
    return DataProcess(
        data=dataframe,
        time_covariates=["week_of_year"],
        validation_start=VALIDATION_START,
        test_start=TEST_START,
        output_col=[0],
    )


def finetune_lstm(sigma_v, data_processor, train_data, validation_data):
    model = Model(
        LocalTrend(),
        LstmNetwork(
            look_back_len=LOOK_BACK_LEN,
            num_features=2,
            infer_len=INFER_LEN,
            num_layer=1,
            num_hidden_unit=NUM_HIDDEN_UNITS,
            device="cpu",
            manual_seed=1,
            smoother=USE_SLSTM,
        ),
        WhiteNoise(std_error=sigma_v),
    )

    # Transfer the global means and keep fresh variances for local fine-tuning.
    model.load_lstm_parameter_means(GLOBAL_LSTM_PATH)
    model.auto_initialize_baseline_states(train_data["y"][:BASELINE_INIT_LEN])
    model.mu_states[model.get_states_index("trend")] = 0.0

    validation_observations = data_processor.get_data("validation").flatten()
    for epoch in range(NUM_EPOCHS):
        model.lstm_net.num_samples = len(train_data["y"]) + len(validation_data["y"])
        validation_mean, validation_std, _ = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
            white_noise_decay=False,
        )

        validation_mean = normalizer.unstandardize(
            validation_mean,
            data_processor.scale_const_mean[data_processor.output_col],
            data_processor.scale_const_std[data_processor.output_col],
        )
        validation_std = normalizer.unstandardize_std(
            validation_std,
            data_processor.scale_const_std[data_processor.output_col],
        )
        validation_log_likelihood = float(
            metric.log_likelihood(
                validation_mean, validation_observations, validation_std
            )
        )
        model.early_stopping(
            evaluate_metric=validation_log_likelihood,
            current_epoch=epoch,
            max_epoch=NUM_EPOCHS,
            mode="max",
            skip_epoch=0,
        )
        if model.stop_training:
            break

    return model, float(model.early_stop_metric)


def finetune_for_sigma_v(arguments):
    """Fine-tune one grid point. Returns the model as a dictionary because `Model` holds a
    `cutagi.Sequential`, which cannot be pickled back from a worker process."""

    sigma_v, data_processor, train_data, validation_data = arguments
    model, validation_log_likelihood = finetune_lstm(
        sigma_v,
        data_processor,
        train_data,
        validation_data,
    )
    return sigma_v, validation_log_likelihood, model.get_dict(time_step=0)


def skf_with_parameters(parameters, model_input):
    normal_model = Model.load_dict(model_input["normal_model"])
    abnormal_model = Model(
        LocalAcceleration(),
        LstmNetwork(),
        WhiteNoise(std_error=model_input["sigma_v"]),
    )
    skf = SKF(
        norm_model=normal_model,
        abnorm_model=abnormal_model,
        std_transition_error=parameters["std_transition_error"],
        norm_to_abnorm_prob=parameters["norm_to_abnorm_prob"],
        abnorm_to_norm_prob=parameters["abnorm_to_norm_prob"],
    )
    skf.model["norm_norm"].lstm_net.teacher_forcing = False
    skf.save_initial_states()

    slope = float(parameters["slope"])
    synthetic_data = DataProcess.add_synthetic_anomaly(
        model_input["train_validation_data"],
        num_samples=NUM_SYNTHETIC_ANOMALIES,
        slope=[slope / 52, -slope / 52],
        anomaly_start=0.25,
        anomaly_end=0.75,
    )
    detection_rate, false_alarms = skf.detect_synthetic_anomaly(
        data=model_input["train_validation_data"],
        synthetic_data=synthetic_data,
        threshold=parameters["threshold"],
        max_timestep_to_detect=MAX_DETECTION_STEPS,
        # Kept sequential: this runs inside a Ray Tune trial, and forking a worker
        # pool from a Ray worker crashes it.
        n_jobs=1,
    )
    false_alarms_per_year = false_alarms / model_input["data_length_years"]

    skf.metric_optim = skf.objective(detection_rate, false_alarms_per_year, slope)
    skf.print_metric = {
        "detection_rate": detection_rate,
        "false_alarms_per_year": false_alarms_per_year,
    }
    skf.load_initial_states()
    return skf


def main():
    data_processor = prepare_data()
    train_data, validation_data, _, all_data = data_processor.get_splits()

    if TRAIN_AND_OPTIMIZE_SKF:
        grid_arguments = [
            (sigma_v, data_processor, train_data, validation_data)
            for sigma_v in SIGMA_V_GRID
        ]
        n_jobs = os.cpu_count() if GRID_SEARCH_N_JOBS == -1 else GRID_SEARCH_N_JOBS
        if n_jobs > 1:
            context = mp.get_context("fork")
            with context.Pool(n_jobs) as pool:
                grid_results = pool.map(finetune_for_sigma_v, grid_arguments)
                pool.close()
                pool.join()
        else:
            grid_results = [finetune_for_sigma_v(a) for a in grid_arguments]

        for sigma_v, validation_log_likelihood, _ in grid_results:
            print(
                f"sigma_v={sigma_v:.3f}: "
                f"validation log-likelihood={validation_log_likelihood:.4f}"
            )
        best_sigma_v, _, best_model_dict = max(
            grid_results, key=lambda result: result[1]
        )

        print(f"Best sigma_v: {best_sigma_v}")
        train_validation_data = data_processor.get_splits(split="train_val")
        data_length_years = (
            train_validation_data["time"][-1] - train_validation_data["time"][0]
        ).days / 365.25
        model_input = {
            "normal_model": best_model_dict,
            "sigma_v": best_sigma_v,
            "train_validation_data": train_validation_data,
            "data_length_years": data_length_years,
        }

        parameter_space = {
            "std_transition_error": tune.loguniform(1e-6, 1e-4),
            "norm_to_abnorm_prob": tune.loguniform(1e-6, 1e-4),
            "abnorm_to_norm_prob": tune.quniform(0.10, 0.20, 0.01),
            "threshold": tune.quniform(0.05, 0.50, 0.01),
            "slope": tune.choice(ANOMALY_SLOPES),
        }
        optimizer = Optimizer(
            model=skf_with_parameters,
            param=parameter_space,
            model_input=model_input,
            num_optimization_trial=SKF_OPTIMIZATION_TRIALS,
            mode="max",
            num_startup_trials=SKF_STARTUP_TRIALS,
            max_concurrent=1,  # allows for more informative sequential trials
        )
        optimizer.optimize()
        best_skf_parameters = optimizer.get_best_param()
        best_skf = skf_with_parameters(best_skf_parameters, model_input)

        # Save before testing so both workflows test a model loaded from disk.
        saved_model = best_skf.get_dict()
        saved_model["model_param"] = {"sigma_v": best_sigma_v}
        saved_model["skf_param"] = best_skf_parameters
        saved_model["threshold"] = best_skf_parameters["threshold"]
        saved_model["cov_names"] = train_data["cov_names"]

        SAVED_SKF_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SAVED_SKF_PATH.open("wb") as file:
            pickle.dump(saved_model, file)
        print(f"Saved complete fine-tuned SKF model to {SAVED_SKF_PATH}")

    if not SAVED_SKF_PATH.exists():
        raise FileNotFoundError(
            f"No saved SKF found at {SAVED_SKF_PATH}. "
            "Set TRAIN_AND_OPTIMIZE_SKF = True first."
        )

    with SAVED_SKF_PATH.open("rb") as file:
        saved_model = pickle.load(file)
    best_skf = SKF.load_dict(saved_model)
    best_skf.save_initial_states()
    best_skf_parameters = saved_model["skf_param"]
    print(f"Loaded complete fine-tuned SKF model from {SAVED_SKF_PATH}")

    evaluation_results = {}
    num_steps = len(all_data["y"])
    anomaly_start = data_processor.test_start / num_steps
    anomaly_end = (data_processor.test_end - MAX_DETECTION_STEPS) / num_steps
    data_length_years = (
        data_processor.data.index[data_processor.test_end - 1]
        - data_processor.data.index[data_processor.train_start]
    ).days / 365.25

    print("Multi-realization evaluation:")
    for slope in ANOMALY_SLOPES:
        synthetic_data = DataProcess.add_synthetic_anomaly(
            all_data,
            num_samples=NUM_EVALUATION_REALIZATIONS,
            slope=[slope / 52, -slope / 52],
            anomaly_start=anomaly_start,
            anomaly_end=anomaly_end,
        )
        detection_rate, false_alarms = best_skf.detect_synthetic_anomaly(
            data=all_data,
            synthetic_data=synthetic_data,
            threshold=best_skf_parameters["threshold"],
            max_timestep_to_detect=MAX_DETECTION_STEPS,
            n_jobs=SKF_N_JOBS,
        )
        evaluation_results[slope] = {
            "detection_rate": detection_rate,
            "false_alarms_per_year": false_alarms / data_length_years,
            "num_realizations": len(synthetic_data),
        }
        result = evaluation_results[slope]
        print(
            f"  slope={slope:.3f}: detection={result['detection_rate']:.2f}, "
            f"false alarms/year={result['false_alarms_per_year']:.2f}"
        )

    if TRAIN_AND_OPTIMIZE_SKF:
        saved_model["multi_realization_evaluation"] = evaluation_results
        with SAVED_SKF_PATH.open("wb") as file:
            pickle.dump(saved_model, file)


if __name__ == "__main__":
    main()
