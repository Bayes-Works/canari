"""Retrain one LSTM per seed (local condition), run SKF filter on the same
deterministic synthetic-anomaly realization, and save per-step P(abnormal).

Outputs one .npz per seed in experiments/out/local_seed_reproducibility/, plus
a metadata JSON. Intended to feed the beamer notebook
experiments/notebooks/local_seed_reproducibility_beamer.ipynb.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml
from pytagi import Normalizer as normalizer
from pytagi import metric
from scipy.stats import norm as _norm_dist

from canari import Model, SKF
from canari.component import (
    LocalAcceleration,
    LocalTrend,
    LstmNetwork,
    WhiteNoise,
)
from canari.data_process import DataProcess

try:
    from experiments.utils import prepare_dataset
except ModuleNotFoundError:
    from utils import prepare_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]

MODE = "global_ltu"  # "local", "global" (LGA008), or "global_ltu" (LTU009)
SEEDS = [1, 2, 3]
MAGNITUDE = 0.15
MAX_EPOCHS = 100
LSTM_NUM_THREAD = 1
SIGMA_V_OVERRIDE = None  # set to None to use per-seed optimum

if MODE == "local":
    CONFIG_PATH = REPO_ROOT / "experiments/config/ID_timeseries/LTU009EFAPRG024.yaml"
    BENCHMARK_RUNS_DIR = (
        REPO_ROOT
        / "experiments/out"
        / "Experiment_LTU009EFAPRG024_benchmark"
        / "runs"
    )
    BENCHMARK_RUN_PREFIX = "Experiment_LTU009EFAPRG024_benchmark_local"
    OUTPUT_DIR = REPO_ROOT / "experiments/out/local_seed_reproducibility"
elif MODE == "global":
    BENCHMARK_RUNS_DIR = (
        REPO_ROOT
        / "experiments/out"
        / "experiment_lstm_LGA008EFAPRG910_global_benchmark"
        / "runs"
    )
    BENCHMARK_RUN_PREFIX = "experiment_lstm_LGA008EFAPRG910_global_benchmark_global"
    CONFIG_PATH = (
        BENCHMARK_RUNS_DIR
        / f"{BENCHMARK_RUN_PREFIX}_seed{SEEDS[0]}"
        / "experiment_config_used.yaml"
    )
    OUTPUT_DIR = REPO_ROOT / "experiments/out/global_seed_reproducibility"
elif MODE == "global_ltu":
    BENCHMARK_RUNS_DIR = (
        REPO_ROOT
        / "experiments/out"
        / "Experiment_LTU009EFAPRG024_benchmark"
        / "runs"
    )
    BENCHMARK_RUN_PREFIX = "Experiment_LTU009EFAPRG024_benchmark_global"
    CONFIG_PATH = (
        BENCHMARK_RUNS_DIR
        / f"{BENCHMARK_RUN_PREFIX}_seed{SEEDS[0]}"
        / "experiment_config_used.yaml"
    )
    OUTPUT_DIR = REPO_ROOT / "experiments/out/global_ltu_seed_reproducibility"
else:
    raise ValueError(f"Unknown MODE: {MODE}")


def _crps_gaussian(mu: np.ndarray, std: np.ndarray, obs: np.ndarray) -> float:
    z = (obs - mu) / std
    return float(
        np.nanmean(
            std * (z * (2 * _norm_dist.cdf(z) - 1) + 2 * _norm_dist.pdf(z) - 1.0 / np.sqrt(np.pi))
        )
    )


def _load_seed_params(seed: int) -> dict:
    seed_dir = BENCHMARK_RUNS_DIR / f"{BENCHMARK_RUN_PREFIX}_seed{seed}"
    payload = json.loads((seed_dir / "summary.json").read_text())
    params = dict(payload["model_parameters_used"])
    if "threshold" not in params:
        cfg = yaml.safe_load((seed_dir / "experiment_config_used.yaml").read_text())
        params["threshold"] = float(cfg["anomaly_detection_threshold"])
    return params


def _train_lstm(base_config: dict, seed: int, sigma_v: float, dataset: dict) -> Model:
    cfg = base_config
    train_data = dataset["train_data"]
    validation_data = dataset["validation_data"]
    data_processor = dataset["data_processor"]
    warmup_lookback_mu = dataset["warmup_lookback_mu"]
    warmup_lookback_var = dataset["warmup_lookback_var"]

    early_stop_metric = cfg.get("lstm_early_stopping_metric", "crps").strip().lower()
    if early_stop_metric == "ll":
        early_stop_key = "validation_log_likelihood"
        early_stop_mode = "max"
    elif early_stop_metric == "crps":
        early_stop_key = "validation_crps"
        early_stop_mode = "min"
    else:
        raise ValueError(f"Unknown lstm_early_stopping_metric: {early_stop_metric}")

    validation_obs = data_processor.get_data("validation").flatten()

    lstm_kwargs = dict(
        look_back_len=cfg["lstm_look_back_len"],
        num_features=cfg["lstm_num_features"],
        num_layer=cfg["lstm_num_layer"],
        infer_len=cfg["lstm_infer_len"],
        num_hidden_unit=cfg["num_hidden_unit"],
        device="cpu",
        num_thread=LSTM_NUM_THREAD,
        manual_seed=seed,
        smoother=cfg["smoother"],
        stateless=cfg["lstm_stateless"],
        finetune=cfg["lstm_finetune"],
        increase_output_variance=bool(cfg.get("lstm_increase_output_variance", False)),
        load_lstm_net=cfg.get("lstm_global_params"),
        model_noise=cfg.get("use_tagiv", False),
        zeroshot=cfg.get("lstm_zeroshot", False),
    )

    components = [LocalTrend(), LstmNetwork(**lstm_kwargs), WhiteNoise(std_error=sigma_v)]
    model = Model(*components)
    model.auto_initialize_baseline_states(
        train_data["y"][: int(cfg["baseline_init_len"])]
    )
    model.mu_states[model.get_states_index("trend")] = 0.0
    model.lstm_net.teacher_forcing = False

    scale_mean = data_processor.scale_const_mean[data_processor.output_col]
    scale_std = data_processor.scale_const_std[data_processor.output_col]

    for epoch in range(MAX_EPOCHS):
        if model.lstm_net.smooth is False:
            model.lstm_output_history.set(warmup_lookback_mu, warmup_lookback_var)

        mu_val, std_val, _ = model.lstm_train(
            train_data=train_data,
            validation_data=validation_data,
            white_noise_decay=False,
        )
        mu_val = normalizer.unstandardize(mu_val, scale_mean, scale_std)
        std_val = normalizer.unstandardize_std(std_val, scale_std)

        val_ll = metric.log_likelihood(prediction=mu_val, observation=validation_obs, std=std_val)
        val_rmse = float(np.sqrt(np.nanmean((mu_val - validation_obs) ** 2)))
        val_crps = _crps_gaussian(mu_val, std_val, validation_obs)
        epoch_metrics = {
            "epoch": epoch,
            "validation_log_likelihood": float(val_ll),
            "validation_rmse": val_rmse,
            "validation_crps": val_crps,
        }

        model.early_stopping(
            evaluate_metric=epoch_metrics[early_stop_key],
            current_epoch=epoch,
            max_epoch=MAX_EPOCHS,
            mode=early_stop_mode,
            skip_epoch=0,
        )
        model.metric_optim = model.early_stop_metric

        if model.stop_training:
            model.early_stop_lstm_output_mu = model.lstm_output_history.mu.copy()
            model.early_stop_lstm_output_var = model.lstm_output_history.var.copy()
            break

    print(
        f"  [seed {seed}] trained {epoch + 1} epochs | "
        f"best {early_stop_key} @ epoch {int(model.optimal_epoch)}"
    )
    return model


def _build_skf(trained_model: Model, params: dict, dataset: dict) -> SKF:
    warmup_lookback_mu = dataset["warmup_lookback_mu"]
    warmup_lookback_var = dataset["warmup_lookback_var"]

    abnorm_components = [LocalAcceleration(), LstmNetwork()]
    abnorm_components.append(WhiteNoise(std_error=params["sigma_v"]))
    abnorm_model = Model(*abnorm_components)
    skf = SKF(
        norm_model=trained_model,
        abnorm_model=abnorm_model,
        std_transition_error=params["std_transition_error"],
        norm_to_abnorm_prob=params["norm_to_abnorm_prob"],
        abnorm_to_norm_prob=params["abnorm_to_norm_prob"],
        likelihood_covariance_floor=params.get("likelihood_covariance_floor", 0.0),
    )
    if skf.model["norm_norm"].lstm_net.smooth is False:
        skf.model["norm_norm"].lstm_output_history.set(
            warmup_lookback_mu, warmup_lookback_var
        )
    skf.model["norm_norm"].lstm_net.teacher_forcing = False
    skf.save_initial_states()
    return skf


def _run_seed(seed: int, base_config: dict):
    print(f"=== seed {seed} ===")
    params = _load_seed_params(seed)
    if SIGMA_V_OVERRIDE is not None:
        params["sigma_v"] = SIGMA_V_OVERRIDE
    print(f"  saved params: {params}")

    cfg = copy.deepcopy(base_config)
    cfg["lstm_manual_seed"] = seed
    if MODE == "local":
        cfg["lstm_global_params"] = None
        cfg["lstm_num_layer"] = 1
    cfg["sigma_v"] = params["sigma_v"]

    dataset = prepare_dataset(
        train_split=float(cfg["train_split"]),
        anomaly_slope=0.0,
        experiment_config=cfg,
    )

    trained_model = _train_lstm(cfg, seed, params["sigma_v"], dataset)
    skf = _build_skf(trained_model, params, dataset)

    data_processor = dataset["data_processor"]
    all_data = dataset["all_data"]

    total_eval_steps = len(all_data["y"])
    max_timestep_to_detect = float(cfg.get("max_timestep_to_detect", 156))
    test_start_ratio = data_processor.test_start / total_eval_steps
    anomaly_end_ratio = (
        data_processor.test_end - max_timestep_to_detect
    ) / total_eval_steps

    mag_slope = MAGNITUDE / 52
    realizations = DataProcess.add_synthetic_anomaly(
        all_data,
        num_samples=1,
        slope=[mag_slope],
        anomaly_start=test_start_ratio,
        anomaly_end=anomaly_end_ratio,
    )
    realization = realizations[0]
    anomaly_timestep = int(realization["anomaly_timestep"])

    filter_probs, _ = skf.filter(data=realization)
    filter_probs = np.asarray(filter_probs).reshape(-1)

    scale_mean = float(data_processor.scale_const_mean[data_processor.output_col])
    scale_std = float(data_processor.scale_const_std[data_processor.output_col])
    clean_y = np.asarray(all_data["y"]).reshape(-1)
    anomaly_y = np.asarray(realization["y"]).reshape(-1)

    clean_y_unnorm = normalizer.unstandardize(clean_y, scale_mean, scale_std)
    anomaly_y_unnorm = normalizer.unstandardize(anomaly_y, scale_mean, scale_std)
    time_axis = np.asarray(data_processor.data.index.to_numpy())[: len(anomaly_y)]

    out_path = OUTPUT_DIR / f"seed{seed}.npz"
    np.savez(
        out_path,
        seed=seed,
        filter_probs=filter_probs,
        clean_y=clean_y_unnorm,
        anomaly_y=anomaly_y_unnorm,
        anomaly_timestep=anomaly_timestep,
        threshold=float(params["threshold"]),
        time_axis=time_axis,
        magnitude=MAGNITUDE,
    )
    print(
        f"  saved: {out_path} | anomaly@{anomaly_timestep} | "
        f"first_prob_peak={float(filter_probs[anomaly_timestep:].max()):.3f}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = yaml.safe_load(CONFIG_PATH.read_text())
    base_config.setdefault("lstm_early_stopping_metric", "crps")

    for seed in SEEDS:
        _run_seed(seed, base_config)

    meta = {
        "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "seeds": SEEDS,
        "magnitude": MAGNITUDE,
    }
    (OUTPUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nAll seeds done. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
