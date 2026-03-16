import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from ray import tune
from pytagi import metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, Optimizer, SKF, plot_data, plot_skf_states
import canari.common as common
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise

# Plotting defaults
import matplotlib as mpl

mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
        "lines.linewidth": 1,
    }
)

RUN_STYLE = {
    "stateful_local": {
        "label": r"L$_{sf}$",
        "color": "C3",
        "marker": "o",
        "linestyle": "-",
    },
    "stateful_global": {
        "label": r"G$^{ft}_{sf/w}$",
        "color": "C0",
        "marker": "o",
        "linestyle": "-",
    },
}

DETECTION_THRESHOLD = 0.5
DEFAULT_SEEDS = [1, 42, 3]
DEFAULT_TRAIN_SPLITS = [1.0, 0.8, 0.6, 0.4]
LOOK_BACK_LEN_CANDIDATES = [4, 8, 12, 13, 16, 24, 26, 36, 52]
GLOBAL_WARMUP_LOOKBACK_LEN = 52
DEFAULT_LOOK_BACK_LEN = 52
DEFAULT_SIGMA_V = 0.09
DEFAULT_STD_TRANSITION_ERROR = 1e-5
DEFAULT_NORM_TO_ABNORM_PROB = 1e-4
DEFAULT_CDF_SLOPE = 0.1
DEFAULT_LL_FALSE_ALARM_PENALTY_WEIGHT = 1.0
BASELINE_INIT_LEN = 52 * 4
LSTM_NUM_FEATURES = 2
LSTM_NUM_HIDDEN_UNITS = 40
LSTM_INFER_LEN = 52 * 3
LSTM_NUM_EPOCH = 50
REQUIRED_CONFIG_KEYS = (
    "data_file",
    "validation_start",
    "test_start",
    "anomaly_start_time",
)


def _format_float_tag(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _parse_list(values, cast):
    if isinstance(values, (list, tuple)):
        return [cast(v) for v in values]
    if isinstance(values, str):
        return [cast(v.strip()) for v in values.split(",") if v.strip()]
    return [cast(values)]


def _load_experiment_config(config_path: str) -> dict:
    config_file = Path(config_path).expanduser()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_file}")

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if not config.get(key)]
    if missing_keys:
        raise ValueError(
            f"Config file is missing required values {missing_keys}: {config_file}"
        )

    validated_config = {"config_path": str(config_file.resolve())}
    data_file = Path(str(config["data_file"])).expanduser()
    if not data_file.exists():
        raise FileNotFoundError(f"Configured data file not found: {data_file}")
    validated_config["data_file"] = str(data_file.resolve())

    for key in ("validation_start", "test_start", "anomaly_start_time"):
        validated_config[key] = str(pd.Timestamp(config[key]))

    return validated_config


def _print_experiment_config(experiment_config: dict):
    print("Experiment config:")
    print("  config_path:", experiment_config["config_path"])
    print("  data_file:", experiment_config["data_file"])
    print("  validation_start:", experiment_config["validation_start"])
    print("  test_start:", experiment_config["test_start"])
    print("  anomaly_start_time:", experiment_config["anomaly_start_time"])


def _load_base_dataframe(experiment_config: dict) -> pd.DataFrame:
    df = pd.read_csv(experiment_config["data_file"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.index.name = "date_time"
    return df


def _apply_train_split_via_part_to_remove(
    df_raw_full: pd.DataFrame,
    train_split: float,
    validation_start: str,
):
    validation_ts = pd.Timestamp(validation_start)
    n_train_total = int(df_raw_full.index.searchsorted(validation_ts, side="left"))
    if n_train_total <= GLOBAL_WARMUP_LOOKBACK_LEN:
        raise ValueError("Training window is too short after preprocessing.")

    n_train_use = max(int(train_split * n_train_total), GLOBAL_WARMUP_LOOKBACK_LEN + 1)
    n_train_use = min(n_train_use, n_train_total)
    part_to_remove = n_train_total - n_train_use

    df_subset = df_raw_full.iloc[part_to_remove:].copy()
    return df_subset, n_train_total, n_train_use, part_to_remove


def _resolve_anomaly_index(index: pd.DatetimeIndex, anomaly_start_time: str) -> int:
    if len(index) == 0:
        raise ValueError("Cannot resolve anomaly time on an empty index.")
    target_time = pd.Timestamp(anomaly_start_time)
    anomaly_idx = int(index.searchsorted(target_time, side="left"))
    if anomaly_idx >= len(index):
        anomaly_idx = len(index) - 1
    return anomaly_idx


def _prepare_dataset(
    train_split: float,
    anomaly_slope: float,
    experiment_config: dict,
    warmup_len: int = GLOBAL_WARMUP_LOOKBACK_LEN,
):
    df_original = _load_base_dataframe(experiment_config)
    data_pro_scale = DataProcess(
        data=df_original,
        time_covariates=["week_of_year"],
        train_split=1.0,
        output_col=[0],
    )

    df_raw, n_train_total, n_train_use, part_to_remove = (
        _apply_train_split_via_part_to_remove(
            df_original,
            train_split,
            experiment_config["validation_start"],
        )
    )

    warmup_lookback_mu = None
    warmup_lookback_var = None
    if len(df_raw) <= warmup_len:
        raise ValueError(
            f"Not enough data after part removal to create a {warmup_len}-step warmup."
        )
    df_warmup = df_raw.iloc[:warmup_len].copy()
    df_raw = df_raw.iloc[warmup_len:].copy()
    warmup_lookback_mu = df_warmup.iloc[:, 0].values.flatten()
    warmup_lookback_mu = normalizer.standardize(
        warmup_lookback_mu,
        data_pro_scale.scale_const_mean[data_pro_scale.output_col],
        data_pro_scale.scale_const_std[data_pro_scale.output_col],
    )
    warmup_lookback_var = np.zeros_like(warmup_lookback_mu)

    trend = np.linspace(0, 0, num=len(df_raw))
    time_anomaly = _resolve_anomaly_index(
        df_raw.index, experiment_config["anomaly_start_time"]
    )
    new_trend = np.linspace(0, anomaly_slope, num=len(df_raw) - time_anomaly)
    trend[time_anomaly:] = trend[time_anomaly:] + new_trend
    df = df_raw.add(trend, axis=0)

    df_plot_full = df_original.iloc[warmup_len:].copy()
    plot_trend = np.linspace(0, 0, num=len(df_plot_full))
    plot_time_anomaly = _resolve_anomaly_index(
        df_plot_full.index, experiment_config["anomaly_start_time"]
    )
    plot_new_trend = np.linspace(
        0, anomaly_slope, num=len(df_plot_full) - plot_time_anomaly
    )
    plot_trend[plot_time_anomaly:] = plot_trend[plot_time_anomaly:] + plot_new_trend
    df_plot_full = df_plot_full.add(plot_trend, axis=0)

    data_processor = DataProcess(
        data=df,
        time_covariates=["week_of_year"],
        validation_start=experiment_config["validation_start"],
        test_start=experiment_config["test_start"],
        output_col=[0],
        scale_const_mean=data_pro_scale.scale_const_mean,
        scale_const_std=data_pro_scale.scale_const_std,
    )
    plot_data_processor = DataProcess(
        data=df_plot_full,
        time_covariates=["week_of_year"],
        validation_start=experiment_config["validation_start"],
        test_start=experiment_config["test_start"],
        output_col=[0],
        scale_const_mean=data_pro_scale.scale_const_mean,
        scale_const_std=data_pro_scale.scale_const_std,
    )
    train_data, validation_data, _, all_data = data_processor.get_splits()

    metadata = {
        "time_anomaly": time_anomaly,
        "anomaly_time": df.index[time_anomaly],
        "plot_anomaly_time": df_plot_full.index[plot_time_anomaly],
        "plot_data_processor": plot_data_processor,
        "train_rows_full": n_train_total,
        "train_rows_used": n_train_use,
        "part_to_remove": part_to_remove,
        "unused_train_rows": part_to_remove,
    }
    return (
        data_processor,
        train_data,
        validation_data,
        all_data,
        warmup_lookback_mu,
        warmup_lookback_var,
        metadata,
    )


def _slice_warmup_lookback(
    warmup_lookback_mu: np.ndarray,
    warmup_lookback_var: np.ndarray,
    look_back_len: int,
):
    if warmup_lookback_mu is None or warmup_lookback_var is None:
        return None, None
    if look_back_len > len(warmup_lookback_mu):
        raise ValueError(
            f"Requested look_back_len={look_back_len} but only {len(warmup_lookback_mu)} warmup steps are available."
        )
    return (
        warmup_lookback_mu[-look_back_len:],
        warmup_lookback_var[-look_back_len:],
    )


def _skf_log_lik_without_hete_noise(skf: SKF, data: dict) -> float:
    y = np.asarray(data["y"]).flatten()
    if len(skf.states.mu_prior) == 0:
        return np.nan

    obs_matrix = skf.model["norm_norm"].observation_matrix.copy()
    hete_idx = skf.model["norm_norm"].get_states_index("heteroscedastic noise")
    if hete_idx is not None:
        obs_matrix[0, hete_idx] = 0.0

    mu_pred = []
    std_pred = []
    for mu_state, var_state in zip(skf.states.mu_prior, skf.states.var_prior):
        mu_obs, var_obs = common.calc_observation(mu_state, var_state, obs_matrix)
        mu_pred.append(float(np.asarray(mu_obs).flatten()[0]))
        var_scalar = float(np.asarray(var_obs).flatten()[0])
        std_pred.append(np.sqrt(np.maximum(var_scalar, 1e-12)))

    mu_pred = np.asarray(mu_pred)
    std_pred = np.asarray(std_pred)
    n = min(len(y), len(mu_pred))
    y = y[:n]
    mu_pred = mu_pred[:n]
    std_pred = std_pred[:n]
    valid = ~np.isnan(y)
    if not np.any(valid):
        return np.nan

    return float(
        metric.log_likelihood(
            prediction=mu_pred[valid],
            observation=y[valid],
            std=std_pred[valid],
        )
    )


def _first_detection_index(
    model_prob: np.ndarray,
    threshold: float = DETECTION_THRESHOLD,
    start_idx: int = 0,
):
    model_prob = np.asarray(model_prob).flatten()
    if start_idx >= len(model_prob):
        return None
    detected = np.where(model_prob[start_idx:] > threshold)[0]
    if len(detected) == 0:
        return None
    return int(start_idx + detected[0])


def _detection_lag_weeks(index: pd.DatetimeIndex, anomaly_idx: int, detection_idx: int):
    if detection_idx is None:
        return np.nan
    anomaly_time = pd.Timestamp(index[anomaly_idx])
    detection_time = pd.Timestamp(index[detection_idx])
    return (detection_time - anomaly_time).days / 7.0


def _false_alarm_rate_per_year(
    model_prob: np.ndarray,
    index: pd.DatetimeIndex,
    anomaly_idx: int,
    threshold: float = DETECTION_THRESHOLD,
):
    if anomaly_idx <= 0:
        return np.nan

    pre_prob = np.asarray(model_prob).flatten()[:anomaly_idx]
    if len(pre_prob) == 0:
        return np.nan

    above = pre_prob > threshold
    crossing_count = int(np.sum(above & ~np.r_[False, above[:-1]]))

    start_time = pd.Timestamp(index[0])
    anomaly_time = pd.Timestamp(index[anomaly_idx])
    duration_years = max((anomaly_time - start_time).days / 365.25, 1e-12)
    return crossing_count / duration_years


def _skf_log_lik_false_alarm_objective(
    log_lik_all: float,
    false_alarm_rate_year: float,
    penalty_weight: float = DEFAULT_LL_FALSE_ALARM_PENALTY_WEIGHT,
):
    penalty_rate = (
        float(false_alarm_rate_year) if np.isfinite(false_alarm_rate_year) else 0.0
    )
    return -float(log_lik_all) + float(penalty_weight) * penalty_rate


def _resolve_global_lstm_path(global_lstm_dir: str, look_back_len: int) -> str:
    path = Path(global_lstm_dir) / (
        f"ByWindow_global_no-embeddings_seed42_whitenoise.bin"  # TODO: Update this if the global model training settings change
    )
    if not path.exists():
        raise FileNotFoundError(f"Global LSTM checkpoint not found: {path}")
    return str(path)


def _plot_split_figure(
    data_processor,
    anomaly_time,
    unused_train_rows: int,
    combo_dir: Path,
    combo_tag: str,
):
    fig_split, ax_split = plt.subplots(figsize=(10, 3))
    plot_data(
        data_processor=data_processor,
        standardization=False,
        plot_column=[0],
        sub_plot=ax_split,
    )

    if len(ax_split.lines) > 0:
        train_line = ax_split.lines[0]
        train_color = train_line.get_color()
        train_linestyle = train_line.get_linestyle()
        train_line.remove()

        train_start = data_processor.train_start
        train_end = data_processor.train_end
        unused_train_end = min(train_start + max(unused_train_rows, 0), train_end)
        train_time = data_processor.data.index[train_start:train_end]
        train_values = data_processor.data.iloc[train_start:train_end, 0].to_numpy()

        if unused_train_end > train_start:
            unused_len = unused_train_end - train_start
            ax_split.plot(
                train_time[:unused_len],
                train_values[:unused_len],
                color=train_color,
                linestyle=train_linestyle,
                alpha=0.2,
            )

        used_train_start = max(unused_train_end, train_start)
        if used_train_start < train_end:
            used_offset = used_train_start - train_start
            ax_split.plot(
                train_time[used_offset:],
                train_values[used_offset:],
                color=train_color,
                linestyle=train_linestyle,
                alpha=1.0,
            )

    ax_split.axvline(
        x=anomaly_time,
        color="k",
        linestyle="--",
        linewidth=1.5,
        label="Anomaly",
    )
    ax_split.legend(loc="best")
    fig_split.tight_layout()
    fig_split_path = combo_dir / f"DATA_split_anomaly_{combo_tag}.svg"
    fig_split.savefig(fig_split_path, bbox_inches="tight")
    print("Saved figure:", fig_split_path)
    plt.close(fig_split)


def _default_model_param():
    return {
        "look_back_len": DEFAULT_LOOK_BACK_LEN,
        "sigma_v": DEFAULT_SIGMA_V,
    }


def _default_skf_param(skf_objective: str):
    param = {
        "std_transition_error": DEFAULT_STD_TRANSITION_ERROR,
        "norm_to_abnorm_prob": DEFAULT_NORM_TO_ABNORM_PROB,
    }
    if skf_objective == "cdf":
        param["slope"] = DEFAULT_CDF_SLOPE
    return param


def _build_model_param_space():
    return {
        # "look_back_len": tune.choice(LOOK_BACK_LEN_CANDIDATES),
        "sigma_v": [1e-3, 1e-1],
    }


def _build_skf_param_space(skf_objective: str):
    param_space = {
        "std_transition_error": [1e-6, 1e-4],
        "norm_to_abnorm_prob": [1e-6, 1e-4],
    }
    if skf_objective == "cdf":
        param_space["slope"] = [0.1, 0.6]
    return param_space


def _print_split_summary(seed: int, train_split: float, metadata: dict, data_processor):
    print("\n" + "#" * 80)
    print(
        f"Seed={seed} | train_split={train_split} | full_train_rows={metadata['train_rows_full']} | "
        f"used_train_rows={metadata['train_rows_used']} | part_to_remove={metadata['part_to_remove']} | "
        f"fixed_front_slice={GLOBAL_WARMUP_LOOKBACK_LEN}"
    )
    print("Train start:", data_processor.data.index[data_processor.train_start])
    print("Train end:", data_processor.data.index[data_processor.train_end - 1])
    print(
        "Validation:",
        data_processor.data.index[data_processor.validation_start],
        "to",
        data_processor.data.index[data_processor.validation_end - 1],
    )
    print(
        "Test:",
        data_processor.data.index[data_processor.test_start],
        "to",
        data_processor.data.index[data_processor.test_end - 1],
    )


def _run_single_mode(
    run_name: str,
    manual_seed: int,
    finetune: bool,
    use_tagiv: bool,
    skf_objective: str,
    ll_false_alarm_penalty_weight: float,
    model_param_optimization: bool,
    skf_param_optimization: bool,
    num_trial_optim_normal_model: int,
    num_trial_optim_skf: int,
    train_data: dict,
    validation_data: dict,
    all_data: dict,
    data_processor: DataProcess,
    warmup_lookback_mu: np.ndarray,
    warmup_lookback_var: np.ndarray,
    anomaly_idx: int,
    combo_dir: Path,
    combo_tag: str,
    global_lstm_dir: str = None,
):
    model_mode = "global" if global_lstm_dir else "local"
    mode_finetune = finetune if model_mode == "global" else False
    smoother = model_mode == "local"
    use_warmup_lookback = model_mode == "global"

    print("\n" + "=" * 70)
    print("Running:", run_name)
    print("Mode:", model_mode)
    print("Global weights dir:", global_lstm_dir if global_lstm_dir else "None")
    print(
        "Settings:",
        {
            "stateless": False,
            "smoother": smoother,
            "finetune": mode_finetune,
            "warmup_lookback_len": (
                GLOBAL_WARMUP_LOOKBACK_LEN if use_warmup_lookback else 0
            ),
        },
    )

    validation_obs = data_processor.get_data("validation").flatten()

    def normal_model_with_parameters(model_param):
        sigma_v = model_param.get("sigma_v", DEFAULT_SIGMA_V)
        look_back_len = int(model_param.get("look_back_len", DEFAULT_LOOK_BACK_LEN))
        load_lstm_path = (
            _resolve_global_lstm_path(global_lstm_dir, look_back_len)
            if model_mode == "global"
            else None
        )
        sliced_warmup_mu, sliced_warmup_var = _slice_warmup_lookback(
            warmup_lookback_mu,
            warmup_lookback_var,
            look_back_len,
        )
        norm_model = Model(
            LocalTrend(mu_states=[0,0], var_states=[1e-6, 1e-6]),
            LstmNetwork(
                look_back_len=look_back_len,
                num_features=LSTM_NUM_FEATURES,
                num_layer=1 if model_mode == "local" else 3,
                num_hidden_unit=LSTM_NUM_HIDDEN_UNITS,
                manual_seed=manual_seed,
                infer_len=LSTM_INFER_LEN,
                smoother=smoother,
                model_noise=use_tagiv,
                load_lstm_net=load_lstm_path,
                finetune=mode_finetune,
                stateless=False,
                zero_shot=False,
            ),
            WhiteNoise(std_error=sigma_v),
        )

        # norm_model.auto_initialize_baseline_states(train_data["y"][0:BASELINE_INIT_LEN])

        for epoch in range(LSTM_NUM_EPOCH):
            if use_warmup_lookback:
                norm_model.lstm_output_history.set(sliced_warmup_mu, sliced_warmup_var)
            mu_validation_preds, std_validation_preds, _ = norm_model.lstm_train(
                train_data=train_data,
                validation_data=validation_data,
                white_noise_max_std=1.0,  # TODO: check how it affects the global model finetuning results
            )

            mu_validation_preds_unnorm = normalizer.unstandardize(
                mu_validation_preds,
                data_processor.scale_const_mean[data_processor.output_col],
                data_processor.scale_const_std[data_processor.output_col],
            )
            std_validation_preds_unnorm = normalizer.unstandardize_std(
                std_validation_preds,
                data_processor.scale_const_std[data_processor.output_col],
            )

            validation_log_lik = metric.log_likelihood(
                prediction=mu_validation_preds_unnorm,
                observation=validation_obs,
                std=std_validation_preds_unnorm,
            )

            norm_model.early_stopping(
                evaluate_metric=-validation_log_lik,
                current_epoch=epoch,
                max_epoch=LSTM_NUM_EPOCH,
            )
            norm_model.metric_optim = norm_model.early_stop_metric

            if norm_model.stop_training:
                break

        norm_model.metric_optim = float(norm_model.metric_optim)
        norm_model.print_metric = {
            "validation_log_lik": float(-norm_model.metric_optim)
        }
        return norm_model

    def skf_with_parameters(skf_param, skf_input):
        norm_model = Model.load_dict(skf_input["norm_model_dict"])
        look_back_len = int(skf_input["look_back_len"])
        std_transition_error = skf_param.get(
            "std_transition_error", DEFAULT_STD_TRANSITION_ERROR
        )
        norm_to_abnorm_prob = skf_param.get(
            "norm_to_abnorm_prob", DEFAULT_NORM_TO_ABNORM_PROB
        )
        slope = skf_param.get("slope", DEFAULT_CDF_SLOPE)

        sliced_warmup_mu, sliced_warmup_var = _slice_warmup_lookback(
            warmup_lookback_mu,
            warmup_lookback_var,
            look_back_len,
        )
        if use_warmup_lookback:
            norm_model.lstm_output_history.set(sliced_warmup_mu, sliced_warmup_var)

        abnorm_model = Model(
            LocalAcceleration(),
            LstmNetwork(
                look_back_len=look_back_len,
                model_noise=use_tagiv,
                smoother=False,
                stateless=False,
            ),
            WhiteNoise(),
        )
        skf = SKF(
            norm_model=norm_model,
            abnorm_model=abnorm_model,
            std_transition_error=std_transition_error,
            norm_to_abnorm_prob=norm_to_abnorm_prob,
        )
        skf.save_initial_states()

        if skf_objective in {"ll", "ll_false_alarm"}:
            filter_marginal_abnorm_prob, _ = skf.filter(data=all_data)
            log_lik_all = _skf_log_lik_without_hete_noise(skf, all_data)
            if skf_objective == "ll":
                skf.metric_optim = -log_lik_all
                skf.print_metric = {"log_lik_all": log_lik_all}
            else:
                false_alarm_rate_year = _false_alarm_rate_per_year(
                    model_prob=filter_marginal_abnorm_prob,
                    index=data_processor.data.index,
                    anomaly_idx=anomaly_idx,
                )
                skf.metric_optim = _skf_log_lik_false_alarm_objective(
                    log_lik_all=log_lik_all,
                    false_alarm_rate_year=false_alarm_rate_year,
                    penalty_weight=ll_false_alarm_penalty_weight,
                )
                skf.print_metric = {
                    "log_lik_all": log_lik_all,
                    "false_alarm_rate_year": false_alarm_rate_year,
                    "ll_false_alarm_penalty_weight": ll_false_alarm_penalty_weight,
                }
        else:
            detection_rate, false_rate, false_alarm_train = (
                skf.detect_synthetic_anomaly(
                    data=train_data,
                    num_anomaly=30,
                    slope_anomaly=slope / 52,
                )
            )
            data_len_year = (
                data_processor.data.index[data_processor.train_end]
                - data_processor.data.index[data_processor.train_start]
            ).days / 365.25
            false_rate_yearly = false_rate / max(data_len_year, 1e-12)
            metric_optim = skf.objective(detection_rate, false_rate_yearly, slope)
            skf.metric_optim = metric_optim
            skf.print_metric = {
                "detection_rate": detection_rate,
                "yearly_false_rate": false_rate_yearly,
                "false_alarm_train": false_alarm_train,
            }

        skf.load_initial_states()
        return skf

    if model_param_optimization:
        normal_model_optimizer = Optimizer(
            model=normal_model_with_parameters,
            param=_build_model_param_space(),
            num_optimization_trial=num_trial_optim_normal_model,
            num_startup_trials=10,
            mode="min",
        )
        normal_model_optimizer.optimize()
        model_param = normal_model_optimizer.get_best_param()
        normal_model_optim = normal_model_optimizer.get_best_model()
    else:
        model_param = _default_model_param()
        normal_model_optim = normal_model_with_parameters(model_param)

    skf_input = {
        "norm_model_dict": normal_model_optim.get_dict(time_step=0),
        "look_back_len": int(model_param.get("look_back_len", DEFAULT_LOOK_BACK_LEN)),
    }

    if skf_param_optimization:
        skf_optimizer = Optimizer(
            model=skf_with_parameters,
            param=_build_skf_param_space(skf_objective),
            model_input=skf_input,
            num_optimization_trial=num_trial_optim_skf,
            num_startup_trials=30,
            mode="max" if skf_objective == "cdf" else "min",
        )
        skf_optimizer.optimize()
        skf_param = skf_optimizer.get_best_param()
        skf_optim = skf_optimizer.get_best_model()
    else:
        skf_param = _default_skf_param(skf_objective)
        skf_optim = skf_with_parameters(skf_param, skf_input)

    skf_optim_dict = skf_optim.get_dict()
    skf_optim_dict["model_param"] = model_param
    skf_optim_dict["skf_param"] = skf_param

    print("Model parameters used:", skf_optim_dict["model_param"])
    print("SKF parameters used:", skf_optim_dict["skf_param"])
    print("SKF optimization objective:", skf_objective)
    if skf_objective == "ll_false_alarm":
        print("LL false-alarm penalty weight:", ll_false_alarm_penalty_weight)

    if use_warmup_lookback:
        sliced_warmup_mu, sliced_warmup_var = _slice_warmup_lookback(
            warmup_lookback_mu,
            warmup_lookback_var,
            int(
                skf_optim_dict["model_param"].get(
                    "look_back_len", DEFAULT_LOOK_BACK_LEN
                )
            ),
        )
        skf_optim.model["norm_norm"].lstm_output_history.set(
            sliced_warmup_mu, sliced_warmup_var
        )
    filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)
    model_prob = np.asarray(filter_marginal_abnorm_prob).flatten()
    detection_idx = _first_detection_index(model_prob, threshold=DETECTION_THRESHOLD)
    post_anomaly_detection_idx = _first_detection_index(
        model_prob,
        threshold=DETECTION_THRESHOLD,
        start_idx=anomaly_idx,
    )
    detection_time = (
        data_processor.data.index[detection_idx]
        if detection_idx is not None and detection_idx < len(data_processor.data.index)
        else None
    )
    post_anomaly_detection_time = (
        data_processor.data.index[post_anomaly_detection_idx]
        if post_anomaly_detection_idx is not None
        and post_anomaly_detection_idx < len(data_processor.data.index)
        else None
    )
    print(
        "First detection time (threshold " f"{DETECTION_THRESHOLD:.2f}):",
        detection_time if detection_time is not None else "not detected",
    )
    print(
        "First post-anomaly detection time:",
        (
            post_anomaly_detection_time
            if post_anomaly_detection_time is not None
            else "not detected"
        ),
    )

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        model_prob=filter_marginal_abnorm_prob,
        standardization=True,
        states_type="prior",
    )
    anomaly_time = data_processor.data.index[anomaly_idx]
    ax[-1].axvline(
        x=anomaly_time,
        color="k",
        linestyle="--",
        linewidth=1.2,
        label="Anomaly",
    )
    ax[-1].legend(loc="upper right")
    fig_name = f"{run_name}_SKF_{skf_objective.upper()}_{combo_tag}.svg"
    fig_path = combo_dir / fig_name
    fig.savefig(fig_path, bbox_inches="tight")
    print("Saved figure:", fig_path)
    plt.close(fig)

    return {
        "run_name": run_name,
        "model_mode": model_mode,
        "time_index": data_processor.data.index.copy(),
        "model_prob": model_prob,
        "detection_idx": detection_idx,
        "detection_time": detection_time,
        "post_anomaly_detection_idx": post_anomaly_detection_idx,
        "post_anomaly_detection_time": post_anomaly_detection_time,
        "look_back_len": int(
            skf_optim_dict["model_param"].get("look_back_len", DEFAULT_LOOK_BACK_LEN)
        ),
        "detection_lag_weeks": _detection_lag_weeks(
            data_processor.data.index,
            anomaly_idx,
            post_anomaly_detection_idx,
        ),
        "false_alarm_rate_year": _false_alarm_rate_per_year(
            model_prob,
            data_processor.data.index,
            anomaly_idx,
        ),
    }


def _plot_detection_comparison(
    data_processor,
    run_results,
    anomaly_time,
    combo_dir: Path,
    combo_tag: str,
    skf_objective: str,
):
    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 3))
    for run_result in run_results:
        run_name = run_result["run_name"]
        style = RUN_STYLE[run_name]
        time = pd.DatetimeIndex(run_result["time_index"])
        model_prob = run_result["model_prob"]
        n = min(len(time), len(model_prob))
        time = time[:n]
        model_prob = model_prob[:n]

        ax_cmp.plot(
            time,
            model_prob,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.2,
            label=style["label"],
        )

        detection_idx = run_result["detection_idx"]
        if detection_idx is not None and detection_idx < n:
            ax_cmp.scatter(
                time[detection_idx],
                model_prob[detection_idx],
                color=style["color"],
                marker=style["marker"],
                s=28,
                zorder=3,
            )
            ax_cmp.axvline(
                x=time[detection_idx],
                color=style["color"],
                linestyle=":",
                linewidth=0.8,
                alpha=0.5,
            )

    ax_cmp.axvline(
        x=anomaly_time,
        color="k",
        linestyle="--",
        linewidth=1.5,
        label="Anomaly",
    )
    ax_cmp.axhline(
        y=DETECTION_THRESHOLD,
        color="0.4",
        linestyle=":",
        linewidth=1.0,
        label=f"Threshold ({DETECTION_THRESHOLD:.2f})",
    )
    ax_cmp.set_ylabel("Abnormal Probability")
    ax_cmp.set_ylim(-0.02, 1.02)
    ax_cmp.set_title("Anomaly Detection Comparison")
    ax_cmp.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig_cmp.tight_layout()
    fig_cmp_path = (
        combo_dir / f"ALL_detection_compare_SKF_{skf_objective.upper()}_{combo_tag}.svg"
    )
    fig_cmp.savefig(fig_cmp_path, bbox_inches="tight")
    print("Saved figure:", fig_cmp_path)
    plt.close(fig_cmp)


def _plot_single_summary_heatmap(
    matrix: np.ndarray,
    panel_title: str,
    colorbar_label: str,
    figure_stem: str,
    ordered_seeds,
    ordered_splits,
    save_dir: Path,
    vmin: float,
    vmax: float,
):
    fig, ax = plt.subplots(figsize=(3.6, 2.8))

    cmap = plt.cm.viridis_r.copy()
    cmap.set_bad(color="0.95")
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(panel_title, fontsize=10, pad=6)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Train size")
    ax.set_xticks(np.arange(len(ordered_seeds)))
    ax.set_xticklabels([str(seed) for seed in ordered_seeds], fontsize=8)
    ax.set_yticks(np.arange(len(ordered_splits)))
    ax.set_yticklabels([f"{int(ts * 100)}%" for ts in ordered_splits], fontsize=8)

    ax.set_xticks(np.arange(-0.5, len(ordered_seeds), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordered_splits), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    mid = (vmin + vmax) / 2
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            label = "ND" if np.isnan(value) else f"{value:.1f}"
            text_color = "black" if np.isnan(value) or value < mid else "white"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label(colorbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    svg_path = save_dir / f"{figure_stem}.svg"
    pdf_path = save_dir / f"{figure_stem}.pdf"
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print("Saved figure:", svg_path)
    print("Saved figure:", pdf_path)
    plt.close(fig)


def _plot_summary_metric_heatmaps(
    summary_df: pd.DataFrame,
    seeds,
    train_splits,
    save_dir: Path,
    metric_col: str,
    title_suffix: str,
    colorbar_label: str,
    file_suffix: str,
):
    ordered_splits = list(train_splits)
    ordered_seeds = list(seeds)
    model_order = ["stateful_local", "stateful_global"]
    panel_titles = {
        "stateful_local": rf"Local Model ($L_{{sf}}$): {title_suffix}",
        "stateful_global": rf"Global Fine-Tuned Model ($G^{{ft}}_{{sf/w}}$): {title_suffix}",
    }

    matrices = {}
    for model_name in model_order:
        matrix = np.full((len(ordered_splits), len(ordered_seeds)), np.nan)
        for split_idx, train_split in enumerate(ordered_splits):
            for seed_idx, seed in enumerate(ordered_seeds):
                match = summary_df[
                    (summary_df["run_name"] == model_name)
                    & (summary_df["seed"] == seed)
                    & (summary_df["train_split"] == train_split)
                ]
                if not match.empty:
                    matrix[split_idx, seed_idx] = match.iloc[0][metric_col]
        matrices[model_name] = matrix

    finite_values = [matrix[np.isfinite(matrix)] for matrix in matrices.values()]
    finite_values = [vals for vals in finite_values if len(vals) > 0]
    if finite_values:
        all_values = np.concatenate(finite_values)
        vmin = float(np.min(all_values))
        vmax = float(np.max(all_values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    for model_name in model_order:
        _plot_single_summary_heatmap(
            matrix=matrices[model_name],
            panel_title=panel_titles[model_name],
            colorbar_label=colorbar_label,
            figure_stem=f"{model_name}_{file_suffix}",
            ordered_seeds=ordered_seeds,
            ordered_splits=ordered_splits,
            save_dir=save_dir,
            vmin=vmin,
            vmax=vmax,
        )


def _plot_summary_heatmap(
    summary_df: pd.DataFrame, seeds, train_splits, save_dir: Path
):
    _plot_summary_metric_heatmaps(
        summary_df=summary_df,
        seeds=seeds,
        train_splits=train_splits,
        save_dir=save_dir,
        metric_col="detection_lag_weeks",
        title_suffix="Detection Lag",
        colorbar_label="Detection lag (weeks)",
        file_suffix="summary_detection_lag_heatmap",
    )
    _plot_summary_metric_heatmaps(
        summary_df=summary_df,
        seeds=seeds,
        train_splits=train_splits,
        save_dir=save_dir,
        metric_col="false_alarm_rate_year",
        title_suffix="False Alarm Rate",
        colorbar_label="False alarms per year",
        file_suffix="summary_false_alarm_rate_heatmap",
    )


def main(
    num_trial_optim_model: int = 100,
    num_trial_optim_normal_model: int = 30,
    num_trial_optim_skf: int = 70,
    ll_false_alarm_penalty_weight: float = DEFAULT_LL_FALSE_ALARM_PENALTY_WEIGHT,
    model_param_optimization: bool = False,
    skf_param_optimization: bool = False,
    global_lstm_dir: str = "saved_params/global_models",
    skf_objective: str = "ll_false_alarm",
    smoother: bool = False,
    finetune: bool = True,
    use_tagiv: bool = False,
    train_splits: str = "1.0,0.8,0.6",
    anomaly_slope: float = 0.125,
    seeds: str = "1, 42",
    config_path: str = "examples/config/skf_experiment_LGA008EFAPRG910.yaml",
):
    skf_objective = skf_objective.lower()
    if skf_objective not in {"ll", "ll_false_alarm", "cdf"}:
        raise ValueError(
            "`skf_objective` must be either 'll', 'll_false_alarm', or 'cdf'."
        )
    if smoother:
        print(
            "Ignoring `smoother` input: local runs force `smoother=True`, global runs force `smoother=False`."
        )
    if num_trial_optim_normal_model is None:
        num_trial_optim_normal_model = num_trial_optim_model
    if num_trial_optim_skf is None:
        num_trial_optim_skf = num_trial_optim_model

    seeds = _parse_list(seeds, int) if seeds is not None else DEFAULT_SEEDS
    train_splits = (
        _parse_list(train_splits, float)
        if train_splits is not None
        else DEFAULT_TRAIN_SPLITS
    )
    train_splits = list(train_splits)
    seeds = list(seeds)
    global_lstm_dir = str(Path(global_lstm_dir).resolve())
    experiment_config = _load_experiment_config(config_path)
    _print_experiment_config(experiment_config)

    save_dir = (
        Path("saved_results")
        / "skf_experiment"
        / (
            f"objective_{skf_objective}"
            + (
                f"_fa_penalty_{_format_float_tag(ll_false_alarm_penalty_weight)}"
                if skf_objective == "ll_false_alarm"
                else ""
            )
            + f"_slope_{_format_float_tag(anomaly_slope)}"
            + f"_tagiv_{use_tagiv}"
        )
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Saving plots to:", save_dir.resolve())

    summary_rows = []

    for seed in seeds:
        for train_split in train_splits:
            combo_tag = f"seed{seed}_train{_format_float_tag(train_split)}"
            combo_dir = (
                save_dir
                / f"seed_{seed}"
                / f"train_split_{_format_float_tag(train_split)}"
            )
            combo_dir.mkdir(parents=True, exist_ok=True)

            dataset_bundle = _prepare_dataset(
                train_split,
                anomaly_slope,
                experiment_config,
                warmup_len=GLOBAL_WARMUP_LOOKBACK_LEN,
            )

            (
                data_processor,
                train_data,
                validation_data,
                all_data,
                warmup_lookback_mu,
                warmup_lookback_var,
                metadata,
            ) = dataset_bundle

            _print_split_summary(seed, train_split, metadata, data_processor)

            _plot_split_figure(
                metadata["plot_data_processor"],
                metadata["plot_anomaly_time"],
                metadata["unused_train_rows"],
                combo_dir,
                combo_tag,
            )

            run_results = []
            for run_name, run_global_lstm_dir in (
                ("stateful_local", None),
                ("stateful_global", global_lstm_dir),
            ):
                result = _run_single_mode(
                    run_name=run_name,
                    manual_seed=seed,
                    finetune=finetune,
                    use_tagiv=use_tagiv,
                    skf_objective=skf_objective,
                    ll_false_alarm_penalty_weight=ll_false_alarm_penalty_weight,
                    model_param_optimization=model_param_optimization,
                    skf_param_optimization=skf_param_optimization,
                    num_trial_optim_normal_model=num_trial_optim_normal_model,
                    num_trial_optim_skf=num_trial_optim_skf,
                    train_data=train_data,
                    validation_data=validation_data,
                    all_data=all_data,
                    data_processor=data_processor,
                    warmup_lookback_mu=warmup_lookback_mu,
                    warmup_lookback_var=warmup_lookback_var,
                    anomaly_idx=metadata["time_anomaly"],
                    combo_dir=combo_dir,
                    combo_tag=combo_tag,
                    global_lstm_dir=run_global_lstm_dir,
                )
                run_results.append(result)
                summary_rows.append(
                    {
                        "seed": seed,
                        "train_split": train_split,
                        "run_name": run_name,
                        "look_back_len": result["look_back_len"],
                        "detection_time": result["detection_time"],
                        "post_anomaly_detection_time": result[
                            "post_anomaly_detection_time"
                        ],
                        "detection_lag_weeks": result["detection_lag_weeks"],
                        "false_alarm_rate_year": result["false_alarm_rate_year"],
                    }
                )

            _plot_detection_comparison(
                data_processor=data_processor,
                run_results=run_results,
                anomaly_time=metadata["anomaly_time"],
                combo_dir=combo_dir,
                combo_tag=combo_tag,
                skf_objective=skf_objective,
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = save_dir / "summary_detection_metrics.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print("Saved summary table:", summary_csv_path)

    _plot_summary_heatmap(summary_df, seeds, train_splits, save_dir)


if __name__ == "__main__":
    fire.Fire(main)
