import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pytagi import metric
from pytagi import Normalizer as normalizer
from canari import (
    DataProcess,
    Model,
    Optimizer,
    SKF,
    plot_data,
    plot_skf_states,
)
import canari.common as common
from canari.component import LocalTrend, LocalAcceleration, LstmNetwork, WhiteNoise

# Plotting defaults
import matplotlib as mpl

# Update matplotlib parameters in a single dictionary
mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": False,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
        "lines.linewidth": 1,  # Set line width to 1
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
    "stateful_global_zero_shot": {
        "label": r"$G^{zs}_{sf/w}$",
        "color": "C0",
        "marker": "o",
        "linestyle": "--",
    },
    "stateless_global": {
        "label": r"G$^{ft}_{sl}$",
        "color": "C0",
        "marker": "^",
        "linestyle": "-",
    },
    "stateless_global_zero_shot": {
        "label": r"$G^{zs}_{sl}$",
        "color": "C0",
        "marker": "^",
        "linestyle": "--",
    },
}

DETECTION_THRESHOLD = 0.5


def _format_float_tag(value: float) -> str:
    """Compact float formatting for folder names."""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _trim_trailing_nans(x: np.ndarray, dt: np.ndarray):
    """Trim padded trailing NaNs in the *target* series, keep the same cut for datetime."""
    if len(x) == 0:
        return x, dt
    valid = ~np.isnan(x)
    if not np.any(valid):
        return np.array([], dtype=np.float32), np.array([], dtype="datetime64[ns]")
    last = np.where(valid)[0][-1]
    x = x[: last + 1]
    dt = dt[: last + 1]
    if not np.issubdtype(dt.dtype, np.datetime64):
        dt = np.array(dt, dtype="datetime64[ns]")
    return x.astype(np.float32), dt


def _skf_log_lik_without_hete_noise(skf: SKF, data: dict) -> float:
    """Compute SKF log-likelihood while excluding heteroscedastic-noise state contribution."""

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


def _resolve_mode_settings(model_mode: str, global_lstm_path: str):
    """Resolve loading/training settings from model mode."""
    normalized_mode = model_mode.lower()
    valid_modes = {"global", "local", "zero_shot"}
    if normalized_mode not in valid_modes:
        raise ValueError(
            "`model_mode` must be one of: 'global', 'local', or 'zero_shot'."
        )

    if normalized_mode in {"global", "zero_shot"} and not global_lstm_path:
        raise ValueError(
            "`global_lstm_path` is required when model_mode is 'global' or 'zero_shot'."
        )

    load_lstm_path = (
        global_lstm_path if normalized_mode in {"global", "zero_shot"} else None
    )
    train_lstm = normalized_mode in {"global", "local"}
    return normalized_mode, load_lstm_path, train_lstm


def _saved_model_uses_smoother(global_lstm_path: str):
    """Best-effort detection of whether a saved pyTAGI model uses SLSTM layers."""
    if not global_lstm_path:
        return None

    file_path = Path(global_lstm_path)
    if not file_path.exists():
        return None

    file_text = file_path.read_bytes().decode("latin1", errors="ignore")
    if "SLSTM(" in file_text or "SLinear(" in file_text:
        return True
    if "LSTM(" in file_text or "Linear(" in file_text:
        return False
    return None


def _fix_to_smoothing_layers(global_lstm_path: str):
    """Best-effort fix to convert a saved pyTAGI model with LSTM/Linear layers to use SLSTM/SLinear layers."""
    if not global_lstm_path:
        return None

    file_path = Path(global_lstm_path)
    if not file_path.exists():
        return None

    file_text = file_path.read_bytes().decode("latin1", errors="ignore")
    file_text = file_text.replace("LSTM(", "SLSTM(").replace("Linear(", "SLinear(")
    new_file_path = file_path.with_name(file_path.stem + "_smoother" + file_path.suffix)
    new_file_path.write_bytes(file_text.encode("latin1"))
    return str(new_file_path)


def _first_detection_index(
    model_prob: np.ndarray, threshold: float = DETECTION_THRESHOLD
):
    """Return first index where abnormal probability exceeds threshold."""
    model_prob = np.asarray(model_prob).flatten()
    detected = np.where(model_prob > threshold)[0]
    if len(detected) == 0:
        return None
    return int(detected[0])


def main(
    num_trial_optim_model: int = 70,
    param_optimization: bool = True,
    stateful_global_lstm_path: str = "/Users/davidwardan/Library/CloudStorage/OneDrive-Personal/Projects/canari/saved_params/global_models/ByWindow_global_no-embeddings_seed42.bin",
    stateless_global_lstm_path: str = "/Users/davidwardan/Library/CloudStorage/OneDrive-Personal/Projects/canari/saved_params/global_models/Stateless_global_no-embeddings_seed42.bin",
    skf_objective: str = "cdf",
    smoother: bool = False,
    finetune: bool = True,
    use_tagiv: bool = True,
    train_split: float = 0.5,
    anomaly_slope: float = 1.0,
):
    skf_objective = skf_objective.lower()
    if skf_objective not in {"ll", "cdf"}:
        raise ValueError("`skf_objective` must be either 'll' or 'cdf'.")
    if smoother:
        print(
            "Requested `smoother=True`, but this script enforces `smoother=False` "
            "for all configured runs."
        )
    smoother = False
    save_dir = Path("saved_results") / (
        f"train_split_{_format_float_tag(train_split)}"
        f"_slope_{_format_float_tag(anomaly_slope)}"
        f"_tagiv_{use_tagiv}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    print("Saving plots to:", save_dir.resolve())

    ######### Data processing #########
    # # Read data
    # data_file = "data/benchmark_data/test_2_data.csv"
    # df_raw = pd.read_csv(data_file, skiprows=0, delimiter=",")
    # date_time = pd.to_datetime(df_raw["date"])
    # df_raw = df_raw.drop("date", axis=1)
    # df_raw.index = date_time
    # df_raw.index.name = "date_time"

    # Read data from experiment 01
    ts = 17
    # ts = 18
    df_raw = pd.read_csv(
        "data/exp01_data/ts_weekly_values.csv",
        skiprows=1,
        delimiter=",",
        header=None,
        usecols=[ts],
    )
    df_dates = pd.read_csv(
        "data/exp01_data/ts_weekly_datetimes.csv",
        skiprows=1,
        delimiter=",",
        header=None,
        usecols=[ts],
    )
    values, dates = _trim_trailing_nans(
        df_raw.values.flatten(), df_dates.values.flatten()
    )

    df_raw = pd.DataFrame(values, columns=[0])
    df_raw["Date"] = pd.to_datetime(dates)
    df_raw.set_index("Date", inplace=True)
    df_raw.index.name = "date_time"

    # df = pd.read_csv("data/exp02_data/LGA002EFAPRG910_cleaned.csv")
    # df["Date"] = pd.to_datetime(df["Date"])
    # df.set_index("Date", inplace=True)
    # df.index.name = "date_time"
    # df_raw = df

    # get scaling constants from data processor fitted on raw data (before anomaly injection)
    data_pro_scale = DataProcess(
        data=df_raw,
        time_covariates=["week_of_year"],
        train_split=1.0,
        output_col=[0],
    )

    # Prepare warmup looback
    df_lookback = df_raw.iloc[:52]
    df_raw = df_raw.iloc[52:]
    warmup_lookback_mu = df_lookback.iloc[:, 0].values.flatten()
    warmup_lookback_mu = normalizer.standardize(
        warmup_lookback_mu,
        data_pro_scale.scale_const_mean[data_pro_scale.output_col],
        data_pro_scale.scale_const_std[data_pro_scale.output_col],
    )
    warmup_lookback_var = np.zeros_like(warmup_lookback_mu)

    # Add synthetic anomaly to data
    df = df_raw.copy()
    trend = np.linspace(0, 0, num=len(df_raw))
    time_anomaly = 700  # 200
    new_trend = np.linspace(0, anomaly_slope, num=len(df_raw) - time_anomaly)
    trend[time_anomaly:] = trend[time_anomaly:] + new_trend
    df = df_raw.add(trend, axis=0)

    # Data pre-processing
    data_processor = DataProcess(
        data=df,
        time_covariates=["week_of_year"],
        train_split=train_split,
        validation_split=0.08,
        output_col=[0],
        scale_const_mean=data_pro_scale.scale_const_mean,
        scale_const_std=data_pro_scale.scale_const_std,
    )
    train_data, validation_data, _, all_data = data_processor.get_splits()

    fig_split, ax_split = plt.subplots(figsize=(10, 3))
    plot_data(
        data_processor=data_processor,
        standardization=False,
        plot_column=[0],
        sub_plot=ax_split,
    )
    ax_split.axvline(
        x=df.index[time_anomaly],
        color="k",
        linestyle="--",
        linewidth=1.5,
        label="Anomaly",
    )
    ax_split.legend(loc="best")
    fig_split.tight_layout()
    fig_split_name = save_dir / "DATA_split_anomaly.svg"
    fig_split.savefig(fig_split_name, bbox_inches="tight")
    print("Saved figure:", fig_split_name)
    plt.show()

    # define runs (all with smoother=False)
    run_specs = (
        {
            "run_name": "stateful_local",
            "model_mode": "local",
            "stateless": False,
            "global_lstm_path": stateful_global_lstm_path,
        },
        {
            "run_name": "stateful_global",
            "model_mode": "global",
            "stateless": False,
            "global_lstm_path": stateful_global_lstm_path,
        },
        {
            "run_name": "stateful_global_zero_shot",
            "model_mode": "zero_shot",
            "stateless": False,
            "global_lstm_path": stateful_global_lstm_path,
        },
        {
            "run_name": "stateless_global",
            "model_mode": "global",
            "stateless": True,
            "global_lstm_path": stateless_global_lstm_path,
        },
        {
            "run_name": "stateless_global_zero_shot",
            "model_mode": "zero_shot",
            "stateless": True,
            "global_lstm_path": stateless_global_lstm_path,
        },
    )
    run_results = []
    anomaly_time = df.index[time_anomaly]

    for run_spec in run_specs:
        model_mode = run_spec["model_mode"]
        run_stateless = run_spec["stateless"]
        run_global_lstm_path = run_spec["global_lstm_path"]
        model_mode, load_lstm_path, train_lstm = _resolve_mode_settings(
            model_mode=model_mode,
            global_lstm_path=run_global_lstm_path,
        )
        mode_finetune = finetune if model_mode == "global" else False
        print("\n" + "=" * 70)
        print("Running:", run_spec["run_name"])
        print("Mode:", model_mode)
        if load_lstm_path:
            print("Global weights:", load_lstm_path)
        else:
            print("Global weights: None (local training from scratch)")
        print(
            "Settings:",
            {
                "stateless": run_stateless,
                "smoother": smoother,
                "finetune": mode_finetune,
            },
        )

        ######### Define model with parameters #########
        def model_with_parameters(param):
            model = Model(
                LocalTrend(),
                LstmNetwork(
                    look_back_len=52,
                    num_features=2,
                    num_layer=1 if model_mode == "local" else 3,
                    num_hidden_unit=40,
                    manual_seed=1,
                    infer_len=52 * 3,
                    smoother=smoother,
                    model_noise=use_tagiv,
                    load_lstm_net=load_lstm_path,
                    finetune=mode_finetune,
                    stateless=run_stateless,
                    zero_shot=(model_mode == "zero_shot"),
                ),
                # WhiteNoise(std_error=0.11),
            )

            model.auto_initialize_baseline_states(train_data["y"][0:52])

            if train_lstm:
                num_epoch = 50
                for epoch in range(num_epoch):
                    model.lstm_output_history.set(
                        warmup_lookback_mu, warmup_lookback_var
                    )

                    mu_validation_preds, std_validation_preds, _ = model.lstm_train(
                        train_data=train_data,
                        validation_data=validation_data,
                        white_noise_decay=True if model_mode == "local" else False,
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

                    validation_obs = data_processor.get_data("validation").flatten()
                    validation_log_lik = metric.log_likelihood(
                        prediction=mu_validation_preds_unnorm,
                        observation=validation_obs,
                        std=std_validation_preds_unnorm,
                    )

                    model.early_stopping(
                        evaluate_metric=-validation_log_lik,
                        current_epoch=epoch,
                        max_epoch=num_epoch,
                    )
                    model.metric_optim = model.early_stop_metric

                    if model.stop_training:
                        break
            else:
                print("Using zero-shot mode (loaded global model, no LSTM training)")
                model.lstm_output_history.set(warmup_lookback_mu, warmup_lookback_var)
                model.filter(train_data, train_lstm=False)
                model.smoother()
                model.set_memory(time_step=0)

            #### Define SKF model with parameters #########
            abnorm_model = Model(
                LocalAcceleration(),
                LstmNetwork(model_noise=use_tagiv, stateless=run_stateless),
                # WhiteNoise(),
            )
            model.lstm_output_history.set(warmup_lookback_mu, warmup_lookback_var)
            skf = SKF(
                norm_model=model,
                abnorm_model=abnorm_model,
                std_transition_error=param["std_transition_error"],
                norm_to_abnorm_prob=param["norm_to_abnorm_prob"],
            )

            if run_stateless and not smoother:
                skf.model["norm_norm"].lstm_output_history.set(
                    warmup_lookback_mu, warmup_lookback_var
                )
            skf.save_initial_states()

            if skf_objective == "ll":
                skf.filter(data=all_data)
                log_lik_all = _skf_log_lik_without_hete_noise(skf, all_data)
                skf.metric_optim = -log_lik_all
                skf.print_metric = {"log_lik_all": log_lik_all}
            else:
                num_anomaly = 50
                detection_rate, false_rate, false_alarm_train = (
                    skf.detect_synthetic_anomaly(
                        data=train_data,
                        num_anomaly=num_anomaly,
                        slope_anomaly=param["slope"] / 52,
                    )
                )

                data_len_year = (
                    data_processor.data.index[data_processor.train_end]
                    - data_processor.data.index[data_processor.train_start]
                ).days / 365.25
                false_rate_yearly = false_rate / max(data_len_year, 1e-12)
                metric_optim = skf.objective(
                    detection_rate, false_rate_yearly, param["slope"]
                )

                skf.metric_optim = metric_optim
                skf.print_metric = {
                    "detection_rate": detection_rate,
                    "yearly_false_rate": false_rate_yearly,
                    "false_alarm_train": false_alarm_train,
                }

            skf.load_initial_states()
            return skf

        ######### Parameter optimization #########
        if param_optimization:
            param_space = {
                "std_transition_error": [1e-6, 1e-4],
                "norm_to_abnorm_prob": [1e-6, 1e-4],
            }
            if skf_objective == "cdf":
                param_space["slope"] = [0.1, 0.6]

            # Define optimizer
            model_optimizer = Optimizer(
                model=model_with_parameters,
                param=param_space,
                num_optimization_trial=num_trial_optim_model,
                num_startup_trials=30,
                mode="max" if skf_objective == "cdf" else "min",
            )
            model_optimizer.optimize()
            # Get best model
            param = model_optimizer.get_best_param()
            skf_optim = model_with_parameters(param)

            skf_optim_dict = skf_optim.get_dict()
            skf_optim_dict["model_param"] = param
        else:
            param = {
                "std_transition_error": 5e-5,
                "norm_to_abnorm_prob": 5e-5,
            }
            if skf_objective == "cdf":
                param["slope"] = 0.1
            skf_optim = model_with_parameters(param)
            skf_optim_dict = skf_optim.get_dict()
            skf_optim_dict["model_param"] = param

        ######### Detect anomaly #########
        print("Model parameters used:", skf_optim_dict["model_param"])
        print("SKF optimization objective:", skf_objective)

        if run_stateless and not smoother:
            skf_optim.model["norm_norm"].lstm_output_history.set(
                warmup_lookback_mu, warmup_lookback_var
            )
        filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)
        detection_idx = _first_detection_index(
            model_prob=filter_marginal_abnorm_prob,
            threshold=DETECTION_THRESHOLD,
        )
        detection_time = (
            data_processor.data.index[detection_idx]
            if detection_idx is not None and detection_idx < len(data_processor.data.index)
            else None
        )
        print(
            "First detection time (threshold "
            f"{DETECTION_THRESHOLD:.2f}):",
            detection_time if detection_time is not None else "not detected",
        )
        run_results.append(
            {
                "run_name": run_spec["run_name"],
                "model_mode": model_mode,
                "stateless": run_stateless,
                "model_prob": np.asarray(filter_marginal_abnorm_prob).flatten(),
                "detection_idx": detection_idx,
                "detection_time": detection_time,
            }
        )

        fig, ax = plot_skf_states(
            data_processor=data_processor,
            states=states,
            model_prob=filter_marginal_abnorm_prob,
            standardization=True,
            states_type="posterior",
        )
        ax[-1].axvline(
            x=anomaly_time,
            color="k",
            linestyle="--",
            linewidth=1.2,
            label="Anomaly",
        )
        ax[-1].legend(loc="upper right")
        model_tag = {
            "global": "G",
            "local": "L",
            "zero_shot": "G-zeroshot",
        }[model_mode]
        mode_tag = "stateless" if run_stateless else "stateful"
        fig_name = f"{model_tag}_{mode_tag}_SKF_{skf_objective.upper()}.svg"
        fig_path = save_dir / fig_name
        plt.savefig(fig_path, bbox_inches="tight")
        print("Saved figure:", fig_path)
        plt.close(fig)

    ######### Compare anomaly detection across all runs #########
    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 3))
    for run_result in run_results:
        run_name = run_result["run_name"]
        style = RUN_STYLE[run_name]
        model_prob = run_result["model_prob"]
        n = min(len(data_processor.data.index), len(model_prob))
        time = data_processor.data.index[:n]
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
    fig_cmp_name = f"ALL_detection_compare_SKF_{skf_objective.upper()}.svg"
    fig_cmp_path = save_dir / fig_cmp_name
    fig_cmp.savefig(fig_cmp_path, bbox_inches="tight")
    print("Saved figure:", fig_cmp_path)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
