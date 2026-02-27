import fire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def main(
    num_trial_optim_model: int = 70,
    param_optimization: bool = True,
    zero_shot: bool = True,
    skf_objective: str = "ll",
):
    skf_objective = skf_objective.lower()
    if skf_objective not in {"ll", "cdf"}:
        raise ValueError("`skf_objective` must be either 'll' or 'cdf'.")

    ######### Data processing #########
    # # Read data
    # data_file = "data/benchmark_data/test_2_data.csv"
    # df_raw = pd.read_csv(data_file, skiprows=0, delimiter=",")
    # date_time = pd.to_datetime(df_raw["date"])
    # df_raw = df_raw.drop("date", axis=1)
    # df_raw.index = date_time
    # df_raw.index.name = "date_time"

    # # Read data from experiment 01
    # ts = 17
    # # ts = 18
    # df_raw = pd.read_csv(
    #     "data/exp01_data/ts_weekly_values.csv",
    #     skiprows=1,
    #     delimiter=",",
    #     header=None,
    #     usecols=[ts],
    # )
    # df_dates = pd.read_csv(
    #     "data/exp01_data/ts_weekly_datetimes.csv",
    #     skiprows=1,
    #     delimiter=",",
    #     header=None,
    #     usecols=[ts],
    # )
    # values, dates = _trim_trailing_nans(
    #     df_raw.values.flatten(), df_dates.values.flatten()
    # )

    # df_raw = pd.DataFrame(values, columns=[0])
    # df_raw["Date"] = pd.to_datetime(dates)
    # df_raw.set_index("Date", inplace=True)
    # df_raw.index.name = "date_time"

    df = pd.read_csv("data/exp02_data/LGA002EFAPRG910_cleaned.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.index.name = "date_time"
    df_raw = df

    # # slice first n weeks of data for faster training and testing
    df_lookback = df_raw.iloc[:52]
    df_raw = df_raw.iloc[52:]

    # plt.figure(figsize=(10, 4))
    # plt.plot(df_raw.index, df_raw[0], label="Original data")
    # plt.show()

    # Add synthetic anomaly to data (optional)
    df = df_raw.copy()
    time_anomaly = None
    trend = np.linspace(0, 0, num=len(df_raw))
    time_anomaly = 700  # 200
    if 0 <= time_anomaly < len(df_raw):
        new_trend = np.linspace(0, 0.25, num=len(df_raw) - time_anomaly)
        trend[time_anomaly:] = trend[time_anomaly:] + new_trend
    else:
        time_anomaly = None
    df = df_raw.add(trend, axis=0)

    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df,
        time_covariates=["week_of_year"],
        train_split=0.4,
        validation_split=0.08,
        output_col=output_col,
    )
    train_data, validation_data, _, all_data = data_processor.get_splits()

    fig_split, ax_split = plt.subplots(figsize=(10, 4))
    plot_data(
        data_processor=data_processor,
        standardization=False,
        plot_column=output_col,
        sub_plot=ax_split,
        train_label="Train",
        validation_label="Validation",
        test_label="Test",
    )
    if time_anomaly is not None and 0 <= time_anomaly < len(df):
        ax_split.axvline(
            x=df.index[time_anomaly],
            color="r",
            linestyle="--",
            linewidth=1.2,
            label="Synthetic anomaly start",
        )

    ax_split.set_title("Data splits and anomaly time")
    ax_split.legend(loc="best")
    fig_split.tight_layout()
    plt.show()

    # prepare warmup lookback data for LSTM training
    warmup_lookback_mu = df_lookback.iloc[:, output_col].values.flatten()

    # Normalize data
    warmup_lookback_mu = normalizer.standardize(
        warmup_lookback_mu,
        data_processor.scale_const_mean[data_processor.output_col],
        data_processor.scale_const_std[data_processor.output_col],
    )
    warmup_lookback_var = np.zeros_like(warmup_lookback_mu)
    # load_lstm_path = "/Users/davidwardan/Library/CloudStorage/OneDrive-Personal/Projects/canari/saved_params/global_models/Stateless_global_no-embeddings_seed42.bin"
    load_lstm_path = None
    lstm_stateless = True

    ######### Define model with parameters #########
    def model_with_parameters(param):
        model = Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=52,
                num_features=2,
                num_layer=1,
                num_hidden_unit=40,
                manual_seed=42,
                smoother=False,
                model_noise=True,
                load_lstm_net=load_lstm_path,
                finetune=False,
                stateless=lstm_stateless,
            ),
        )

        model.auto_initialize_baseline_states(train_data["y"][0:52*4])

        if len(warmup_lookback_mu) != model.lstm_net.lstm_look_back_len:
            raise ValueError(
                "Warmup lookback length must match LSTM look_back_len "
                f"({len(warmup_lookback_mu)} != {model.lstm_net.lstm_look_back_len})"
            )

        if zero_shot:
            print("Using zero-shot model without LSTM finetuning")
        else:
            num_epoch = 50
            for epoch in range(num_epoch):
                model.lstm_output_history.set(warmup_lookback_mu, warmup_lookback_var)
                mu_validation_preds, std_validation_preds, states = model.lstm_train(
                    train_data=train_data,
                    validation_data=validation_data,
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

        #### Define SKF model with parameters #########
        abnorm_model = Model(
            LocalAcceleration(),
            LstmNetwork(model_noise=True),
            # WhiteNoise(),
        )
        model.lstm_output_history.set(warmup_lookback_mu, warmup_lookback_var)
        skf = SKF(
            norm_model=model,
            abnorm_model=abnorm_model,
            std_transition_error=param["std_transition_error"],
            norm_to_abnorm_prob=param["norm_to_abnorm_prob"],
        )

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
            "std_transition_error": [1e-5, 1e-3],
            "norm_to_abnorm_prob": [1e-5, 1e-3],
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
            "std_transition_error": 1e-4,
            "norm_to_abnorm_prob": 1e-3,
        }
        if skf_objective == "cdf":
            param["slope"] = 0.3
        skf_optim = model_with_parameters(param)
        skf_optim_dict = skf_optim.get_dict()
        skf_optim_dict["model_param"] = param

    ######### Detect anomaly #########
    print("Model parameters used:", skf_optim_dict["model_param"])
    print("SKF optimization objective:", skf_objective)

    skf_optim.model["norm_norm"].lstm_output_history.set(
        warmup_lookback_mu, warmup_lookback_var
    )
    filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)

    fig, ax = plot_skf_states(
        data_processor=data_processor,
        states=states,
        model_prob=filter_marginal_abnorm_prob,
        standardization=True,
    )
    if time_anomaly is not None and 0 <= time_anomaly < len(df):
        anomaly_time = df.index[time_anomaly]
        ax[-1].axvline(
            x=anomaly_time,
            color="r",
            linestyle="--",
            linewidth=1.2,
            label="Synthetic anomaly start",
        )
        ax[-1].legend(loc="upper right")
    fig.suptitle("SKF hidden states", fontsize=10, y=1)
    model_tag = "L"
    if load_lstm_path is not None:
        model_tag = "G-zeroshot" if zero_shot else "G"
    mode_tag = "stateless" if lstm_stateless else "stateful"
    fig_name = f"{model_tag}_{mode_tag}_SKF_{skf_objective.upper()}.svg"
    plt.savefig(fig_name)
    print("Saved figure:", fig_name)
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
