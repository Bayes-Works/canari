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
    zero_shot: bool = False,
):

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

    # # slice first n weeks of data for faster training and testing
    df_lookback = df_raw.iloc[:52]
    df_raw = df_raw.iloc[52:]

    # plt.figure(figsize=(10, 4))
    # plt.plot(df_raw.index, df_raw[0], label="Original data")
    # plt.show()

    # df = pd.read_csv("data/exp02_data/LTU007PIAEVA920_x_cleaned.csv")
    # df["Date"] = pd.to_datetime(df["Date"])
    # df.set_index("Date", inplace=True)
    # df.index.name = "date_time"
    # df_raw =df

    # ofsset time with one week
    # df_raw.index = df_raw.index + pd.Timedelta(weeks=1)

    # Add synthetic anomaly to data (optional)
    df = df_raw.copy()
    time_anomaly = None
    trend = np.linspace(0, 0, num=len(df_raw))
    time_anomaly = 700  # 200
    new_trend = np.linspace(0, 1, num=len(df_raw) - time_anomaly)
    trend[time_anomaly:] = trend[time_anomaly:] + new_trend
    df = df_raw.add(trend, axis=0)

    # Data pre-processing
    output_col = [0]
    data_processor = DataProcess(
        data=df,
        time_covariates=["week_of_year"],
        train_split=0.25,
        validation_split=0.08,
        output_col=output_col,
    )
    train_data, validation_data, _, all_data = data_processor.get_splits()

    # prepare warmup lookback data for LSTM training
    warmup_lookback_mu = df_lookback.iloc[:, output_col].values.flatten()

    # Normalize data
    warmup_lookback_mu = normalizer.standardize(
        warmup_lookback_mu,
        data_processor.scale_const_mean[data_processor.output_col],
        data_processor.scale_const_std[data_processor.output_col],
    )
    warmup_lookback_var = np.zeros_like(warmup_lookback_mu)

    ######### Define model with parameters #########
    def model_with_parameters(param):
        model = Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=52,
                num_features=2,
                num_layer=3,
                num_hidden_unit=40,
                manual_seed=42,
                smoother=False,
                model_noise=True,
                load_lstm_net="/Users/davidwardan/Library/CloudStorage/OneDrive-Personal/Projects/canari/saved_params/global_models/Stateless_global_no-embeddings_seed42.bin",
                # load_lstm_net="/Users/davidwardan/Library/CloudStorage/OneDrive-Personal/Projects/canari/saved_params/global_models/ByWindow_global_no-embeddings_seed42.bin",
                finetune=True,
                stateless=True,
            ),
        )

        model.auto_initialize_baseline_states(train_data["y"][0 : 52*4])

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

        skf.save_initial_states()

        skf.model["norm_norm"].lstm_output_history.set(
            warmup_lookback_mu, warmup_lookback_var
        )
        skf.filter(data=all_data)
        log_lik_all = _skf_log_lik_without_hete_noise(skf, all_data)
        skf.metric_optim = -log_lik_all

        skf.load_initial_states()
        return skf

    ######### Parameter optimization #########
    if param_optimization:
        param_space = {
            "std_transition_error": [1e-6, 1e-4],
            "norm_to_abnorm_prob": [1e-6, 1e-4],
        }
        # Define optimizer
        model_optimizer = Optimizer(
            model=model_with_parameters,
            param=param_space,
            num_optimization_trial=num_trial_optim_model,
            num_startup_trials=30,
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
            "norm_to_abnorm_prob": 1e-5,
        }
        skf_optim = model_with_parameters(param)
        skf_optim_dict = skf_optim.get_dict()
        skf_optim_dict["model_param"] = param

    ######### Detect anomaly #########
    print("Model parameters used:", skf_optim_dict["model_param"])

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
    if time_anomaly is not None:
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
    plt.savefig(f"L2_SKF_LL_obj.svg")
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
