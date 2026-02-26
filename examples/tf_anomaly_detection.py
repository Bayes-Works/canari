import os
import tqdm
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
    plot_data,
    plot_prediction,
)
from canari.component import (
    LocalTrend,
    LocalAcceleration,
    LstmNetwork,
    WhiteNoise,
)


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


# Define model with parameters
def train_lstm_skf(
    title,
    model,
    data_processor,
    train_set,
    val_set,
    warmup_lookback,
):

    model.auto_initialize_baseline_states(train_set["y"][0:52])

    # Train model
    num_epoch = 100
    pbar = tqdm.tqdm(range(num_epoch), desc=title)
    for epoch in pbar:

        # model.white_noise_decay(epoch, white_noise_max_std=1, white_noise_decay_factor=0.9)

        model.lstm_output_history.set(
            warmup_lookback, np.zeros_like(warmup_lookback)
        )  # important for global model

        # warm-up for infer_len steps
        if model.lstm_net.smooth:
            model.pretraining_filter(train_set)

        model.filter(
            train_set,
            update_embedding=False,
        )

        # forecast on the validation set
        mu_validation_preds, std_validation_preds, _ = model.filter(
            val_set,
            train_lstm=False,
            update_embedding=False,
            yes_init=False,
        )

        # Unstandardize the predictions
        mu_validation_preds = normalizer.unstandardize(
            mu_validation_preds,
            data_processor.scale_const_mean[0],
            data_processor.scale_const_std[0],
        )
        std_validation_preds = normalizer.unstandardize_std(
            std_validation_preds,
            data_processor.scale_const_std[0],
        )

        # Calculate the log-likelihood metric
        validation_obs = data_processor.get_data(
            "validation", standardization=False
        ).flatten()
        mse = metric.mse(mu_validation_preds, validation_obs)
        pbar.set_postfix({"MSE": f"{mse:.4f}"})

        # smooth on train data
        model.smoother()

        model.set_memory(time_step=0)
        model._current_epoch += 1

        # Early-stopping
        model.early_stopping(
            evaluate_metric=mse,
            current_epoch=epoch,
            max_epoch=num_epoch,
            patience=10,
        )
        if epoch == model.optimal_epoch:
            mu_validation_preds_optim = mu_validation_preds
            std_validation_preds_optim = std_validation_preds

        if model.stop_training:
            break

    # plot predictions vs observations for validation set
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data(
        data_processor=data_processor,
        standardization=False,
        plot_column=[0],
        validation_label="y",
    )
    plot_prediction(
        data_processor=data_processor,
        mean_validation_pred=mu_validation_preds_optim,
        std_validation_pred=std_validation_preds_optim,
        validation_label=[r"$\mu$", r"$\pm\sigma$"],
    )
    plt.legend(loc=(0.1, 1.01), ncol=6, fontsize=12)
    plt.tight_layout()
    plt.show()

    return model


TEST_FRACTION = 0.5
VAL_FRACTION = 0.1


def load_data(df, col, train_split):
    # original number of data
    n_total = len(df)
    n_test = int(TEST_FRACTION * n_total)
    n_val = int(VAL_FRACTION * n_total)
    n_train_total = n_total - n_test - n_val

    # scale using the full training set
    train_end_idx = n_train_total
    df_train_full = df.iloc[:train_end_idx]

    # Calculate statistics on the FULL training set
    for_scaling = DataProcess(
        data=df_train_full,
        time_covariates=["week_of_year"],
        train_split=1,
        validation_split=0,
        test_split=0,
        output_col=[0],
    )

    # Now truncate the training set
    # meaningful training data: use only the last train_split fraction of the FULL training data
    n_train_use = int(train_split * n_train_total)

    # Slice the dataframe: [Remaining Train] + [Validation] + [Test]
    # We take the *last* n_train_use rows from the training section
    if n_train_use < n_train_total:
        start_idx = n_train_total - n_train_use
    else:
        start_idx = 0

    df_subset = df.iloc[start_idx:].copy()

    # slice first 52 to use as warmup for the filter
    df_warmup = df_subset.iloc[:52].copy()
    df_subset = df_subset.iloc[52:].copy()

    # get first lookback read first column
    warmup_lookback = df_warmup.iloc[:, 0].values
    warmup_lookback = normalizer.standardize(
        warmup_lookback,
        for_scaling.scale_const_mean[0],
        for_scaling.scale_const_std[0],
    )

    # Recalculate fractions for the NEW subset
    n_subset_total = len(df_subset)

    # The validation and test counts are fixed (n_val, n_test)
    # So we compute their fraction relative to the new total
    new_val_fraction = n_val / n_subset_total
    new_test_fraction = n_test / n_subset_total
    new_train_fraction = 1.0 - new_val_fraction - new_test_fraction

    data_processor = DataProcess(
        data=df_subset,
        time_covariates=["week_of_year"],
        train_split=new_train_fraction,
        validation_split=new_val_fraction,
        test_split=new_test_fraction,
        output_col=[0],
        scale_const_mean=for_scaling.scale_const_mean,
        scale_const_std=for_scaling.scale_const_std,
    )

    return data_processor, warmup_lookback, start_idx


def main(
    experiment_name: str = "G_finetuned_CDF",
    num_trial_optim_model: int = 70,
    model_noise: bool = True,
    pretrained_parameters: str = None,
    zeroshot: bool = False,
    train_split: float = 1.0,
):

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

    # df = pd.read_csv("data/exp02_data/LTU007PIAEVA920_x_cleaned.csv")
    # df["Date"] = pd.to_datetime(df["Date"])
    # df.set_index("Date", inplace=True)
    # df.index.name = "date_time"
    # df_raw =df

    # ofsset time with one week
    # df_raw.index = df_raw.index + pd.Timedelta(weeks=1)

    # Add synthetic anomaly to data
    trend = np.linspace(0, 0, num=len(df_raw))
    time_anomaly = 700  # 200
    new_trend = np.linspace(0, 1, num=len(df_raw) - time_anomaly)
    trend[time_anomaly:] = trend[time_anomaly:] + new_trend
    df_raw = df_raw.add(trend, axis=0)

    # Data pre-processing
    data_processor, warmup_lookback, data_start_idx = load_data(
        df_raw, col=0, train_split=train_split
    )
    train_data, validation_data, _, all_data = data_processor.get_splits()

    model = Model(
        LocalTrend(),
        LstmNetwork(
            look_back_len=52,
            num_features=2,
            num_layer=3,
            num_hidden_unit=40,
            manual_seed=42,
            load_lstm_net=pretrained_parameters,
            finetune=pretrained_parameters is not None and not zeroshot,
            model_noise=model_noise,
            smoother=False,
        ),
        WhiteNoise(std_error=0.11),
    )

    if zeroshot and pretrained_parameters is not None:
        model.lstm_net.load(filename=pretrained_parameters)
        model.auto_initialize_baseline_states(train_data["y"][0:52])
    else:
        model = train_lstm_skf(
            title=experiment_name,
            model=model,
            data_processor=data_processor,
            train_set=train_data,
            val_set=validation_data,
            warmup_lookback=warmup_lookback,
        )

    # Save the trained LSTM parameters and learned initial states
    learned_mu_states = model.mu_states.copy()
    learned_var_states = model.var_states.copy()
    warmup_lookback_mu = warmup_lookback.copy()
    warmup_lookback_var = np.zeros_like(warmup_lookback_mu)

    # Save the trained LSTM to disk so the objective can reconstruct
    # the model without capturing the unpicklable pytagi C++ object.
    import tempfile

    _lstm_tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    _lstm_save_path = _lstm_tmp.name
    _lstm_tmp.close()
    model.lstm_net.save(_lstm_save_path)

    # Drop the reference so the closure never sees the pytagi object
    del model

    def run_skf_with_parameters(param):

        # Reconstruct the full model from scratch to avoid Ray pickle issues
        norm_model = Model(
            LocalTrend(),
            LstmNetwork(
                look_back_len=52,
                num_features=2,
                num_layer=3,
                num_hidden_unit=40,
                manual_seed=42,
                load_lstm_net=_lstm_save_path,
                model_noise=model_noise,
                smoother=False,
            ),
            WhiteNoise(std_error=0.11),
        )

        # Use the learned initial states from training
        norm_model.set_states(learned_mu_states, learned_var_states)
        norm_model.lstm_output_history.set(warmup_lookback_mu, warmup_lookback_var)

        #  Define SKF model with parameters
        abnorm_model = Model(
            LocalAcceleration(),
            LstmNetwork(model_noise=model_noise),
            WhiteNoise(),
        )
        skf = SKF(
            norm_model=norm_model,
            abnorm_model=abnorm_model,
            # std_transition_error=param["std_transition_error"],
            # norm_to_abnorm_prob=param["norm_to_abnorm_prob"],
            std_transition_error=1e-4,
            norm_to_abnorm_prob=1e-5,
        )

        skf.save_initial_states()

        ## Method 1
        skf.filter(data=all_data)
        log_lik_all = np.nanmean(skf.ll_history)
        skf.metric_optim = -log_lik_all
        skf.load_initial_states()

        return skf

    # Parameter optimization
    param_space = {
        "std_transition_error": [1e-6, 1e-4],
        "norm_to_abnorm_prob": [1e-6, 1e-4],
        # "sigma_v": [1e-3, 2e-1],
    }
    # Define optimizer
    model_optimizer = Optimizer(
        model=run_skf_with_parameters,
        param=param_space,
        num_optimization_trial=num_trial_optim_model,
        num_startup_trials=30,
    )

    try:
        # Get best SKF hyperparameters
        model_optimizer.optimize()
        param = model_optimizer.get_best_param()
        skf_optim = run_skf_with_parameters(param)

        ######### Detect anomaly #########
        filter_marginal_abnorm_prob, states = skf_optim.filter(data=all_data)

        fig, ax = plot_skf_states(
            data_processor=data_processor,
            states=states,
            model_prob=filter_marginal_abnorm_prob,
            standardization=True,
        )
        anomaly_idx = time_anomaly - data_start_idx - 52
        ax[0].axvline(
            x=data_processor.data.index[anomaly_idx],
            color="r",
            linestyle="--",
            label="Anomaly",
        )
        ax[5].axvline(
            x=data_processor.data.index[anomaly_idx],
            color="r",
            linestyle="--",
            label="Anomaly",
        )
        fig.suptitle("SKF hidden states", fontsize=10, y=1)
        plt.savefig(f"results/{experiment_name}.svg", bbox_inches="tight")
        plt.show()
    finally:
        if os.path.exists(_lstm_save_path):
            os.remove(_lstm_save_path)


if __name__ == "__main__":
    main(
        experiment_name="agvi-check-local",
        num_trial_optim_model=70,
        # pretrained_parameters="saved_params/global_models/ByWindow_global-small_no-embeddings_seed42_whitenoise.bin",
        pretrained_parameters="saved_params/global_models/ByWindow_global_no-embeddings_seed42_whitenoise.bin",
        # pretrained_parameters="saved_params/global_models/ByWindow_global_no-embeddings_seed42.bin",
        model_noise=False,
        zeroshot=False,
        train_split=1.0,
    )
