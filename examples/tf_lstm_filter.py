import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)
from scipy.stats import norm
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction
from tqdm import tqdm
from canari.component import LstmNetwork, WhiteNoise
from canari import common


import matplotlib as mpl

# Plotting defaults
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
# Helper functions


# Build model
def build_model(seed=1, param=None, embedding=None, finetune=False, layers=3):
    model = Model(
        LstmNetwork(
            look_back_len=52,
            num_features=2,
            num_layer=layers,
            num_hidden_unit=40,
            device="cpu",
            manual_seed=seed,
            model_noise=True,
            smoother=False,
            load_lstm_net=param,
            embedding=embedding,
            finetune=finetune,
        ),
    )

    return model


# Load data
TEST_FRACTION = 0.2
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

    return data_processor, warmup_lookback


# run tuning experiment
def train_model(
    title,
    model,
    data_processor,
    train_set,
    val_set,
    warmup_lookback,
    stateless=False,
    embedding=None,
    embedding_update_mask=None,
):

    # Train model
    num_epoch = 100
    pbar = tqdm(range(num_epoch), desc=f"Training {title}")
    for epoch in pbar:
        model.lstm_output_history.set(
            warmup_lookback, np.zeros_like(warmup_lookback)
        )  # important for global model

        model.filter(
            train_set,
            update_embedding=True if embedding is not None else False,
            update_mask=embedding_update_mask,
        )

        # forecast on the validation set
        mu_validation_preds, std_validation_preds, _ = model.filter(
            val_set,
            train_lstm=False,
            update_embedding=False,
            yes_init=False,
            stateless=stateless,
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
            states_optim = copy.copy(
                model.states
            )  # If we want to plot the states, plot those from optimal epoch

        if model.stop_training:
            break

    return model


def filter_model(model, data_processor, all_data, warmup_lookback, stateless=False):
    # reset memory
    try:
        model.set_memory(time_step=0)
    except:
        print("Could not reset memory")
        pass

    model.lstm_output_history.set(
        warmup_lookback, np.zeros_like(warmup_lookback)
    )  # important for global model
    mu_test_preds, std_test_preds, _ = model.filter(
        all_data,
        update_embedding=False,
        stateless=stateless,
    )

    # slice mu_test_preds and std_test_preds to match the test data
    mu_test_preds = mu_test_preds[data_processor.get_split_indices()[2]]
    std_test_preds = std_test_preds[data_processor.get_split_indices()[2]]

    return (mu_test_preds, std_test_preds)


def plot_results(data_processor, mu_test_preds, std_test_preds, title):

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data(
        data_processor=data_processor,
        standardization=True,
        plot_column=[0],
        validation_label="y",
    )
    plot_prediction(
        data_processor=data_processor,
        mean_test_pred=mu_test_preds,
        std_test_pred=std_test_preds,
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


METRIC_NAMES = ["rmse", "mae", "p50", "p90", "loglik"]


def calculate_metrics(data_processor, mu_test_preds, std_test_preds):
    """Calculate MSE, MAE, P50, P90, and log-likelihood."""
    y_true = data_processor.get_data("test", standardization=True).flatten()
    y_pred = mu_test_preds.flatten()
    y_std = std_test_preds.flatten()

    # Filter out NaN entries
    valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_std))
    y_true, y_pred, y_std = y_true[valid], y_pred[valid], y_std[valid]

    abs_errors = np.abs(y_true - y_pred)
    loglik = float(np.mean(norm.logpdf(y_true, loc=y_pred, scale=y_std)))

    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "p50": float(np.percentile(abs_errors, 50)),
        "p90": float(np.percentile(abs_errors, 90)),
        "loglik": loglik,
    }


def run_experiments(df, train_split, seed):

    metrics = {
        "local_model": {},
        "local_model_stateless": {},
        "global_model_ft": {},
        "global_model_ft_stateless": {},
        "global_model_zs": {},
        "global_model_zs_stateless": {},
    }

    # load data
    data_processor, warmup_lookback = load_data(df, [0], train_split)
    train_set, val_set, test_set, all_data = data_processor.get_splits()

    # 1-1.Run local model
    title = f"Local Model_s{seed}_t{train_split}"
    local_model = build_model(seed=seed, param=None)
    local_model = train_model(
        title, local_model, data_processor, train_set, val_set, warmup_lookback
    )
    local_model_preds = filter_model(
        local_model, data_processor, all_data, warmup_lookback
    )
    local_model_metrics = calculate_metrics(
        data_processor, local_model_preds[0], local_model_preds[1]
    )
    metrics["local_model"] = local_model_metrics
    # plot_results(data_processor, local_model_preds[0], local_model_preds[1], title)

    # 1-2.Run local model stateless
    title = f"Local Model_s{seed}_t{train_split}"
    local_model = build_model(seed=seed, param=None)
    local_model = train_model(
        title,
        local_model,
        data_processor,
        train_set,
        val_set,
        warmup_lookback,
        stateless=True,
    )
    local_model_preds = filter_model(
        local_model, data_processor, all_data, warmup_lookback, stateless=True
    )
    local_model_metrics = calculate_metrics(
        data_processor, local_model_preds[0], local_model_preds[1]
    )
    metrics["local_model_stateless"] = local_model_metrics
    # plot_results(data_processor, local_model_preds[0], local_model_preds[1], title)

    # 2-1. Run global model with finetuning
    title = f"Global Model_s{seed}_t{train_split}"
    global_model_ft = build_model(
        seed=seed,
        param=f"saved_params/global_models/ByWindow_global_no-embeddings_seed{seed}.bin",
        finetune=True,
    )
    global_model_ft = train_model(
        title, global_model_ft, data_processor, train_set, val_set, warmup_lookback
    )
    global_model_ft_preds = filter_model(
        global_model_ft, data_processor, all_data, warmup_lookback
    )
    global_model_ft_metrics = calculate_metrics(
        data_processor, global_model_ft_preds[0], global_model_ft_preds[1]
    )
    metrics["global_model_ft"] = global_model_ft_metrics
    # plot_results(
    #     data_processor, global_model_ft_preds[0], global_model_ft_preds[1], title
    # )

    # 2-2. Run global model with finetuning stateless
    title = f"Global Model_s{seed}_t{train_split}"
    global_model_ft = build_model(
        seed=seed,
        param=f"saved_params/global_models/Stateless_global_no-embeddings_seed{seed}.bin",
        finetune=True,
    )
    global_model_ft = train_model(
        title,
        global_model_ft,
        data_processor,
        train_set,
        val_set,
        warmup_lookback,
        stateless=True,
    )
    global_model_ft_preds = filter_model(
        global_model_ft, data_processor, all_data, warmup_lookback, stateless=True
    )
    global_model_ft_metrics = calculate_metrics(
        data_processor, global_model_ft_preds[0], global_model_ft_preds[1]
    )
    metrics["global_model_ft_stateless"] = global_model_ft_metrics
    # plot_results(
    #     data_processor, global_model_ft_preds[0], global_model_ft_preds[1], title
    # )

    # 3-1. Global model zeroshot
    title = f"Global Model Zeroshot_s{seed}_t{train_split}"
    global_model_zs = build_model(
        seed=seed,
        param=f"saved_params/global_models/ByWindow_global_no-embeddings_seed{seed}.bin",
        finetune=False,
    )
    global_model_zs_preds = filter_model(
        global_model_zs, data_processor, all_data, warmup_lookback
    )
    global_model_zs_metrics = calculate_metrics(
        data_processor, global_model_zs_preds[0], global_model_zs_preds[1]
    )
    metrics["global_model_zs"] = global_model_zs_metrics
    # plot_results(
    #     data_processor, global_model_zs_preds[0], global_model_zs_preds[1], title
    # )

    # 3-2. Global model zeroshot stateless
    title = f"Global Model Zeroshot_s{seed}_t{train_split}"
    global_model_zs = build_model(
        seed=seed,
        param=f"saved_params/global_models/Stateless_global_no-embeddings_seed{seed}.bin",
        finetune=False,
    )
    global_model_zs_preds = filter_model(
        global_model_zs, data_processor, all_data, warmup_lookback, stateless=True
    )
    global_model_zs_metrics = calculate_metrics(
        data_processor, global_model_zs_preds[0], global_model_zs_preds[1]
    )
    metrics["global_model_zs_stateless"] = global_model_zs_metrics
    # plot_results(
    #     data_processor, global_model_zs_preds[0], global_model_zs_preds[1], title
    # )

    return metrics


def aggregate_across_series(all_series_raw, train_splits, model_keys):
    """Aggregate results: macro-average over series, then mean ± std over seeds.

    For each seed, compute the macro average of each metric across all time
    series.  Then report mean ± std of those per-seed macro averages.

    Args:
        all_series_raw: dict mapping series_name -> {seed -> {train_split -> {model_key -> dict}}}
        train_splits: list of train split fractions
        model_keys: list of model key names

    Returns:
        dict mapping model_key -> {"<metric>_mean": [...], "<metric>_std": [...]}
    """
    # Collect all seeds that appear in the data
    all_seeds = set()
    for seed_dict in all_series_raw.values():
        all_seeds.update(seed_dict.keys())
    all_seeds = sorted(all_seeds)

    aggregated = {}
    for mk in model_keys:
        agg_data = {f"{m}_mean": [] for m in METRIC_NAMES}
        agg_data.update({f"{m}_std": [] for m in METRIC_NAMES})
        for ts in train_splits:
            # For each seed, compute the macro average across series
            seed_macros = {m: [] for m in METRIC_NAMES}
            for seed in all_seeds:
                series_vals = {m: [] for m in METRIC_NAMES}
                for series_name, seed_dict in all_series_raw.items():
                    split_dict = seed_dict.get(seed, {})
                    result = split_dict.get(ts, {}).get(mk, None)
                    if result and isinstance(result, dict):
                        for m in METRIC_NAMES:
                            if m in result:
                                series_vals[m].append(result[m])
                for m in METRIC_NAMES:
                    if series_vals[m]:
                        seed_macros[m].append(np.mean(series_vals[m]))

            for m in METRIC_NAMES:
                if seed_macros[m]:
                    agg_data[f"{m}_mean"].append(np.mean(seed_macros[m]))
                    agg_data[f"{m}_std"].append(np.std(seed_macros[m]))
                else:
                    agg_data[f"{m}_mean"].append(np.nan)
                    agg_data[f"{m}_std"].append(np.nan)

        for key in agg_data:
            agg_data[key] = np.array(agg_data[key])
        aggregated[mk] = agg_data
    return aggregated


# Style config for plots
MODEL_STYLE = {
    "local_model": {
        "label": r"L$_{sf}$",
        "color": "C3",
        "marker": "o",
        "linestyle": "-",
    },
    "local_model_stateless": {
        "label": r"L$_{sl}$",
        "color": "C3",
        "marker": "^",
        "linestyle": "-",
    },
    "global_model_ft": {
        "label": r"G$^{ft}_{sf/w}$",
        "color": "C0",
        "marker": "o",
        "linestyle": "-",
    },
    "global_model_ft_stateless": {
        "label": r"G$^{ft}_{sl}$",
        "color": "C0",
        "marker": "^",
        "linestyle": "-",
    },
    "global_model_zs": {
        "label": r"$G^{zs}_{sf/w}$",
        "color": "C0",
        "marker": "o",
        "linestyle": "--",
    },
    "global_model_zs_stateless": {
        "label": r"$G^{zs}_{sl}$",
        "color": "C0",
        "marker": "^",
        "linestyle": "--",
    },
}

PLOT_METRICS = [
    ("rmse", r"RMSE ($\downarrow$)"),
    ("mae", r"MAE ($\downarrow$)"),
    ("p50", r"P50 ($\downarrow$)"),
    ("p90", r"P90 ($\downarrow$)"),
    ("loglik", r"Log-Likelihood ($\uparrow$)"),
]

SAVE_DIR = "saved_results/experiment2"


def plot_metrics(aggregated, train_splits):
    """Create one SVG plot per metric, matching reference script style."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_pcts = [int(ts * 100) for ts in train_splits]

    # Explicit plot order from MODEL_STYLE
    plot_order = list(MODEL_STYLE.keys())

    for metric_key, metric_label in PLOT_METRICS:
        fig = plt.figure(figsize=(5, 3))
        ax = fig.gca()

        for mk in plot_order:
            data = aggregated.get(mk)
            if data is None:
                continue
            style = MODEL_STYLE[mk]
            means = data.get(f"{metric_key}_mean")
            stds = data.get(f"{metric_key}_std")
            if means is None or np.all(np.isnan(means)):
                continue

            ax.errorbar(
                train_pcts,
                means,
                yerr=np.where(np.isnan(stds), 0, stds),
                fmt=f"{style['linestyle']}{style['marker']}",
                color=style["color"],
                label=style["label"],
                markersize=4,
                alpha=0.7,
            )

        ax.set_xlabel("Train size (%)")
        ax.set_ylabel(metric_label)
        ax.set_xticks(train_pcts)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                ncol=1,
                frameon=False,
                fontsize="x-small",
                handletextpad=0.4,
            )

        fig.subplots_adjust(right=0.70)
        fig.tight_layout()
        fname = os.path.join(SAVE_DIR, f"{metric_key}.svg")
        fig.savefig(fname, bbox_inches="tight")
        print(f"  Saved {fname}")
        plt.close(fig)


def print_summary_table(aggregated, train_splits):
    """Print a formatted summary table of metrics."""
    col_labels = [
        ("RMSE", "rmse"),
        ("MAE", "mae"),
        ("P50", "p50"),
        ("P90", "p90"),
        ("LogLik", "loglik"),
    ]

    for ts_idx, ts in enumerate(train_splits):
        print(f"\n  Train split = {ts}")
        # Header row
        header = f"{'Model':<25}"
        for label, _ in col_labels:
            header += f"{label:>16}"
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for mk, data in aggregated.items():
            if np.all(np.isnan(data["rmse_mean"])):
                continue
            table_labels = {
                "local_model": "Local",
                "local_model_stateless": "Local Stateless",
                "global_model_ft": "Global Fine-Tuned",
                "global_model_ft_stateless": "Global Fine-Tuned Stateless",
                "global_model_zs": "Global Zero-Shot",
                "global_model_zs_stateless": "Global Zero-Shot Stateless",
            }
            label = table_labels.get(mk, mk)
            row = f"{label:<25}"
            for _, key in col_labels:
                mean_val = data[f"{key}_mean"][ts_idx]
                std_val = data[f"{key}_std"][ts_idx]
                row += f"{mean_val:>8.4f}±{std_val:<5.4f}"
            print(row)

        print("=" * len(header))


SEEDS = [11, 42, 3]
TRAIN_SPLITS = [1, 0.8, 0.6, 0.4]
MODEL_KEYS = [
    "local_model",
    "local_model_stateless",
    "global_model_ft",
    "global_model_ft_stateless",
    "global_model_zs",
    "global_model_zs_stateless",
]


def main(df, series_name=""):
    """Run all experiments for a single time series.

    Returns:
        dict mapping seed -> {train_split -> {model_key -> (mse, loglik)}}
    """
    all_seed_results = {}
    for seed in SEEDS:
        all_seed_results[seed] = {}
        for train_split in TRAIN_SPLITS:
            all_seed_results[seed][train_split] = run_experiments(df, train_split, seed)
    return all_seed_results


if __name__ == "__main__":
    all_data_path = "data/exp02_data/"
    data_list = sorted(os.listdir(all_data_path))

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Collect raw results: series_name -> {seed -> {train_split -> {model_key -> (mse, loglik)}}}
    all_series_raw = {}
    for data_file in data_list:
        if not data_file.endswith(".csv"):
            continue
        series_name = data_file.replace(".csv", "")
        print(f"\n{'#'*60}")
        print(f"# Processing: {series_name}")
        print(f"{'#'*60}")

        # Load from individual CSV file
        df = pd.read_csv(os.path.join(all_data_path, data_file))
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.index.name = "date_time"

        all_series_raw[series_name] = main(df, series_name=series_name)

    # ── Final aggregation across ALL series and seeds ──
    aggregated = aggregate_across_series(all_series_raw, TRAIN_SPLITS, MODEL_KEYS)

    n_series = len(all_series_raw)
    n_seeds = len(SEEDS)
    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS  ({n_series} series × {n_seeds} seeds)")
    print(f"{'='*60}")
    print_summary_table(aggregated, TRAIN_SPLITS)
    plot_metrics(aggregated, TRAIN_SPLITS)
