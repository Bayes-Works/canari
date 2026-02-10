import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_states
from tqdm import tqdm
from canari.component import LstmNetwork, WhiteNoise


# Helper function
def adjust_params(model, mode="add", value=1e-2, threshold=5e-4, which_layer=None):
    """
    Adjusts the variances of weights and biases in the network's state dictionary.

    For each layer (or specified layers), if a variance value is below the given threshold,
    either adds or sets it to the specified value depending on the mode.

    Args:
        net: The neural network whose parameters will be modified.
        mode (str): "add" to increment variances, "set" to assign the value directly.
        value (float): The value to add or set for the variances.
        threshold (float): Variances below this value will be adjusted.
        which_layer (list or None): List of layer names to adjust. If None, all layers are adjusted.

    Returns:
        None. The network's parameters are updated in-place.
    """

    state_dict = model.lstm_net.state_dict()
    for layer_name, (mu_w, var_w, mu_b, var_b) in state_dict.items():
        if which_layer is None or layer_name in which_layer:
            if mode.lower() == "add":
                var_w = [x + value if x < threshold else x for x in var_w]
                var_b = [x + value if x < threshold else x for x in var_b]
            elif mode.lower() == "set":
                var_w = [value if x < threshold else x for x in var_w]
                var_b = [value if x < threshold else x for x in var_b]
            state_dict[layer_name] = (mu_w, var_w, mu_b, var_b)
    model.lstm_net.load_state_dict(state_dict)


# Define the model building function
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
        # WhiteNoise(std_error=0.1),
    )

    return model


# Load data
def load_data(df, col, train_split=0.8):

    for_scaling = DataProcess(
        data=df,
        time_covariates=["week_of_year"],
        train_split=1.0,
        output_col=[col],
    )

    data_processor = DataProcess(
        data=df,
        time_covariates=["week_of_year"],
        train_split=train_split,
        validation_split=0.2,
        output_col=[col],
    )

    return data_processor


# run experiment
def run_experiment(
    title, df, col, finetune, train_split, seed, layers=3, param=None, embedding=None
):

    # Extract first lookback for start
    first_lookback = df.iloc[:52, col].values.flatten()
    df = df.iloc[52:]

    # Load data
    data_processor, scalers = load_data(df, col, train_split=train_split)
    train_data, validation_data, test_data, normalized_data = (
        data_processor.get_splits()
    )

    # Prepare first lookback
    train_mean = data_processor.scale_const_mean[0]
    train_std = data_processor.scale_const_std[0]
    first_lookback = (first_lookback - train_mean) / train_std

    # Build model
    model = build_model(
        param=param, embedding=embedding, seed=seed, finetune=finetune, layers=layers
    )

    if finetune:

        # Train model
        num_epoch = 100
        pbar = tqdm(range(num_epoch), desc="Training")
        for epoch in pbar:

            model.lstm_output_history.set(first_lookback, np.zeros_like(first_lookback))
            model.filter(
                train_data, update_embedding=True if embedding is not None else False
            )

            # forecast on the validation set
            mu_validation_preds, std_validation_preds, _ = model.forecast(
                validation_data
            )

            # Unstandardize the predictions
            mu_validation_preds = normalizer.unstandardize(
                mu_validation_preds,
                data_processor.scale_const_mean[col],
                data_processor.scale_const_std[col],
            )
            std_validation_preds = normalizer.unstandardize_std(
                std_validation_preds,
                data_processor.scale_const_std[col],
            )

            # Calculate the log-likelihood metric
            validation_obs = data_processor.get_data(
                "validation", standardization=False
            ).flatten()
            mse = metric.mse(mu_validation_preds, validation_obs)
            pbar.set_postfix({"MSE": f"{mse:.4f}"})

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

            # smooth on train data
            model.smoother()

            model.set_memory(time_step=0)
            model.lstm_net.reset_lstm_states()
            model._current_epoch += 1

            if model.stop_training:
                break

        print(f"Optimal epoch       : {model.optimal_epoch}")
        print(f"Validation MSE      :{model.early_stop_metric: 0.4f}")

    # filter now but on all the data
    model.lstm_output_history.set(first_lookback, np.zeros_like(first_lookback))
    mu_preds, std_preds, _ = model.filter(
        normalized_data,
        train_lstm=False,
        update_embedding=embedding is not None and not finetune,
    )

    # plot data and states
    fig, axes = plot_states(
        data_processor=data_processor,
        states=model.states,
        states_type="prior",
        standardization=True,
    )
    plot_data(
        data_processor=data_processor,
        plot_column=[0],
        validation_label="y",
        sub_plot=axes[0],
        standardization=True,
    )
    plt.tight_layout()
    plt.savefig(f"./saved_results/experiment2/{title}.svg")
    plt.close()
    plt.show()

    # Calculate test metrics
    mu_preds_unstd = normalizer.unstandardize(
        mu_preds,
        data_processor.scale_const_mean[col],
        data_processor.scale_const_std[col],
    )
    std_preds_unstd = normalizer.unstandardize_std(
        std_preds,
        data_processor.scale_const_std[col],
    )
    obs = data_processor.get_data("all", standardization=False).flatten()

    test_mse = metric.mse(mu_preds_unstd, obs)
    test_mae = np.nanmean(np.abs(mu_preds_unstd - obs))
    # Log-likelihood: sum of log N(obs | mu, std^2) over all non-NaN points
    test_loglik = np.nansum(
        -0.5 * np.log(2 * np.pi)
        - np.log(std_preds_unstd)
        - 0.5 * ((obs - mu_preds_unstd) / std_preds_unstd) ** 2
    )

    return {"MSE": test_mse, "MAE": test_mae, "LogLik": test_loglik}


def run_all_experiments(df, col, train_split, seed):
    """Run all experiments for a given seed and train_split, return dict of results."""
    experiments = {}

    # Local Model
    experiments["Local Model"] = run_experiment(
        title=f"local_model_s{seed}_t{train_split}",
        df=df,
        col=col,
        train_split=train_split,
        finetune=True,
        param=None,
        seed=seed,
        layers=2,
    )

    # Global Model no embeddings
    experiments["Global Zero-Shot"] = run_experiment(
        title=f"global_model_zero_shot_s{seed}_t{train_split}",
        df=df,
        col=col,
        train_split=train_split,
        finetune=False,
        param="saved_params/global_models/ByWindow_Obs_global_no-embeddings.bin",
        seed=seed,
    )

    experiments["Global Fine-Tuned"] = run_experiment(
        title=f"global_model_finetune_s{seed}_t{train_split}",
        df=df,
        col=col,
        train_split=train_split,
        finetune=True,
        param="saved_params/global_models/BySeries_Obs_global_no-embeddings.bin",
        seed=seed,
    )

    # Global Model with simple per series embeddings
    np.random.seed(seed)
    embed_dim = 10
    embed_mu = np.random.randn(embed_dim)
    embed_var = np.full(embed_dim, 1.0)

    experiments["Global SimpEmbed Fine-Tuned"] = run_experiment(
        title=f"global_embed_finetune_s{seed}_t{train_split}",
        df=df,
        col=col,
        train_split=train_split,
        finetune=True,
        param="saved_params/global_models/BySeries_Obs_global_simple-embeddings.bin",
        seed=seed,
        embedding=(embed_mu, embed_var),
    )

    # Global Model with hierarchical embeddings
    np.random.seed(seed)
    sensor_embed_dim = 2
    sensor_embed_mu = np.random.randn(sensor_embed_dim)
    sensor_embed_var = np.full(sensor_embed_dim, 1.0)

    # read the saved embeddings
    hierembedding_path = "saved_params/embeddings/embeddings_final"
    dam_embed = np.load(f"{hierembedding_path}_dam_id.npz")
    dam_type_embed = np.load(f"{hierembedding_path}_dam_type_id.npz")
    sensor_type_embed = np.load(f"{hierembedding_path}_sensor_type_id.npz")
    direction_embed = np.load(f"{hierembedding_path}_direction_id.npz")

    # Define embedding map
    embedding_map = (2, 1, 0, 0)  # LGA

    embed_mu = np.concatenate(
        [
            dam_embed["mu"][embedding_map[0]],
            dam_type_embed["mu"][embedding_map[1]],
            sensor_type_embed["mu"][embedding_map[2]],
            direction_embed["mu"][embedding_map[3]],
            sensor_embed_mu,
        ]
    )
    embed_var = np.concatenate(
        [
            dam_embed["var"][embedding_map[0]],
            dam_type_embed["var"][embedding_map[1]],
            sensor_type_embed["var"][embedding_map[2]],
            direction_embed["var"][embedding_map[3]],
            sensor_embed_var,
        ]
    )

    experiments["Global HierEmbed Fine-Tuned"] = run_experiment(
        title=f"global_hierembed_finetune_s{seed}_t{train_split}",
        df=df,
        col=col,
        train_split=train_split,
        finetune=True,
        param="saved_params/global_models/BySeries_Obs_global_hierarchical-embedding.bin",
        seed=seed,
        embedding=(embed_mu, embed_var),
    )

    experiments["Global HierEmbed Zero-Shot"] = run_experiment(
        title=f"global_hierembed_zeroshot_s{seed}_t{train_split}",
        df=df,
        col=col,
        train_split=train_split,
        finetune=False,
        param="saved_params/global_models/BySeries_Obs_global_hierarchical-embedding.bin",
        seed=seed,
        embedding=(embed_mu, embed_var),
    )

    return experiments


def print_summary_table(all_results, seeds, train_splits):
    """Print a summary table: mean±std across seeds, columns per train_split."""
    metric_names = ["MSE", "MAE", "LogLik"]
    experiment_names = list(
        next(iter(next(iter(all_results.values())).values())).keys()
    )

    for metric_name in metric_names:
        # Determine best (min for MSE/MAE, max for LogLik)
        higher_better = metric_name == "LogLik"

        col_width = 22
        header = f"{'Experiment':<30}"
        for ts in train_splits:
            header += f" | {f'split={ts}':^{col_width}}"
        print("\n" + "=" * len(header))
        print(f"{metric_name}")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        # Collect all mean values to find the best per column
        table_data = {}
        for exp_name in experiment_names:
            row_vals = []
            for ts in train_splits:
                values = [
                    all_results[ts][s][exp_name][metric_name]
                    for s in seeds
                    if all_results[ts][s][exp_name] is not None
                ]
                mean_val = np.mean(values)
                row_vals.append(mean_val)
            table_data[exp_name] = row_vals

        # Find best per column
        best_per_col = []
        for col_idx in range(len(train_splits)):
            col_vals = [(name, table_data[name][col_idx]) for name in experiment_names]
            if higher_better:
                best_name = max(col_vals, key=lambda x: x[1])[0]
            else:
                best_name = min(col_vals, key=lambda x: x[1])[0]
            best_per_col.append(best_name)

        # Print rows
        for exp_name in experiment_names:
            row = f"{exp_name:<30}"
            for col_idx, ts in enumerate(train_splits):
                values = [
                    all_results[ts][s][exp_name][metric_name]
                    for s in seeds
                    if all_results[ts][s][exp_name] is not None
                ]
                mean_val = np.mean(values)
                if len(values) > 1:
                    std_val = np.std(values)
                    cell = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    cell = f"{mean_val:.4f}"
                if exp_name == best_per_col[col_idx]:
                    cell += " **"
                row += f" | {cell:^{col_width}}"
            print(row)
        print("=" * len(header))


def main(
    seeds=None,
    data_path="data/exp02_data/LGA002EFAPRG910_cleaned.csv",
    train_splits=None,
):
    if seeds is None:
        seeds = [42]
    if train_splits is None:
        train_splits = [0.8]

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.index.name = "date_time"

    # all_results[train_split][seed] = {experiment_name: {MSE, MAE, LogLik}}
    all_results = defaultdict(dict)

    for ts in train_splits:
        for seed in seeds:
            print(f"\n>>> train_split={ts}, seed={seed}")
            all_results[ts][seed] = run_all_experiments(
                df=df, col=0, train_split=ts, seed=seed
            )

    print_summary_table(all_results, seeds, train_splits)


if __name__ == "__main__":
    main(
        seeds=[42],
        data_path="data/exp02_data/LTU012PEAP-E379_cleaned.csv",
        train_splits=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
