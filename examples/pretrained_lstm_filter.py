import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytagi.metric as metric
from pytagi import Normalizer as normalizer
from canari import DataProcess, Model, plot_data, plot_prediction, plot_states
from canari.component import LstmNetwork, WhiteNoise


# Possible bugs to checks:
# TODO: check time covariates compatibility with pretrained models


# data path
data_path = "./data/exp01_data"
# sensor_csv = "PEA06AB-080001_x_cleaned.csv"
sensor_csv = "ts_weekly_values.csv"

# read data
df = pd.read_csv(f"{data_path}/{sensor_csv}", usecols=[17])

# read date_time from another csv
df_time = pd.read_csv("data/exp01_data/ts_weekly_values.csv", usecols=[17])

df.index = pd.to_datetime(df_time.iloc[:, 0])

# shift index by lag of 1 week
# df.index = df.index - pd.Timedelta(weeks=1)

# Define parameters
output_col = [0]

# first look back is the 52 first rows of the data
first_lookback = df.iloc[:52, output_col].values.flatten()

# remove from df
df = df.iloc[52:]

# Build data processor
data_processor = DataProcess(
    data=df,
    time_covariates=["week_of_year"],
    train_split=0.7,
    output_col=output_col,
)

# split data
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Normalize first_lookback using training data statistics
train_mean = data_processor.scale_const_mean[0]
train_std = data_processor.scale_const_std[0]
first_lookback = normalizer.standardize(first_lookback, train_mean, train_std)

# input data for functions

## Global No Embeddings
global_noembeddings_params = (
    "saved_params/global_models/BySeries_Obs_global_no-embeddings.bin"
)
# global_noembeddings_params = "saved_params/global_models/global_noembeddings_seed1.pth"

## Global Hierarchical Embeddings
global_hierarchicalembeddings_params = (
    "saved_params/global_models/global_hierarchicalembeddings_seed1.pth"
)
global_hierarchicalembeddings_path = (
    "saved_params/global_models/hierarchical_embed/embeddings_final"
)

LTU_EXT_embedding_map = (
    3,
    1,
    1,
    0,
)  # (dam_id, dam_type_id, sensor_type_id, direction_id)
LTU_PEN_embedding_map = (3, 1, 2, 1)  # change last value to set direction X Y Z
LTU_PIZ_embedding_map = (3, 1, 0, 0)

LGA_EXT_embedding_map = (2, 0, 1, 0)
LGA_PIZ_embedding_map = (2, 1, 0, 0)

M5_PEN_embedding_map = (5, 0, 2, 1)  # change last value to set direction X Y Z

embedding_map = LTU_EXT_embedding_map

## Global Autoencoder Embeddings
global_semisuperembeddings_params = (
    "saved_params/global_models/global_semisuperembeddings_seed1.pth"
)
global_semisuper_embed = [
    -0.24480787,
    -0.6251097,
    -0.28016204,
    -0.5189541,
    -0.45802015,
    -0.3749219,
    0.16958192,
    0.59203553,
    -0.14351466,
    -0.39392793,
]


def run_global_no_embeddings(first_lookback, param, train_lstm=False):
    model = Model(
        LstmNetwork(
            look_back_len=52,
            num_features=2,
            num_layer=3,
            num_hidden_unit=40,
            device="cpu",
            manual_seed=11,
            model_noise=True,
            smoother=False,
            load_lstm_net=param,
        ),
        # WhiteNoise(std_error=0.01),
    )

    # set first lookback from actual data
    model.lstm_output_history.mu = first_lookback
    model.lstm_output_history.var = np.zeros_like(first_lookback)

    print("First lookback (normalized):", first_lookback)

    # filter on all data
    mu_filter, std_filter, states_history = model.filter(
        train_data, train_lstm=train_lstm, update_embedding=False
    )

    # mean and std
    mu_preds = states_history.get_mean("lstm", "posterior", standardization=True)
    std_preds = states_history.get_std("lstm", "posterior", standardization=True)

    # calculate the test metrics
    # test_obs = data_processor.get_data("train").flatten()
    # mse = metric.mse(mu_preds, test_obs)
    # log_lik = metric.log_likelihood(mu_preds, test_obs, std_preds)

    # print(f"Test MSE            :{mse: 0.4f}")
    # print(f"Test Log-Lik        :{log_lik: 0.2f}")

    # plot states
    plt.figure(figsize=(10, 4))
    plt.plot(
        data_processor.get_data("train", standardization=True).flatten(), color="r"
    )
    plt.plot(mu_filter, label="Filtered Mean")
    # plt.plot(mu_forecast, color="k", label="Forecasted Mean")
    plt.fill_between(
        np.arange(len(mu_filter)),
        mu_filter - std_filter,
        mu_filter + std_filter,
        color="b",
        alpha=0.3,
    )
    # plt.fill_between(
    #     np.arange(len(mu_forecast)),
    #     mu_forecast - 2 * std_forecast,
    #     mu_forecast + 2 * std_forecast,
    #     color="k",
    #     alpha=0.2,
    # )
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.show()


def run_global_hierarchicalembeddings(
    first_lookback, param, embed_path, embedding_map, train_lstm=False
):
    np.random.seed(1)  # Ensure reproducibility

    dam_embed = np.load(f"{embed_path}_dam_id.npz")
    dam_type_embed = np.load(f"{embed_path}_dam_type_id.npz")
    sensor_type_embed = np.load(f"{embed_path}_sensor_type_id.npz")
    direction_embed = np.load(f"{embed_path}_direction_id.npz")

    # sensor_embed is newly generated for each sensor
    sensor_embedding_dim = 2
    sensor_embed_mu = np.random.randn(sensor_embedding_dim)
    sensor_embed_var = np.full(sensor_embedding_dim, 1.0)

    # build embedding
    # build embedding vector mu and var
    embedd_mu = np.concatenate(
        [
            dam_embed["mu"][embedding_map[0]],
            dam_type_embed["mu"][embedding_map[1]],
            sensor_type_embed["mu"][embedding_map[2]],
            direction_embed["mu"][embedding_map[3]],
            sensor_embed_mu,
        ]
    )
    embedd_var = np.concatenate(
        [
            dam_embed["var"][embedding_map[0]],
            dam_type_embed["var"][embedding_map[1]],
            sensor_type_embed["var"][embedding_map[2]],
            direction_embed["var"][embedding_map[3]],
            sensor_embed_var,
        ]
    )

    model = Model(
        LstmNetwork(
            look_back_len=52,
            num_features=2,
            num_layer=3,
            num_hidden_unit=40,
            device="cpu",
            manual_seed=1,
            model_noise=True,
            smoother=False,
            load_lstm_net=param,
            embedding=(embedd_mu, embedd_var),
        ),
        # WhiteNoise(std_error=0.01),
    )

    # set first lookback from actual data
    model.lstm_output_history.mu = first_lookback
    model.lstm_output_history.var = np.ones_like(first_lookback)

    # filter on all data
    mu_filter, std_filter, states_history = model.filter(
        train_data,
        train_lstm=train_lstm,
        update_embedding=True,
        update_mask=[
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
        ],
    )

    # set first lookback from actual data
    model.lstm_output_history.mu = first_lookback
    model.lstm_output_history.var = np.ones_like(first_lookback)
    model.lstm_net.reset_lstm_states()
    mu_forecast, std_forecast, _ = model.forecast(train_data)

    # mean and std
    mu_preds = states_history.get_mean("lstm", "posterior", standardization=True)
    std_preds = states_history.get_std("lstm", "posterior", standardization=True)

    # calculate the test metrics
    # test_obs = data_processor.get_data("train").flatten()
    # mse = metric.mse(mu_preds, test_obs)
    # log_lik = metric.log_likelihood(mu_preds, test_obs, std_preds)

    # print(f"Test MSE            :{mse: 0.4f}")
    # print(f"Test Log-Lik        :{log_lik: 0.2f}")

    # plot states
    plt.figure(figsize=(10, 4))
    plt.plot(
        data_processor.get_data("train", standardization=True).flatten(), color="r"
    )
    plt.plot(mu_filter, label="Filtered Mean")
    plt.plot(mu_forecast, color="k", label="Forecasted Mean")
    plt.fill_between(
        np.arange(len(mu_filter)),
        mu_filter - 2 * std_filter,
        mu_filter + 2 * std_filter,
        color="b",
        alpha=0.2,
    )
    plt.fill_between(
        np.arange(len(mu_forecast)),
        mu_forecast - 2 * std_forecast,
        mu_forecast + 2 * std_forecast,
        color="k",
        alpha=0.2,
    )
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.show()


def run_global_semisuperembeddings(first_lookback, param, embed, train_lstm=False):
    np.random.seed(1)  # Ensure reproducibility

    embed_mu = np.array(embed)
    embed_var = np.ones_like(embed_mu)

    model = Model(
        LstmNetwork(
            look_back_len=52,
            num_features=2,
            num_layer=3,
            num_hidden_unit=40,
            device="cpu",
            manual_seed=1,
            model_noise=True,
            smoother=False,
            load_lstm_net=param,
            embedding=(embed_mu, embed_var),
        ),
        # WhiteNoise(std_error=0.01),
    )

    # set first lookback from actual data
    model.lstm_output_history.mu = first_lookback
    model.lstm_output_history.var = np.zeros_like(first_lookback)

    # filter on all data
    mu_filter, std_filter, states_history = model.filter(
        train_data, train_lstm=train_lstm, update_embedding=True
    )

    # mean and std
    mu_preds = states_history.get_mean("lstm", "posterior", standardization=True)
    std_preds = states_history.get_std("lstm", "prior", standardization=True)

    # calculate the test metrics
    test_obs = data_processor.get_data("train").flatten()
    mse = metric.mse(mu_preds, test_obs)
    log_lik = metric.log_likelihood(mu_preds, test_obs, std_preds)

    # print(f"Test MSE            :{mse: 0.4f}")
    # print(f"Test Log-Lik        :{log_lik: 0.2f}")

    # plot states
    plt.figure(figsize=(10, 4))
    plt.plot(mu_filter)
    plt.plot(mu_preds, color="g")

    plt.fill_between(
        np.arange(len(mu_filter)),
        mu_filter - 2 * std_filter,
        mu_filter + 2 * std_filter,
        color="b",
        alpha=0.2,
    )
    plt.plot(data_processor.get_data("train").flatten(), color="r", alpha=0.5)
    plt.xlabel("Time Steps")
    plt.ylabel("Predicted Values")
    plt.show()


if __name__ == "__main__":
    train_lstm = False

    run_global_no_embeddings(first_lookback, global_noembeddings_params, train_lstm)
    # run_global_hierarchicalembeddings(
    #     first_lookback,
    #     global_hierarchicalembeddings_params,
    #     global_hierarchicalembeddings_path,
    #     embedding_map,
    #     train_lstm,
    # )
    # run_global_semisuperembeddings(
    #     first_lookback,
    #     global_semisuperembeddings_params,
    #     global_semisuper_embed,
    #     train_lstm,
    # )
