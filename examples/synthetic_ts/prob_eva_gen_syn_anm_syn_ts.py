import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from canari.component import LocalTrend, LstmNetwork, Periodic, Autoregression
from canari import (
    DataProcess,
    Model,
    plot_data,
    plot_prediction,
    plot_states,
)
import pickle
from tqdm import tqdm
from matplotlib import gridspec
from pytagi import Normalizer as normalizer


# # Read data
data_file = "./data/toy_time_series/syn_data_simple_phi05.csv"
df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)
time_series = pd.to_datetime(df_raw.iloc[:, 0])
df_raw = df_raw.iloc[:, 1:]
df_raw.index = time_series
df_raw.index.name = "date_time"
df_raw.columns = ["obs"]

# Data pre-processing
output_col = [0]
train_split=0.3
validation_split=0.1
data_processor = DataProcess(
    data=df_raw,
    time_covariates=["week_of_year"],
    train_split=train_split,
    validation_split=validation_split,
    output_col=output_col,
)
train_data, validation_data, test_data, normalized_data = data_processor.get_splits()

# Get scale constants
scale_const_mean = copy.deepcopy(data_processor.scale_const_mean)
scale_const_std = copy.deepcopy(data_processor.scale_const_std)

num_test_ts = 10
time_series_all = []
anm_mag_all = np.concatenate([np.arange(0.01, 0.11, 0.01), np.arange(0.2, 1.01, 0.1)])/52

# Generate num_test_ts * len(anm_mag_all) stationary synthetic time series
model = Model(
    LocalTrend(
        mu_states=[0, 0],
        var_states=[1e-12, 1e-12],
        std_error=0,
    ),  
    Periodic(period=52, mu_states=[0, 5 * 5], var_states=[1e-12, 1e-12]),
    Periodic(period=13, mu_states=[0, 10], var_states=[1e-12, 1e-12]),
    Autoregression(
        std_error=2, phi=0.5, mu_states=[-0.0621], var_states=[6.36e-05]
    ),
)
num_time_steps = 52 * 12
gen_ts, _, _, _ = model.generate_time_series(num_time_series=num_test_ts*len(anm_mag_all),
                                    num_time_steps=num_time_steps,
                                    )
train_val_data_raw = df_raw["obs"].values[0:data_processor.validation_end]

# # Normalize the generated time series
# gen_ts = (gen_ts - scale_const_mean[0])/(scale_const_std[0] + 1e-10)

for i in range(gen_ts.shape[0]):
    gen_ts[i, :(len(df_raw) - len(test_data["y"]))] = train_val_data_raw


# Put anomalies in the synthetic time series
for i, anm_mag in tqdm(enumerate(anm_mag_all)):
    for k in range(num_test_ts):

        # Detection window: 3 years
        anm_start_index = np.random.randint(0, len(test_data["y"]) - 52 * 3)
        anm_start_index_global = anm_start_index + len(df_raw) - len(test_data["y"])

        sign = -1. if np.random.rand() < 0.5 else 1. # Randomly assign positive and negative anomalies
        anm_mag *= sign
        anm_mag_unstandardize = anm_mag * (scale_const_std[0] + 1e-10)  # Unstandardize the anomaly magnitude

        anm_baseline = np.arange(len(df_raw)) * anm_mag_unstandardize
        # Set the first 52*12 values in anm_baseline to be 0
        anm_baseline[anm_start_index_global:] -= anm_baseline[anm_start_index_global]
        anm_baseline[:anm_start_index_global] = 0

        # gen_df_raw = copy.deepcopy(df_raw)
        # gen_df_raw = gen_df_raw.add(anm_baseline, axis=0)

        # gen_data_processor = DataProcess(
        #     data=gen_df_raw,
        #     time_covariates=["week_of_year"],
        #     train_split=train_split,
        #     validation_split=validation_split,
        #     output_col=output_col,
        # )

        # values_str = str(list(gen_df_raw.values.flatten()))
        # time_series_all.append([values_str, anm_mag, anm_start_index])

        gen_anm_ts = gen_ts[i*num_test_ts + k, :] + anm_baseline
        values_str = str(list(gen_anm_ts))
        time_series_all.append([values_str, anm_mag, anm_start_index])

        # #  Plot
        # state_type = "prior"
        # #  Plot states from pretrained model
        # fig = plt.figure(figsize=(10, 2))
        # gs = gridspec.GridSpec(1, 1)
        # ax0 = plt.subplot(gs[0])
        # time = gen_data_processor.get_time(split="all")
        # plot_data(
        #     data_processor=gen_data_processor,
        #     standardization=True,
        #     plot_column=output_col,
        #     validation_label="y",
        #     sub_plot=ax0,
        # )
        # ax0.axvline(x=time[anm_start_index_global], color='r', linestyle='--')
        # plt.show()

# Save to CSV
df_time_series_all = pd.DataFrame(time_series_all, columns=["values", "anomaly_magnitude", "anomaly_start_index"])
df_time_series_all.to_csv("data/prob_eva_syn_time_series/syn_simple_ts_regen2_val.csv", index=False)