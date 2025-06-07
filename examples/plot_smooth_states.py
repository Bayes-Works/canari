import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from canari import DataProcess
from pytagi import Normalizer as normalizer


def main(L=24):
    # set lookback

    # --- 1. Load the raw CSV (no header) ---
    df_raw = pd.read_csv("linear_state_summary.csv", header=None, skiprows=1)

    # --- 2. Prepare time‐axis ---
    n_time = df_raw.shape[1] - 2
    time_steps = list(range(n_time))

    # --- 3. Filter for only the 'mu' variables (priors, posts, smooths) ---
    mu_df = df_raw[df_raw[1].str.startswith("mu")]
    var_df = df_raw[df_raw[1].str.startswith("var")]

    # read training data
    data_file = "./data/toy_time_series/sine.csv"
    df_raw = pd.read_csv(data_file, skiprows=1, delimiter=",", header=None)

    data_file_time = "./data/toy_time_series/sine_datetime.csv"
    time_series = pd.read_csv(data_file_time, skiprows=1, delimiter=",", header=None)
    time_series = pd.to_datetime(time_series[0])
    df_raw.index = time_series
    df_raw.index.name = "date_time"
    df_raw.columns = ["values"]

    # Resampling data
    df = df_raw.resample("H").mean()
    data_processor = DataProcess(
        data=df,
        train_split=0.8,
        validation_split=0.1,
        output_col=[0],
    )
    train_data = (
        df["values"]
        .iloc[data_processor.train_start : data_processor.train_end]
        .to_list()
    )
    print(len(train_data))

    # --- 4. Plot ---
    plt.figure(figsize=(8, 4))
    cmap = plt.get_cmap("winter")
    colors = cmap(np.linspace(0, 1, len(mu_df)))

    var_dict = {row[1]: row.iloc[2:].astype(float) for _, row in var_df.iterrows()}

    for color, (_, row) in zip(colors, mu_df.iterrows()):
        label = row[1]
        values = row.iloc[2:].astype(float)
        std = np.sqrt(var_dict.get(label.replace("mu", "var"), np.zeros_like(values)))
        print(len(values))
        plt.plot(time_steps, values, color=color, label=label)
        plt.fill_between(time_steps, values - std, values + std, color=color, alpha=0.3)

    # add real data
    plt.plot(
        time_steps[L:],
        train_data,
        color="red",
        label="y",
    )

    plt.axvline(x=L, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend(loc=(0, 1.01), ncol=4)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
