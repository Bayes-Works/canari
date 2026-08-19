"""Filter a full raw time series with a saved, fine-tuned LSTM SKF.

Run this example from the repository root after training the model with
``examples/anomaly_detection_global_lstm.py``:

    python examples/anomaly_detection_global_lstm_deploy.py

All input and output settings are constants below; no configuration file is read.
"""

import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from canari import DataProcess, SKF, plot_skf_states

SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (6.5, 3.5)

mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}",
        "lines.linewidth": 1,
        "figure.figsize": SINGLE_COL,
        "font.size": 9,
        "savefig.dpi": 300,
    }
)

ROOT = Path(__file__).resolve().parents[1]

SERIES_NAME = "ts50"

DATA_PATH = ROOT / "data/BM_detrend_data/weekly/weekly_values_raw.csv"
DATETIME_PATH = ROOT / "data/BM_detrend_data/weekly/weekly_datetimes_raw.csv"
SAVED_SKF_PATH = ROOT / f"saved_params/{SERIES_NAME}/anomaly_detection_lstm_finetuned.pkl"
OUTPUT_PATH = ROOT / f"saved_results/{SERIES_NAME}_results/anomaly_detection_lstm_filtered.csv"
FIGURE_PATH = ROOT / f"saved_results/{SERIES_NAME}_results/anomaly_detection_global_lstm_deploy"

BASELINE_INIT_LEN = 2 * 52

def read_time_series():
    values = pd.read_csv(DATA_PATH)[SERIES_NAME]
    datetimes = pd.to_datetime(pd.read_csv(DATETIME_PATH)[SERIES_NAME])

    last_valid_index = values.last_valid_index()
    values = values.iloc[: last_valid_index + 1]
    datetimes = datetimes.iloc[: last_valid_index + 1]
    return pd.DataFrame(
        {"values": values.to_numpy()},
        index=pd.DatetimeIndex(datetimes, name="date_time"),
    )


def prepare_full_series(dataframe, preprocessing):
    data_processor = DataProcess(
        data=dataframe,
        time_covariates=preprocessing["time_covariates"],
        train_split=1.0,
        validation_split=0.0,
        output_col=preprocessing["output_col"],
        standardization=preprocessing["standardization"],
        scale_const_mean=preprocessing["scale_const_mean"],
        scale_const_std=preprocessing["scale_const_std"],
    )
    _, _, _, all_data = data_processor.get_splits()
    return data_processor, all_data


def plot_results(data_processor, states, anomaly_probability, threshold):
    figure, axes = plot_skf_states(
        data_processor=data_processor,
        states=states,
        states_type="posterior",
        model_prob=anomaly_probability,
        color="tab:blue",
    )
    figure.set_size_inches(DOUBLE_COL[0], 1.1 * len(axes))

    for axis in axes[:-1]:
        for uncertainty_band in axis.collections:
            uncertainty_band.set_alpha(0.3)
    for observation_line in axes[0].lines:
        if observation_line.get_color() in {"r", "red"}:
            observation_line.set_color("tab:red")

    probability_axis = axes[-1]
    probability_axis.lines[0].set_color("tab:blue")
    probability_axis.lines[0].set_label(r"$p(\mathrm{abnormal})$")
    probability_axis.axhline(
        threshold,
        color="black",
        linestyle="--",
        label=rf"threshold $={threshold:.2f}$",
    )
    probability_axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        ncol=2,
    )

    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH.with_suffix(".pdf"))
    figure.savefig(FIGURE_PATH.with_suffix(".pgf"))
    plt.close(figure)


def main():
    dataframe = read_time_series()
    with SAVED_SKF_PATH.open("rb") as file:
        saved_model = pickle.load(file)

    data_processor, all_data = prepare_full_series(
        dataframe, saved_model["preprocessing"]
    )
    skf = SKF.load_dict(saved_model)
    skf.auto_initialize_baseline_states(all_data["y"][:BASELINE_INIT_LEN])
    anomaly_probability, states = skf.filter(data=all_data)
    threshold = float(saved_model["threshold"])

    results = pd.DataFrame(
        {
            "values": dataframe["values"].to_numpy(),
            "anomaly_probability": anomaly_probability,
            "is_anomaly": anomaly_probability > threshold,
        },
        index=dataframe.index,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH)
    plot_results(data_processor, states, anomaly_probability, threshold)

    print(f"Loaded complete fine-tuned SKF model from {SAVED_SKF_PATH}")
    print(f"Filtered {len(results)} observations with threshold {threshold:.3f}")
    print(f"Detected {int(results['is_anomaly'].sum())} anomalous observations")
    print(f"Saved filtering results to {OUTPUT_PATH}")
    print(f"Saved SKF figure to {FIGURE_PATH.with_suffix('.pdf')}")
    print(f"Saved SKF figure to {FIGURE_PATH.with_suffix('.pgf')}")


if __name__ == "__main__":
    main()
