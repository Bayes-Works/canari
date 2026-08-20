"""Compare six offline/online LSTM strategies on an OOD validation span.

The model and weekly benchmark series match ``offline_vs_online_lstm.py``.  Only the
first two 52-week years are available for offline training; every later step is held out
for validation.  Validation predictions are always one-step-ahead filter predictions,
never open-loop forecasts.

Run from the repository root:

    python examples/ood_experiment.py

Results are written below ``OOD_experiments/out``.
"""

from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "OOD_experiments"
OUT_ROOT = EXPERIMENT_ROOT / "out"

# Matplotlib is imported by canari, so its writable cache must be configured first.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "canari-matplotlib")
)

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pytagi import metric  # noqa: E402

from canari import DataProcess, Model  # noqa: E402
from canari.component import LstmNetwork, WhiteNoise  # noqa: E402

SINGLE_COL = (3.5, 2.5)
DOUBLE_COL = (6.5, 3.5)

PLOT_STYLE = {
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

SERIES = "MAT001PIAP-F510_x_cleaned"
VALUES_PATH = ROOT / "data/exp01_data/ts_weekly_values.csv"
DATETIMES_PATH = ROOT / "data/exp01_data/ts_weekly_datetimes.csv"
GLOBAL_MEANS_PATH = ROOT / "saved_params/global_BM_52_256.bin"

WEEKS_PER_YEAR = 52
TRAIN_YEARS = 2
TRAIN_STEPS = TRAIN_YEARS * WEEKS_PER_YEAR
NUM_EPOCHS = 100
ONLINE_WINDOW_LEN = 52
SIGMA_V = 0.1

Initialization = Literal["local", "global"]
ValidationMode = Literal["frozen", "online"]
OnlineStart = Literal["validation", "beginning"]


@dataclass(frozen=True)
class Scenario:
    """Configuration for one requested comparison."""

    number: int
    slug: str
    label: str
    initialization: Initialization
    offline_pretraining: bool
    validation_mode: ValidationMode
    online_start: OnlineStart | None

    @property
    def output_dir(self) -> Path:
        return OUT_ROOT / f"scenario_{self.number:02d}_{self.slug}"


@dataclass
class ScenarioResult:
    """Validation predictions and metrics for one scenario."""

    scenario: Scenario
    mean: np.ndarray
    std: np.ndarray
    mse: float
    mean_log_likelihood: float
    n_observed: int
    offline_epochs: int


SCENARIOS = (
    Scenario(
        1,
        "offline_local_frozen",
        "offline local, frozen",
        "local",
        True,
        "frozen",
        None,
    ),
    Scenario(
        2,
        "offline_global_frozen",
        "offline global means, frozen",
        "global",
        True,
        "frozen",
        None,
    ),
    Scenario(
        3,
        "offline_local_online",
        "offline local + fixed-lag online",
        "local",
        True,
        "online",
        "validation",
    ),
    Scenario(
        4,
        "offline_global_online",
        "offline global means + fixed-lag online",
        "global",
        True,
        "online",
        "validation",
    ),
    Scenario(
        5,
        "online_local_from_start",
        "fixed-lag online local from start",
        "local",
        False,
        "online",
        "beginning",
    ),
    Scenario(
        6,
        "online_global_from_start",
        "fixed-lag online global means from start",
        "global",
        False,
        "online",
        "beginning",
    ),
)

RESIDUAL_STYLES = (
    ("tab:blue", "-"),
    ("tab:orange", "--"),
    ("tab:green", "-."),
    ("tab:purple", ":"),
    ("tab:brown", (0, (5, 2))),
    ("tab:gray", (0, (3, 1, 1, 1))),
)


def _require_file(path: Path, purpose: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def load_weekly_series() -> pd.DataFrame:
    """Load the same timestamped series as ``offline_vs_online_lstm.py``."""

    _require_file(VALUES_PATH, "weekly values file")
    _require_file(DATETIMES_PATH, "weekly datetimes file")

    values = pd.read_csv(VALUES_PATH)[SERIES]
    datetimes = pd.read_csv(DATETIMES_PATH)[SERIES]
    series = pd.DataFrame({"date_time": pd.to_datetime(datetimes), SERIES: values})
    series = series.dropna(subset=["date_time"]).set_index("date_time")
    if not series.index.is_monotonic_increasing or not series.index.is_unique:
        raise ValueError("Expected unique, increasing weekly timestamps.")
    return series


def build_data_processor(series: pd.DataFrame) -> DataProcess:
    """Use 104 weekly steps for training and every remaining step for validation."""

    if len(series) <= TRAIN_STEPS:
        raise ValueError(
            f"The series needs more than {TRAIN_STEPS} rows for a validation span."
        )

    train_split = TRAIN_STEPS / len(series)
    validation_split = (len(series) - TRAIN_STEPS) / len(series)
    processor = DataProcess(
        data=series,
        time_covariates=["week_of_year"],
        train_split=train_split,
        validation_split=validation_split,
        output_col=[0],
    )
    if (
        processor.validation_start != TRAIN_STEPS
        or processor.validation_end != len(series)
        or processor.test_start != processor.test_end
    ):
        raise RuntimeError("DataProcess did not preserve the requested 104/rest split.")
    return processor


def load_data() -> DataProcess:
    return build_data_processor(load_weekly_series())


def build_model(initialization: Initialization = "local") -> Model:
    """Build the reference architecture, optionally transferring global means."""

    model = Model(
        LstmNetwork(
            look_back_len=52,
            num_features=2,
            infer_len=52 * 3,
            num_hidden_unit=256,
            num_layer=1,
            manual_seed=1,
        ),
        WhiteNoise(std_error=SIGMA_V),
    )
    if initialization == "global":
        _require_file(GLOBAL_MEANS_PATH, "global LSTM parameter file")
        model.load_lstm_parameter_means(GLOBAL_MEANS_PATH)
    elif initialization != "local":
        raise ValueError(f"Unknown initialization: {initialization}")
    return model


def train_offline(
    initialization: Initialization,
    train_data: dict[str, np.ndarray],
    data_processor: DataProcess,
    num_epochs: int = NUM_EPOCHS,
) -> dict:
    """Train only on the two-year span and return a reusable t=0 snapshot.

    ``Model.lstm_train`` normally forecasts a supplied validation set.  Supplying an
    empty validation dictionary retains its training/smoothing implementation without
    exposing any held-out observations or covariates to offline model selection.
    """

    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1")

    model = build_model(initialization)
    empty_validation = {
        "x": train_data["x"][:0],
        "y": train_data["y"][:0],
    }
    print(f"Offline training ({initialization}) for {num_epochs} fixed epochs")
    for epoch in range(num_epochs):
        _, _, states = model.lstm_train(
            train_data=train_data,
            validation_data=empty_validation,
            data_processor=data_processor,
        )
        model.set_memory(states=states, time_step=0)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == num_epochs:
            print(f"  epoch {epoch + 1:>3}/{num_epochs}")

    return copy.deepcopy(model.get_dict())


def run_frozen_validation(
    model: Model,
    train_data: dict[str, np.ndarray],
    validation_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Replay training context, then filter validation with frozen parameters."""

    model.lstm_net.num_samples = len(train_data["y"]) + len(validation_data["y"])
    model.lstm_net.eval()
    model.filter(train_data, train_lstm=False)
    mean, std, _ = model.filter(validation_data, train_lstm=False)
    return np.asarray(mean).flatten(), np.asarray(std).flatten()


def run_online_validation(
    model: Model,
    all_data: dict[str, np.ndarray],
    validation_start: int,
    validation_end: int,
    online_start: OnlineStart,
) -> tuple[np.ndarray, np.ndarray]:
    """Run fixed-lag online filtering and return only validation-aligned predictions."""

    prediction_start = (
        validation_start if online_start == "validation" else ONLINE_WINDOW_LEN
    )
    mean, std, _ = model.online_lstm_filter(
        data=all_data,
        start=prediction_start,
        window_len=ONLINE_WINDOW_LEN,
        end=validation_end,
    )
    validation_offset = validation_start - prediction_start
    mean = np.asarray(mean).flatten()[validation_offset:]
    std = np.asarray(std).flatten()[validation_offset:]
    expected = validation_end - validation_start
    if len(mean) != expected or len(std) != expected:
        raise RuntimeError(
            f"Expected {expected} validation predictions, got {len(mean)} and {len(std)}."
        )
    return mean, std


def score_validation(
    mean: np.ndarray, std: np.ndarray, observations: np.ndarray
) -> dict[str, float | int]:
    """Score the common set of observed validation steps in standardized units."""

    mean = np.asarray(mean).flatten()
    std = np.asarray(std).flatten()
    observations = np.asarray(observations).flatten()
    if not (len(mean) == len(std) == len(observations)):
        raise ValueError(
            "Predictions, standard deviations, and observations must align."
        )

    observed = np.isfinite(observations)
    if not observed.any():
        raise ValueError("Validation contains no observed targets.")
    if not np.all(np.isfinite(mean[observed])):
        raise ValueError(
            "Validation prediction means must be finite at observed steps."
        )
    if not np.all(np.isfinite(std[observed])) or not np.all(std[observed] > 0):
        raise ValueError(
            "Validation predictive standard deviations must be finite and positive."
        )

    return {
        "mse": float(metric.mse(mean[observed], observations[observed])),
        "mean_log_likelihood": float(
            metric.log_likelihood(mean[observed], observations[observed], std[observed])
        ),
        "n_observed": int(observed.sum()),
    }


def run_scenarios(
    data_processor: DataProcess,
    train_data: dict[str, np.ndarray],
    validation_data: dict[str, np.ndarray],
    all_data: dict[str, np.ndarray],
    num_epochs: int = NUM_EPOCHS,
    on_result: Callable[[ScenarioResult], None] | None = None,
) -> list[ScenarioResult]:
    """Train the two reusable offline starts and execute all six scenarios."""

    offline_snapshots = {
        initialization: train_offline(
            initialization, train_data, data_processor, num_epochs=num_epochs
        )
        for initialization in ("local", "global")
    }
    validation_obs = validation_data["y"].flatten()
    results = []

    for scenario in SCENARIOS:
        print(f"Scenario {scenario.number}: {scenario.label}")
        if scenario.offline_pretraining:
            model = Model.load_dict(
                copy.deepcopy(offline_snapshots[scenario.initialization])
            )
        else:
            model = build_model(scenario.initialization)

        if scenario.validation_mode == "frozen":
            mean, std = run_frozen_validation(
                model, train_data=train_data, validation_data=validation_data
            )
        else:
            if scenario.online_start is None:
                raise RuntimeError("Online scenarios require an online start.")
            mean, std = run_online_validation(
                model,
                all_data=all_data,
                validation_start=data_processor.validation_start,
                validation_end=data_processor.validation_end,
                online_start=scenario.online_start,
            )

        scores = score_validation(mean, std, validation_obs)
        result = ScenarioResult(
            scenario=scenario,
            mean=mean,
            std=std,
            mse=float(scores["mse"]),
            mean_log_likelihood=float(scores["mean_log_likelihood"]),
            n_observed=int(scores["n_observed"]),
            offline_epochs=num_epochs if scenario.offline_pretraining else 0,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pgf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_scenario(
    result: ScenarioResult,
    validation_time: pd.DatetimeIndex,
    observations: np.ndarray,
) -> None:
    """Save one validation observation/prediction figure for a scenario."""

    with mpl.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=SINGLE_COL)
        ax.plot(
            validation_time,
            observations,
            color="tab:red",
            linewidth=0.7,
            label="observation",
            zorder=3,
        )
        ax.plot(
            validation_time,
            result.mean,
            color="tab:blue",
            linewidth=1.0,
            label="one-step prediction",
            zorder=2,
        )
        ax.set_xlabel("date")
        ax.set_ylabel(r"standardized $y$")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.margins(x=0)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            frameon=False,
        )
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()
        save_figure(fig, result.scenario.output_dir, "predictions")


def plot_residual_comparison(
    results: list[ScenarioResult],
    validation_time: pd.DatetimeIndex,
    observations: np.ndarray,
) -> None:
    """Overlap the validation residuals from all six scenarios."""

    with mpl.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=DOUBLE_COL)
        for result, (color, linestyle) in zip(results, RESIDUAL_STYLES):
            residual = observations - result.mean
            ax.plot(
                validation_time,
                residual,
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
                alpha=0.8,
                label=f"{result.scenario.number}: {result.scenario.label}",
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("date")
        ax.set_ylabel("residual")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.margins(x=0)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            ncol=2,
            frameon=False,
        )
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()
        save_figure(fig, OUT_ROOT / "summary", "residual_comparison")


def metrics_frame(results: list[ScenarioResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": result.scenario.number,
                "method": result.scenario.label,
                "mse": result.mse,
                "mean_log_likelihood": result.mean_log_likelihood,
                "n_observed": result.n_observed,
                "offline_epochs": result.offline_epochs,
            }
            for result in results
        ]
    )


def markdown_metrics_table(metrics: pd.DataFrame) -> str:
    lines = [
        "| Scenario | Method | MSE | Mean log-likelihood | Observed | Offline epochs |",
        "| ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.method} | {row.mse:.6f} | "
            f"{row.mean_log_likelihood:.6f} | {row.n_observed} | "
            f"{row.offline_epochs} |"
        )
    return "\n".join(lines) + "\n"


def write_scenario_result(
    result: ScenarioResult,
    validation_time: pd.DatetimeIndex,
    observations: np.ndarray,
) -> None:
    """Write one scenario immediately so long runs retain completed work."""

    result.scenario.output_dir.mkdir(parents=True, exist_ok=True)
    residual = observations - result.mean
    pd.DataFrame(
        {
            "date_time": validation_time,
            "observation": observations,
            "prediction": result.mean,
            "predictive_std": result.std,
            "residual": residual,
        }
    ).to_csv(result.scenario.output_dir / "predictions.csv", index=False)
    metrics_frame([result]).to_csv(
        result.scenario.output_dir / "metrics.csv", index=False
    )
    plot_scenario(result, validation_time, observations)


def write_summary(
    results: list[ScenarioResult],
    validation_time: pd.DatetimeIndex,
    observations: np.ndarray,
) -> pd.DataFrame:
    """Write the aggregate metric table and residual comparison."""

    summary_dir = OUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = metrics_frame(results)
    summary.to_csv(summary_dir / "metrics.csv", index=False)
    (summary_dir / "metrics.md").write_text(
        markdown_metrics_table(summary), encoding="utf-8"
    )
    plot_residual_comparison(results, validation_time, observations)
    return summary


def write_results(
    results: list[ScenarioResult],
    validation_time: pd.DatetimeIndex,
    observations: np.ndarray,
) -> pd.DataFrame:
    """Write all isolated scenario artifacts and aggregate outputs."""

    for result in results:
        write_scenario_result(result, validation_time, observations)
    return write_summary(results, validation_time, observations)


def main() -> None:
    _require_file(GLOBAL_MEANS_PATH, "global LSTM parameter file")
    data_processor = load_data()
    train_data, validation_data, test_data, all_data = data_processor.get_splits()
    if len(test_data["y"]):
        raise RuntimeError("The OOD experiment expects an empty test split.")

    validation_time = data_processor.data.index[
        data_processor.validation_start : data_processor.validation_end
    ]
    validation_obs = validation_data["y"].flatten()
    print(
        f"Series {SERIES}: {len(train_data['y'])} training steps and "
        f"{len(validation_data['y'])} validation steps"
    )
    print(
        "Validation starts at "
        f"{validation_time[0].date()}; metrics use {np.isfinite(validation_obs).sum()} "
        "observed targets"
    )

    results = run_scenarios(
        data_processor,
        train_data=train_data,
        validation_data=validation_data,
        all_data=all_data,
        on_result=lambda result: write_scenario_result(
            result, validation_time, validation_obs
        ),
    )
    summary = write_summary(results, validation_time, validation_obs)

    print("\nValidation metrics (standardized units)")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mse": "{:.6f}".format,
                "mean_log_likelihood": "{:.6f}".format,
            },
        )
    )
    print(f"\nSaved results to {OUT_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
