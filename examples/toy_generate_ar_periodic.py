"""Generate a stationary toy time series with periodic, autoregressive, and noise parts.

The series is hourly and contains:

- two harmonics (24 h and 12 h) giving a non-sinusoidal daily pattern,
- a short-memory AR(1) process (phi = 0.6) adding correlated fluctuations,
- white observation noise.

The series is centered and detrended, so it contains no level or trend to learn.

Run from the repository root:

    python examples/toy_generate_ar_periodic.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUM_STEPS = 480
PHI = 0.6
STD_AR = 0.16
STD_NOISE = 0.1
SEED = 1


def generate():
    rng = np.random.default_rng(SEED)
    time = np.arange(NUM_STEPS)

    periodic = np.sin(2 * np.pi * time / 24) + 0.5 * np.sin(
        2 * np.pi * time / 12 + np.pi / 4
    )

    # AR(1) started from its stationary distribution so the series is stationary
    # from the first time step.
    autoregression = np.zeros(NUM_STEPS)
    autoregression[0] = rng.normal(0, STD_AR / np.sqrt(1 - PHI**2))
    for t in range(1, NUM_STEPS):
        autoregression[t] = PHI * autoregression[t - 1] + rng.normal(0, STD_AR)

    noise = rng.normal(0, STD_NOISE, NUM_STEPS)
    values = periodic + autoregression + noise

    # Remove the mean and the residual slope of this realization: the series is
    # centered around zero and has no trend.
    return values - np.polyval(np.polyfit(time, values, 1), time)


def main():
    values = generate()
    datetimes = pd.date_range("2000-01-01", periods=NUM_STEPS, freq="h")

    output_dir = ROOT / "data/toy_time_series"
    pd.DataFrame({"ar_periodic": values.round(4)}).to_csv(
        output_dir / "ar_periodic.csv", index=False
    )
    pd.DataFrame({"date_time": datetimes}).to_csv(
        output_dir / "ar_periodic_datetime.csv", index=False
    )
    print(f"Saved {NUM_STEPS} time steps to {output_dir}")
    print(f"mean: {values.mean(): 0.3f}, std: {values.std(): 0.3f}")


if __name__ == "__main__":
    main()
