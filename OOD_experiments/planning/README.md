# OOD LSTM experiment plan

## Goal and invariant

This experiment compares six ways to initialize and update the LSTM used by
`examples/offline_vs_online_lstm.py`. The first two weekly years are the only offline
training data. Every remaining timestamp is validation, and all metrics are calculated
only on that validation span.

The dataset is weekly, so two modeling years are defined as exactly `2 * 52 = 104`
timestamps. With the current series this gives 104 training steps and 1,444 validation
steps, beginning on 1995-08-20. Of those validation steps, 1,256 currently contain an
observation. `DataProcess` learns its standardization constants from the 104 training
steps only.

Validation must always be filtered. `Model.filter` and `Model.online_lstm_filter` emit a
one-step-ahead prediction before assimilating that step's observation. Do not replace a
validation pass with `Model.forecast`.

## Inputs and reference configuration

The script uses these local, gitignored inputs:

- `data/exp01_data/ts_weekly_values.csv`
- `data/exp01_data/ts_weekly_datetimes.csv`
- `saved_params/global_BM_52_256.bin`

The selected series is `MAT001PIAP-F510_x_cleaned`. The model matches the reference:
52-step look-back, week-of-year covariate, one 256-unit LSTM layer, manual seed 1, and
white-noise standard deviation 0.1.

"Global" initialization calls `Model.load_lstm_parameter_means`. It transfers only the
pretrained weight and bias means and deliberately keeps the new local parameter
variances.

The checkpoint does not encode its training provenance in this repository. Before these
results are described as leakage-free OOD evidence, confirm and record how
`global_BM_52_256.bin` was produced: source series, time ranges, normalization, and
whether `MAT001PIAP-F510_x_cleaned` from 1995-08-20 through 2023-04-16 was excluded.
Until that is confirmed, scenarios 2, 4, and 6 are valid executions of the requested
transfer protocol but their OOD interpretation has an unresolved provenance caveat.

## Scenario matrix

| # | Initial parameter means | Offline training on steps 0:104 | Validation behavior |
| ---: | :--- | :---: | :--- |
| 1 | Local seeded initialization | 100 fixed epochs | Frozen-parameter filter (`train_lstm=False`) |
| 2 | Transferred global means | 100 fixed epochs | Frozen-parameter filter (`train_lstm=False`) |
| 3 | Local seeded initialization | 100 fixed epochs | 52-step fixed-lag online filter |
| 4 | Transferred global means | 100 fixed epochs | 52-step fixed-lag online filter |
| 5 | Local seeded initialization | None | 52-step fixed-lag online filter from the beginning |
| 6 | Transferred global means | None | 52-step fixed-lag online filter from the beginning |

Offline training uses a fixed epoch count rather than validation-based early stopping.
The existing reference helper forecasts its validation argument for model selection;
using the OOD validation there would leak held-out information. An empty validation
dictionary is therefore passed into `Model.lstm_train`, preserving its training and
smoothing behavior without exposing any validation values or covariates.
The 100-epoch value is inherited from the reference experiment and is not selected by a
training-only convergence rule; it is recorded in each scenario's metrics file.

For scenarios 3 and 4, the library's fixed-lag online routine starts its first window 52
steps before the first scored validation timestamp. It re-filters that trailing training
context with parameter updates before producing the first validation prediction. This is
an intentional property of the project's online LSTM algorithm, not extra offline
pretraining. Scenarios 5 and 6 use the same algorithm from its first complete 52-step
window and continue updating through the full series.

## Metrics and artifacts

MSE and mean Gaussian log-likelihood are calculated on the common set of finite
validation observations, in training-standardized units. Higher mean log-likelihood is
better. Residuals are `observation - prediction`.

Each scenario directory under `out/` receives:

- `predictions.csv`: timestamps, observations, predictive means/standard deviations,
  and residuals;
- `metrics.csv`: the scenario's validation metrics;
- `predictions.pgf` and `predictions.pdf`: validation observations and predictions.

Each scenario is written as soon as it finishes, so an interrupted long run retains its
completed scenario artifacts. Aggregate outputs are written only after all six finish.

`out/summary/` receives `metrics.csv`, `metrics.md`, and the overlaid validation residual
plot as both `residual_comparison.pgf` and `residual_comparison.pdf`.

## Agent workflow

When changing this experiment:

1. Keep the data/model constants synchronized with `examples/offline_vs_online_lstm.py`.
2. Preserve the 104/rest split and never use OOD validation for offline model selection.
3. Preserve one prediction per validation timestamp for every scenario.
4. Score all scenarios on the identical finite-observation mask.
5. Keep validation paths filter-only; search for accidental `forecast` calls before
   accepting changes.
6. Run `pytest test/test_ood_experiment.py`, then perform a real-model smoke check before
   launching the full 100-epoch experiment.
