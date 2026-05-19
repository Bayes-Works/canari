"""Prophet online changepoint detector (zhan-exact, with optional calibration).

Port of ``experiments/zhan/prophet.py``. Walks forward one step at a time from
``int(n * online_begin_ratio)``, refitting Prophet on the prefix at each step,
and breaks at the first step where any learned changepoint has
``abs(mean(delta)) >= threshold``. Returns a boolean array with a single True
at the detection step (or all False).

If ``dataset["calibration"]`` is provided by the orchestrator, the threshold
is instead chosen as the smallest value whose flagger produces
``FA_rate_per_y <= fa_target_per_year`` on the clean train/val region. This
costs one additional walk-forward scorer pass at build time.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .calibrate import calibrate_threshold


def build_detector(dataset: dict, options: dict) -> tuple[Callable, dict]:
    logging.getLogger("prophet").setLevel(logging.ERROR)
    cmdstan_logger = logging.getLogger("cmdstanpy")
    cmdstan_logger.setLevel(logging.CRITICAL)
    cmdstan_logger.propagate = False
    from cmdstanpy import disable_logging
    from prophet import Prophet

    begin_ratio = float(options.get("online_begin_ratio", 0.4))
    changepoint_range = float(options.get("changepoint_range", 1.0))
    default_threshold = float(options.get("changepoint_delta_threshold", 0.3))

    def scorer(eval_input: dict) -> np.ndarray:
        y = np.asarray(eval_input["y"], dtype=float).flatten()
        times = pd.to_datetime(np.asarray(eval_input["time"]))
        n = len(y)
        scores = np.full(n, np.nan)
        begin_idx = int(n * begin_ratio)
        for current_idx in range(begin_idx, n):
            prefix = pd.DataFrame(
                {"ds": times[:current_idx], "y": y[:current_idx]}
            )
            if len(prefix) < 2:
                continue
            model = Prophet(changepoint_range=changepoint_range)
            with disable_logging():
                model.fit(prefix)
            if len(model.changepoints) == 0:
                scores[current_idx] = 0.0
                continue
            delta = np.abs(np.nanmean(model.params["delta"], axis=0))
            scores[current_idx] = float(np.max(delta)) if delta.size else 0.0
        return scores

    def flagger(scores: np.ndarray, threshold: float) -> np.ndarray:
        """zhan-exact: fire on the first step where score >= threshold."""
        flags = np.zeros(len(scores), dtype=bool)
        crossings = np.isfinite(scores) & (scores >= threshold)
        if crossings.any():
            flags[int(np.argmax(crossings))] = True
        return flags

    calibration_cfg = dataset.get("calibration")
    calibration_result = None
    if calibration_cfg:
        calibration_result = calibrate_threshold(
            scorer=scorer, flagger=flagger, **calibration_cfg
        )
        threshold = float(calibration_result["threshold"])
    else:
        threshold = default_threshold

    def detector(eval_input: dict) -> np.ndarray:
        return flagger(scorer(eval_input), threshold)

    info = {
        "online_begin_ratio": begin_ratio,
        "changepoint_range": changepoint_range,
        "changepoint_delta_threshold": threshold,
        "threshold_rule": (
            "break at first step where abs(mean(delta)) >= threshold (zhan-exact)"
            if calibration_result is None
            else "break at first step where abs(mean(delta)) >= calibrated_threshold"
        ),
    }
    if calibration_result is not None:
        info["calibration"] = calibration_result
    return detector, info
