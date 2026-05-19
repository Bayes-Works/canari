"""Discord Aware Matrix Profile (DAMP) detector.

Reference: ``experiments/zhan/matrix_profile.py``. Same usage pattern (compute
the left matrix profile, calibrate a fixed threshold at ``1.1 * max`` of the
score on a clean reference region, flag where score > threshold) but using
DAMP instead of the exact matrix profile.

DAMP produces an approximate left matrix profile: for each subsequence after
``sp_index`` it searches backward in powers-of-two segments and stops as soon
as it finds a neighbor closer than the best-so-far discord distance. Entries
that cannot be top discords are pruned forward.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import stumpy

from .calibrate import calibrate_threshold


def damp(
    T: np.ndarray,
    m: int,
    sp_index: int,
    lookahead: int | None = None,
) -> np.ndarray:
    """Return the DAMP approximate left matrix profile of length ``len(T) - m + 1``.

    Entries before ``sp_index`` are left at 0 (not scored). Entries >= sp_index
    hold either the approximate 1-NN distance or, if forward-pruned, an upper
    bound that is already known to be below the best-so-far discord score.
    """

    T = np.asarray(T, dtype=float).flatten()
    n = T.shape[0]
    if m < 4 or m >= n:
        raise ValueError(f"invalid window m={m} for series length {n}")
    if sp_index < m:
        raise ValueError(f"sp_index={sp_index} must be >= m={m}")

    num_sub = n - m + 1
    left_mp = np.zeros(num_sub, dtype=float)
    bsf = 0.0
    init_chunk = int(2 ** np.ceil(np.log2(16 * m)))
    if lookahead is None:
        lookahead = init_chunk

    for i in range(sp_index, num_sub):
        if left_mp[i] != 0.0 and left_mp[i] < bsf:
            continue

        query = T[i : i + m]
        X = init_chunk
        while True:
            if i - X < 0:
                dp = stumpy.core.mass(query, T[0 : i + m - 1])
                left_mp[i] = float(np.nanmin(dp))
                break
            dp = stumpy.core.mass(query, T[i - X : i + m - 1])
            approx_dist = float(np.nanmin(dp))
            if approx_dist < bsf:
                left_mp[i] = approx_dist
                break
            X *= 2

        if left_mp[i] > bsf:
            bsf = left_mp[i]

        if lookahead > 0 and i + m < n:
            seg_end = min(i + 1 + lookahead + m - 1, n)
            segment = T[i + 1 : seg_end]
            if segment.shape[0] >= m:
                dp = stumpy.core.mass(query, segment)
                targets = np.arange(i + 1, i + 1 + dp.shape[0])
                valid = targets < num_sub
                targets = targets[valid]
                dp = dp[valid]
                prunable = dp < bsf
                for t, d in zip(targets[prunable], dp[prunable]):
                    if left_mp[t] == 0.0 or d < left_mp[t]:
                        left_mp[t] = float(d)

    return left_mp


def _sanitize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).flatten()
    if np.isnan(y).any():
        y = pd.Series(y).interpolate("linear", limit_direction="both").to_numpy()
    return y


def build_detector(dataset: dict, options: dict) -> tuple[Callable, dict]:
    window_size = int(options.get("window_size", 52))
    lookahead = options.get("lookahead")
    threshold_multiplier = float(options.get("threshold_multiplier", 1.1))

    train_y = _sanitize(dataset["train_data"]["y"])
    train_val_y = _sanitize(dataset["train_val"]["y"])
    sp_index = train_y.shape[0]

    calibration_scores = damp(train_val_y, m=window_size, sp_index=sp_index, lookahead=lookahead)
    calibration_scores[:sp_index] = np.nan
    valid = calibration_scores[np.isfinite(calibration_scores)]
    if valid.size == 0:
        raise ValueError("DAMP calibration produced no finite scores on train_val.")
    calibration_score_max = float(np.max(valid))

    def scorer(eval_input: dict) -> np.ndarray:
        y = _sanitize(eval_input["y"])
        n = y.shape[0]
        scores = np.full(n, np.nan)
        if n < window_size:
            return scores
        left_mp = damp(y, m=window_size, sp_index=sp_index, lookahead=lookahead)
        scores[: left_mp.shape[0]] = left_mp
        scores[:sp_index] = np.nan
        return scores

    def flagger(scores: np.ndarray, threshold: float) -> np.ndarray:
        flags = np.zeros(len(scores), dtype=bool)
        finite = np.isfinite(scores)
        flags[finite] = scores[finite] > threshold
        return flags

    calibration_cfg = dataset.get("calibration")
    calibration_result = None
    if calibration_cfg:
        calibration_result = calibrate_threshold(
            scorer=scorer, flagger=flagger, **calibration_cfg
        )
        score_threshold = float(calibration_result["threshold"])
    else:
        score_threshold = float(threshold_multiplier * calibration_score_max)

    def detector(eval_input: dict) -> np.ndarray:
        return flagger(scorer(eval_input), score_threshold)

    info = {
        "window_size": window_size,
        "lookahead": lookahead,
        "threshold_multiplier": threshold_multiplier,
        "sp_index": int(sp_index),
        "calibration_score_max": calibration_score_max,
        "score_threshold": score_threshold,
        "threshold_rule": (
            "damp_score > 1.1 * max(damp_score on validation region)"
            if calibration_result is None
            else "damp_score > smallest threshold with FA_rate <= target on clean train/val"
        ),
    }
    if calibration_result is not None:
        info["calibration"] = calibration_result
    return detector, info
