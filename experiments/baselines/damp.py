"""Discord Aware Matrix Profile (DAMP) baseline.

Reference:
    Lu et al., "Matrix Profile XXIV: Scaling Time Series Anomaly Detection to
    Trillions of Datapoints and Ultra-fast Arriving Data Streams", KDD 2022.

DAMP computes an approximate *left* matrix profile: for each subsequence
starting at index i >= sp_index, the distance to its nearest neighbor in the
preceding data T[0:i]. It combines a backward doubling search (with best-so-far
early abandoning) and a forward pruning step that lets it skip subsequences
that cannot be top discords.

Here it is wrapped as a scorer: the train series fixes a threshold via its
self matrix profile (mean + k*std), and the eval series is scored by running
DAMP on [train || eval] with sp_index = len(train).
"""

import numpy as np
import pandas as pd
import stumpy


def damp(
    T: np.ndarray,
    m: int,
    sp_index: int,
    lookahead: int | None = None,
) -> np.ndarray:
    """Run DAMP and return the approximate left matrix profile.

    Entries before sp_index are left at 0 (not computed). Entries at or after
    sp_index hold either the approximate 1-NN distance or, if forward-pruned,
    an upper bound on that distance that is known to be below BSF at pruning
    time (so the subsequence cannot be a top discord).

    Args:
        T: 1-D time series.
        m: subsequence length.
        sp_index: index where test processing begins. T[0:sp_index] is
            treated as the reference region searched by MASS.
        lookahead: forward pruning window (in samples). Defaults to
            2**ceil(log2(16*m)).

    Returns:
        Array of length len(T) - m + 1 with discord scores.
    """
    T = np.asarray(T, dtype=float).flatten()
    n = T.shape[0]
    if m < 4 or m >= n:
        raise ValueError(f"invalid window m={m} for series length {n}")
    if sp_index < m:
        raise ValueError(
            f"sp_index={sp_index} must be >= m={m} so a reference region exists"
        )

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
        approx_dist = np.inf

        while True:
            if i - X < 0:
                prefix = T[0 : i + m - 1]
                dp = stumpy.core.mass(query, prefix)
                approx_dist = float(np.nanmin(dp))
                left_mp[i] = approx_dist
                break
            segment = T[i - X : i + m - 1]
            dp = stumpy.core.mass(query, segment)
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


def build_scorer(
    train_y: np.ndarray,
    window_size: int,
    val_y: np.ndarray | None = None,
    lookahead: int | None = None,
) -> dict:
    """Fit DAMP once and return normalised per-step scores such that the
    detection rule is ``raw > coefficient``.

    Threshold rule follows the matrix-profile convention used by
    zhan/matrix_profile.py: ``score > c * max(calibration_mp)``. The
    calibration MP is produced by running DAMP itself on ``[train || val]``
    with ``sp_index = len(train)`` (so val subsequences are scored against
    train + earlier val, the same way eval subsequences will be scored at
    test time). If ``val_y`` is not supplied, we fall back to the maximum
    of the training self matrix profile via ``stumpy.stump``.

    The raw score at each eval timestep is the DAMP left matrix profile
    value divided by that calibration maximum; positions before the first
    complete window are NaN.
    """
    train = np.asarray(train_y, dtype=float).flatten()
    if np.isnan(train).any():
        train = (
            pd.Series(train)
            .interpolate("linear", limit_direction="both")
            .to_numpy()
        )

    if val_y is not None:
        val = np.asarray(val_y, dtype=float).flatten()
        if np.isnan(val).any():
            val = (
                pd.Series(val)
                .interpolate("linear", limit_direction="both")
                .to_numpy()
            )
        combined_calib = np.concatenate([train, val])
        calib_mp = damp(
            combined_calib,
            m=window_size,
            sp_index=train.shape[0],
            lookahead=lookahead,
        )
        val_scores = calib_mp[train.shape[0] :]
        val_scores = val_scores[np.isfinite(val_scores)]
        if val_scores.size > 0:
            train_max = float(np.max(val_scores))
        else:
            train_max = 0.0
    else:
        mp_train = stumpy.stump(train, m=window_size)
        train_dist = mp_train[:, 0].astype(float)
        train_max = float(np.nanmax(train_dist))
    train_max = max(train_max, 1e-12)

    def raw_scorer(eval_y: np.ndarray) -> np.ndarray:
        y = np.asarray(eval_y, dtype=float).flatten()
        if np.isnan(y).any():
            y = (
                pd.Series(y)
                .interpolate("linear", limit_direction="both")
                .to_numpy()
            )
        full = np.full(y.shape[0], np.nan)
        if y.shape[0] < window_size:
            return full
        combined = np.concatenate([train, y])
        left_mp = damp(
            combined,
            m=window_size,
            sp_index=train.shape[0],
            lookahead=lookahead,
        )
        eval_scores = left_mp[train.shape[0] :]
        end_positions = np.arange(eval_scores.shape[0]) + (window_size - 1)
        mask = end_positions < y.shape[0]
        full[end_positions[mask]] = (
            eval_scores[: int(mask.sum())] / train_max
        )
        return full

    return {
        "raw_scorer": raw_scorer,
        "train_score_max": train_max,
        "coefficient_grid": [0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0],
        "coefficient_name": "max_ratio_coefficient",
        "threshold_rule": "score > c * max(calibration_mp)",
        "window_size": int(window_size),
    }
