"""Threshold calibration on clean train/val (no synthetic anomaly injection).

Pick the **smallest** threshold such that the method's own flagger produces
``FA_rate_per_y <= fa_target_per_year`` on the train/val region of the series.
Candidates are built from the observed train/val score distribution itself
plus a small cap above the max, so the feasible set always contains the
FA=0 option.

Why this instead of synthetic anomalies? All four flaggers are monotonic in
threshold (higher threshold -> fewer flags), so sensitivity is maximized by
the smallest threshold that clears the FA budget. Finding it needs only one
scorer pass on the clean series -- no anomaly realizations, no per-method
grids.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def calibrate_threshold(
    *,
    scorer: Callable[[dict], np.ndarray],
    flagger: Callable[[np.ndarray, float], np.ndarray],
    all_data: dict,
    train_val_end_idx: int,
    steps_per_year: float,
    fa_target_per_year: float = 0.0,
) -> dict:
    total_len = len(np.asarray(all_data["y"]))
    if train_val_end_idx <= 0 or train_val_end_idx > total_len:
        raise ValueError(
            f"train_val_end_idx={train_val_end_idx} out of range for series of length {total_len}"
        )

    scores_full = scorer(all_data)
    scores_tv = scores_full[:train_val_end_idx]
    finite = scores_tv[np.isfinite(scores_tv)]
    if finite.size == 0:
        finite_full_idx = np.flatnonzero(np.isfinite(scores_full))
        if finite_full_idx.size:
            detail = (
                f"first finite score is at index {int(finite_full_idx[0])}, "
                f"after train_val_end_idx={train_val_end_idx}"
            )
        else:
            detail = "scorer produced no finite scores anywhere"
        raise ValueError(
            "No finite scores in train/val region during calibration "
            f"(train_val_end_idx={train_val_end_idx}, total_len={total_len}; {detail})."
        )

    uniq = np.unique(finite)
    # Cap slightly above max so a strict-`>` flagger can achieve zero FAs.
    cap = float(uniq[-1]) + max(abs(uniq[-1]) * 1e-6, 1e-9)
    candidates = np.concatenate([uniq, [cap]])  # ascending

    years = train_val_end_idx / steps_per_year if steps_per_year > 0 else float("nan")
    history: list[dict] = []
    for thr in candidates:
        flags = flagger(scores_full, float(thr))
        fa_count = int(np.sum(flags[:train_val_end_idx]))
        fa_rate = fa_count / years if years > 0 else float("nan")
        history.append({
            "threshold": float(thr),
            "fa_count": fa_count,
            "fa_rate_per_y": float(fa_rate),
        })

    # history is sorted ascending by threshold (monotonic flagger -> FA only drops).
    feasible = [h for h in history if h["fa_rate_per_y"] <= fa_target_per_year]
    if feasible:
        best = feasible[0]
        selection_mode = "smallest_thr_under_fa_target"
    else:
        best = min(history, key=lambda h: h["fa_rate_per_y"])
        selection_mode = "min_fa_target_infeasible"

    return {
        "threshold": best["threshold"],
        "fa_count": best["fa_count"],
        "fa_rate_per_y": best["fa_rate_per_y"],
        "selection_mode": selection_mode,
        "fa_target_per_year": float(fa_target_per_year),
        "years_train_val": float(years),
        "num_candidates": int(len(candidates)),
        "max_clean_score": float(uniq[-1]),
    }
