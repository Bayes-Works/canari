"""Baseline anomaly detectors.

Each module exposes a ``build_detector(dataset, options)`` function that returns
a tuple ``(detector_fn, info)`` where:

    detector_fn(eval_input: dict) -> np.ndarray[bool]
        Takes a data dict with ``"y"`` (and optionally ``"time"``) and returns a
        boolean flag per timestep. ``True`` means the detector fires at that
        step.
    info: dict
        Method-specific metadata (fitted threshold, window size, etc.) that is
        saved to the summary JSON.
"""

from . import damp, lstm_ed, prophet, tranad

METHODS = {
    "prophet": prophet.build_detector,
    "lstm_ed": lstm_ed.build_detector,
    "damp": damp.build_detector,
    "tranad": tranad.build_detector,
}

__all__ = ["METHODS"]
