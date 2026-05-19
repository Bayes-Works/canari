"""TranAD anomaly detector using the official imperial-qore/TranAD model.

The TranAD model class and custom transformer layers come directly from the
cloned repo at ``experiments/baselines/tranad_src/`` (git-ignored). Training
and scoring follow ``main.py:backprop`` (the ``'TranAD'`` branch) from that
repo. Threshold calibration is zhan-exact (``experiments/zhan/tranad.py``):
flag when ``score > 1.1 * max(train_score) OR score < 0.9 * min(train_score)``.

Note on input scaling: TranAD's output layer is ``Linear + Sigmoid``, so it
expects inputs in ``[0, 1]``. We fit a min/max scaler on train and apply it
to all future inputs (the authors do the same in ``preprocess.py``).
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .calibrate import calibrate_threshold

_THIS_DIR = Path(__file__).resolve().parent
_TRANAD_ROOT = _THIS_DIR / "tranad_src"


def _load_tranad_class():
    """Import ``TranAD`` from the vendored repo without triggering its CLI/dgl deps.

    The repo's ``src/models.py`` transitively imports ``src.parser`` (runs
    argparse at import time), ``dgl`` (heavy, unused by TranAD itself), and
    dataset-specific constants. We stub those in ``sys.modules`` before the
    import so the real ``models.py`` runs unchanged and we get their class.
    """
    if "src.models" in sys.modules and hasattr(sys.modules["src.models"], "TranAD"):
        return sys.modules["src.models"].TranAD

    if not (_TRANAD_ROOT / "src" / "models.py").exists():
        raise RuntimeError(
            f"TranAD source not found at {_TRANAD_ROOT}. Clone it with:\n"
            f"    git clone https://github.com/imperial-qore/TranAD.git {_TRANAD_ROOT}"
        )

    # Stub dgl + dgl.nn — models.py imports them at module top but TranAD itself
    # never touches them (only GDN/MTAD_GAT do).
    if "dgl" not in sys.modules:
        dgl_stub = types.ModuleType("dgl")
        dgl_nn = types.ModuleType("dgl.nn")
        dgl_nn.GATConv = object
        dgl_stub.nn = dgl_nn
        sys.modules["dgl"] = dgl_stub
        sys.modules["dgl.nn"] = dgl_nn

    # Register 'src' as a package whose __path__ points at the vendored src/.
    # Subsequent `import src.X` calls will resolve to files in that directory.
    src_pkg = types.ModuleType("src")
    src_pkg.__path__ = [str(_TRANAD_ROOT / "src")]
    sys.modules["src"] = src_pkg

    # Stub parser with default args (synthetic/TranAD) so constants.py can index
    # its dataset-specific tables without calling argparse.parse_args().
    parser_mod = types.ModuleType("src.parser")
    parser_mod.args = argparse.Namespace(
        dataset="synthetic",
        model="TranAD",
        test=False,
        retrain=False,
        less=False,
    )
    sys.modules["src.parser"] = parser_mod

    # Real constants.py + dlutils.py + models.py load on first `import`.
    from src import dlutils as tranad_dlutils  # noqa: E402
    from src import models as tranad_models  # noqa: E402

    # Newer torch's built-in TransformerEncoder.forward passes `is_causal` /
    # `src_key_padding_mask` down to each layer's forward(). TranAD's custom
    # layer predates that signature — wrap to accept and ignore extra kwargs.
    for cls in (tranad_dlutils.TransformerEncoderLayer, tranad_dlutils.TransformerDecoderLayer):
        orig_forward = cls.forward

        def _forward_compat(self, *args, _orig=orig_forward, **kwargs):
            kwargs.pop("is_causal", None)
            kwargs.pop("memory_is_causal", None)
            kwargs.pop("tgt_is_causal", None)
            return _orig(self, *args, **kwargs)

        cls.forward = _forward_compat

    return tranad_models.TranAD


def _sanitize(y: np.ndarray) -> np.ndarray:
    y = pd.Series(np.asarray(y, dtype=float).flatten())
    return y.interpolate("linear", limit_direction="both").to_numpy()


def _convert_to_windows(data: torch.Tensor, w_size: int) -> torch.Tensor:
    """Port of ``main.py:convert_to_windows`` (TranAD branch) — one window per step."""
    windows = []
    for i in range(data.shape[0]):
        if i >= w_size:
            w = data[i - w_size : i]
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)


def _train_epoch(epoch: int, model, data: torch.Tensor, optimizer, feats: int) -> float:
    """TranAD training branch of ``main.py:backprop``."""
    loss_fn = nn.MSELoss(reduction="mean")
    loader = DataLoader(TensorDataset(data, data), batch_size=model.batch, shuffle=False)
    n = epoch + 1
    losses = []
    for d, _ in loader:
        local_bs = d.shape[0]
        window = d.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, local_bs, feats)
        z = model(window, elem)
        if isinstance(z, tuple):
            l = (1.0 / n) * loss_fn(z[0], elem) + (1 - 1.0 / n) * loss_fn(z[1], elem)
        else:
            l = loss_fn(z, elem)
        optimizer.zero_grad()
        l.backward(retain_graph=True)
        optimizer.step()
        losses.append(l.item())
    return float(np.mean(losses)) if losses else float("nan")


def _score_windows(model, data: torch.Tensor, feats: int) -> np.ndarray:
    """TranAD inference branch: per-window reconstruction MSE (shape ``(T, feats)``)."""
    loss_fn = nn.MSELoss(reduction="none")
    bs = data.shape[0]
    loader = DataLoader(TensorDataset(data, data), batch_size=bs)
    with torch.no_grad():
        for d, _ in loader:
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, bs, feats)
            z = model(window, elem)
            if isinstance(z, tuple):
                z = z[1]
        loss = loss_fn(z, elem)[0]
    return loss.detach().cpu().numpy()


def build_detector(dataset: dict, options: dict) -> tuple[Callable, dict]:
    seed = int(options.get("seed", 0))
    num_epochs = int(options.get("num_epochs", 5))
    learning_rate = float(options.get("learning_rate", 1e-3))
    window_size = options.get("window_size")  # None → keep the model's default (10)
    threshold_multiplier = float(options.get("threshold_multiplier", 1.1))
    lower_threshold_multiplier = float(options.get("lower_threshold_multiplier", 0.9))
    min_score_threshold = float(options.get("min_score_threshold", 0.0))

    np.random.seed(seed)
    torch.manual_seed(seed)

    TranAD = _load_tranad_class()

    # Fit min/max on the clean train split; TranAD's output layer is sigmoid.
    train_y_raw = _sanitize(np.asarray(dataset["train_data"]["y"]).flatten())
    y_min = float(np.nanmin(train_y_raw))
    y_max = float(np.nanmax(train_y_raw))
    span = y_max - y_min
    if not np.isfinite(span) or span <= 0:
        span = 1.0

    def _scale(y: np.ndarray) -> np.ndarray:
        y = _sanitize(np.asarray(y).flatten())
        return np.clip((y - y_min) / span, 0.0, 1.0)

    feats = 1
    model = TranAD(feats).double()
    if window_size is not None:
        model.n_window = int(window_size)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    train_scaled = _scale(train_y_raw).reshape(-1, feats)
    train_tensor = torch.from_numpy(train_scaled).double()
    train_windows = _convert_to_windows(train_tensor, model.n_window)

    model.train()
    for epoch in range(num_epochs):
        _train_epoch(epoch, model, train_windows, optimizer, feats)

    model.eval()
    train_loss = _score_windows(model, train_windows, feats)
    train_scores = train_loss[:, 0]
    finite = np.isfinite(train_scores)
    if not finite.any():
        raise ValueError("TranAD produced no finite scores on training data.")
    train_score_max = float(np.max(train_scores[finite]))
    train_score_min = float(np.min(train_scores[finite]))
    lower_threshold = float(lower_threshold_multiplier * train_score_min)

    def scorer(eval_input: dict) -> np.ndarray:
        y = np.asarray(eval_input["y"]).flatten()
        n = len(y)
        if n == 0:
            return np.zeros(0, dtype=float)
        data = torch.from_numpy(_scale(y).reshape(-1, feats)).double()
        windows = _convert_to_windows(data, model.n_window)
        loss = _score_windows(model, windows, feats)
        return loss[:, 0]

    # Upper threshold is the calibrated knob; lower side stays fixed at 0.9 * min_train.
    def flagger(scores: np.ndarray, upper: float) -> np.ndarray:
        flags = np.zeros(len(scores), dtype=bool)
        finite = np.isfinite(scores)
        flags[finite] = (scores[finite] > upper) | (scores[finite] < lower_threshold)
        return flags

    calibration_cfg = dataset.get("calibration")
    calibration_result = None
    if calibration_cfg:
        calibration_result = calibrate_threshold(
            scorer=scorer, flagger=flagger, **calibration_cfg
        )
        upper_threshold = float(calibration_result["threshold"])
    else:
        upper_threshold = float(
            max(threshold_multiplier * train_score_max, min_score_threshold)
        )

    def detector(eval_input: dict) -> np.ndarray:
        return flagger(scorer(eval_input), upper_threshold)

    info = {
        "window_size": int(model.n_window),
        "batch_size": int(model.batch),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "threshold_multiplier": threshold_multiplier,
        "lower_threshold_multiplier": lower_threshold_multiplier,
        "min_score_threshold": min_score_threshold,
        "y_min": y_min,
        "y_max": y_max,
        "train_score_max": train_score_max,
        "train_score_min": train_score_min,
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
        "threshold_rule": (
            "score > max(threshold_multiplier * max_train, min_score_threshold) "
            "OR score < lower_threshold_multiplier * min_train"
            if calibration_result is None
            else "score > smallest upper with FA_rate <= target on clean train/val "
            "OR score < 0.9 * min_train"
        ),
    }
    if calibration_result is not None:
        info["calibration"] = calibration_result
    return detector, info
