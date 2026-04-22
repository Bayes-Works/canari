"""LSTM encoder-decoder autoencoder for univariate anomaly detection.

Malhotra et al., 'LSTM-based Encoder-Decoder for Multi-sensor Anomaly
Detection' (ICML Anomaly Detection Workshop, 2016).

Training: sliding windows of length L from the clean training series are
reconstructed; loss is per-window MSE. Anomaly score at timestep t is the
reconstruction MSE of the window ending at t, normalised by
max(train_scores). Detection rule (matching zhan's convention):
    score(t) > c * max(train_scores)
with the coefficient c tuned on train+val.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def _make_windows(y: np.ndarray, L: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).flatten()
    if len(y) < L:
        raise ValueError(f"series length {len(y)} < window size {L}")
    idx = np.arange(L)[None, :] + np.arange(len(y) - L + 1)[:, None]
    return y[idx]


class _LSTMED(nn.Module):
    def __init__(self, hidden: int = 32, num_layers: int = 1):
        super().__init__()
        self.enc = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dec = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1)
        _, (h, c) = self.enc(x)
        dec_in = torch.zeros_like(x)
        dec_out, _ = self.dec(dec_in, (h, c))
        return self.out(dec_out)


def build_scorer(
    train_y,
    window_size: int = 30,
    hidden: int = 32,
    num_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> dict:
    """Fit the LSTM encoder-decoder once and return normalised reconstruction
    error scores such that the detection rule is ``raw > coefficient``.

    The raw score at each timestep is the window reconstruction MSE divided
    by the maximum MSE observed across training windows (zhan-style
    max-ratio thresholding). Positions before the first complete window are
    NaN.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu"

    train = pd.Series(np.asarray(train_y, dtype=np.float32).flatten())
    train = (
        train.interpolate("linear", limit_direction="both")
        .to_numpy()
        .astype(np.float32)
    )

    wins = _make_windows(train, window_size)
    X = torch.from_numpy(wins[:, :, None]).to(device)  # (N, L, 1)

    model = _LSTMED(hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    N = X.shape[0]
    model.train()
    for _ in range(num_epochs):
        perm = torch.randperm(N)
        for s in range(0, N, batch_size):
            b = X[perm[s : s + batch_size]]
            loss = ((model(b) - b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        train_err = ((model(X) - X) ** 2).mean(dim=(1, 2)).cpu().numpy()
    train_max = float(max(train_err.max(), 1e-12))

    def raw_scorer(eval_y: np.ndarray) -> np.ndarray:
        y = pd.Series(np.asarray(eval_y, dtype=np.float32).flatten())
        y = (
            y.interpolate("linear", limit_direction="both")
            .to_numpy()
            .astype(np.float32)
        )
        full = np.full(len(y), np.nan)
        if len(y) < window_size:
            return full
        wins = _make_windows(y, window_size)
        X_eval = torch.from_numpy(wins[:, :, None]).to(device)
        with torch.no_grad():
            err = ((model(X_eval) - X_eval) ** 2).mean(dim=(1, 2)).cpu().numpy()
        end_positions = np.arange(err.shape[0]) + (window_size - 1)
        mask = end_positions < len(y)
        full[end_positions[mask]] = (err[: int(mask.sum())]) / train_max
        return full

    return {
        "raw_scorer": raw_scorer,
        "train_score_max": train_max,
        "coefficient_grid": [0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0],
        "coefficient_name": "max_ratio_coefficient",
        "threshold_rule": "score > c * max(train_scores)",
        "window_size": int(window_size),
    }
