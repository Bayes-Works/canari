"""TranAD (minimal reimplementation) for univariate anomaly detection.

Tuli, Casale, Jennings, 'TranAD: Deep Transformer Networks for Anomaly
Detection in Multivariate Time Series Data', VLDB 2022.

Faithful in spirit to the paper (transformer encoder + two transformer
decoders, with a phase-2 focus signal given by the phase-1 squared error)
but reimplemented from scratch for univariate CPU use — shapes, positional
encoding, and the output projection are not byte-identical to the authors'
reference code at github.com/imperial-qore/TranAD.

Training loss (per mini-batch):
    L = (1/n) * MSE(x1, src) + (1 - 1/n) * MSE(x2, src)
where n = epoch + 1, mirroring the annealing weight used by the authors.
Anomaly score at timestep t is the MSE of x2 vs src for the window ending
at t, normalised by max(train_scores). Detection rule (matching zhan's
one-sided convention):
    score(t) > c * max(train_scores)
with the coefficient c tuned on train+val.
"""

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)


def _make_windows(y: np.ndarray, L: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).flatten()
    if len(y) < L:
        raise ValueError(f"series length {len(y)} < window size {L}")
    idx = np.arange(L)[None, :] + np.arange(len(y) - L + 1)[:, None]
    return y[idx]


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq, batch, d_model)
        return self.dropout(x + self.pe[: x.size(0)])


class _TranAD(nn.Module):
    def __init__(
        self,
        feats: int = 1,
        window: int = 10,
        d_model: int = 32,
        nhead: int = 2,
        dim_ff: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feats = feats
        self.window = window

        self.embed_src = nn.Linear(feats * 2, d_model)  # (signal, focus)
        self.embed_tgt = nn.Linear(feats, d_model)
        self.pos = _PositionalEncoding(d_model, dropout=dropout, max_len=window + 1)

        enc = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout
        )
        self.encoder = TransformerEncoder(enc, num_layers=1)
        dec1 = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout
        )
        self.decoder1 = TransformerDecoder(dec1, num_layers=1)
        dec2 = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout
        )
        self.decoder2 = TransformerDecoder(dec2, num_layers=1)
        self.fc_out = nn.Linear(d_model, feats)

    def _encode(self, src: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
        x = torch.cat((src, focus), dim=-1)
        x = self.embed_src(x)
        x = self.pos(x)
        return self.encoder(x)

    def _decode(
        self,
        decoder: nn.Module,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        t = self.embed_tgt(tgt)
        t = self.pos(t)
        out = decoder(t, memory)
        return self.fc_out(out)

    def forward(self, src: torch.Tensor):
        focus = torch.zeros_like(src)
        memory = self._encode(src, focus)
        x1 = self._decode(self.decoder1, src, memory)

        focus = (x1 - src) ** 2
        memory = self._encode(src, focus)
        x2 = self._decode(self.decoder2, src, memory)
        return x1, x2


def build_scorer(
    train_y,
    window_size: int = 10,
    num_epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> dict:
    """Fit TranAD once and return normalised reconstruction-error scores
    such that the detection rule is ``raw > coefficient``.

    The raw score at each timestep is the x2-vs-src window MSE divided by
    the maximum such MSE over training windows (zhan-style one-sided
    max-ratio thresholding). Positions before the first complete window
    are NaN.
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

    wins = _make_windows(train, window_size)  # (N, L)
    # Shape convention: (L, N, 1) — time x batch x feats — to match torch's
    # default (seq, batch, feat) ordering for Transformer layers.
    X_seq = torch.from_numpy(wins[:, :, None]).permute(1, 0, 2).to(device)

    model = _TranAD(feats=1, window=window_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    N = X_seq.shape[1]
    model.train()
    for epoch in range(num_epochs):
        n_eff = epoch + 1
        perm = torch.randperm(N)
        for s in range(0, N, batch_size):
            idx = perm[s : s + batch_size]
            src = X_seq[:, idx, :]  # (L, B, 1)
            x1, x2 = model(src)
            loss = (1.0 / n_eff) * ((x1 - src) ** 2).mean() + (
                1 - 1.0 / n_eff
            ) * ((x2 - src) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        _, x2 = model(X_seq)
        train_err = ((x2 - X_seq) ** 2).mean(dim=(0, 2)).cpu().numpy()
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
        X_eval = (
            torch.from_numpy(wins[:, :, None]).permute(1, 0, 2).to(device)
        )  # (L, N, 1)
        with torch.no_grad():
            _, x2 = model(X_eval)
            err = ((x2 - X_eval) ** 2).mean(dim=(0, 2)).cpu().numpy()
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
