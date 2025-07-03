"""
parameter_intervener.py
-----------------------
Selective *variance inflation* + noise injection utilities.

Typical workflow
----------------
>>> model = build_my_model()
>>> from parameter_intervener import ParameterIntervener
>>> pi = ParameterIntervener(model)

# 1) Inflate small variances
>>> mask = pi.inflate_variance(threshold=1e-4, factor=5.0, min_value=1e-4)

# 2) Add Gaussian noise to the means *at those exact locations*
>>> pi.add_noise_to_means(mask, std_scale=1.0)

Extra goodies
-------------
3) Gradient‑free pruning
   >>> pi.prune_means_to_zero(mask)          # zero‑out the selected means

4) Reverse‑KL regulariser
   >>> old_state = pi.snapshot_state()       # grab a copy *before* training
   >>> kl_pen = pi.reverse_kl_penalty(old_state)
   >>> loss = task_loss + 1e‑4 * kl_pen      # add to your objective
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Any
import numpy as np

try:  # avoid hard dependency if you run this on NumPy weights
    import torch

    _HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    _HAS_TORCH = False


def _to_numpy(x) -> np.ndarray:
    """Detach Torch tensor (if any) and convert to float32 NumPy."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


class ParameterIntervener:
    """
    Quickly bump up tiny variances and (optionally) inject noise into means.

    The class discovers buffers called:
        • var_w / var_b   (variances)
        • mu_w  / mu_b    (means)
    on every sub-module, so it plays nicely with cuTAGI *and* any custom layer
    that follows the same naming convention.
    """

    # ------------------------------------------------------------------ #
    #  Construction helpers
    # ------------------------------------------------------------------ #
    def __init__(self, model: Any):
        self.model = model

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def inflate_variance(
        self,
        threshold: float = 1e-4,
        *,
        factor: float = 2.0,
        min_value: float | None = None,
        layers: Tuple[str, ...] | None = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Multiply variance entries **below `threshold`** by `factor`.

        Returns
        -------
        mask_dict : {layer_name: {buffer_name: bool-ndarray}}
            Boolean masks pinpointing every element that was changed.
        """
        mask_dict: Dict[str, Dict[str, np.ndarray]] = {}

        for lname, layer, buf_name, buf in self._iter_buffers(which="var"):
            if layers and lname not in layers:
                continue

            arr = _to_numpy(buf)
            mask = arr < threshold
            if not mask.any():
                continue

            new_arr = arr.copy()
            new_arr[mask] *= factor
            if min_value is not None:
                new_arr[mask] = np.maximum(new_arr[mask], min_value)

            # write back, preserving dtype / device if using Torch
            if _HAS_TORCH and isinstance(buf, torch.Tensor):
                updated = torch.as_tensor(new_arr, dtype=buf.dtype, device=buf.device)
            else:
                updated = new_arr
            setattr(layer, buf_name, updated)

            mask_dict.setdefault(lname, {})[buf_name] = mask

        return mask_dict

    def add_noise_to_means(
        self,
        mask_dict: Dict[str, Dict[str, np.ndarray]],
        *,
        std_scale: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        """
        Inject *Gaussian* noise into every mean covered by `mask_dict`.

        Noise ∼ 𝒩(0, σ²·std_scale²) where σ² is the current (inflated) variance.
        """
        rng = rng or np.random.default_rng()

        for lname, submap in mask_dict.items():
            layer = self._named_modules_dict().get(lname)
            if layer is None:
                continue

            for var_buf_name, mask in submap.items():
                mu_buf_name = var_buf_name.replace("var", "mu")
                if not hasattr(layer, mu_buf_name):
                    continue

                mu = _to_numpy(getattr(layer, mu_buf_name))
                var = _to_numpy(getattr(layer, var_buf_name))

                if mu.shape != mask.shape:
                    raise ValueError(
                        f"Shape mismatch in layer '{lname}': mask {mask.shape} vs mu {mu.shape}"
                    )

                noise = rng.standard_normal(mu.shape) * np.sqrt(var) * std_scale
                mu_new = mu.copy()
                mu_new[mask] += noise[mask]

                if _HAS_TORCH and isinstance(getattr(layer, mu_buf_name), torch.Tensor):
                    updated = torch.as_tensor(
                        mu_new,
                        dtype=getattr(layer, mu_buf_name).dtype,
                        device=getattr(layer, mu_buf_name).device,
                    )
                else:
                    updated = mu_new
                setattr(layer, mu_buf_name, updated)

    # ------------------------------------------------------------------ #
    #  3) Gradient‑free pruning                                          #
    # ------------------------------------------------------------------ #
    def prune_means_to_zero(self, mask_dict: Dict[str, Dict[str, np.ndarray]]):
        """
        Set *means* covered by ``mask_dict`` to **exactly zero**.

        This is a simple, gradient‑free pruning mechanism that can be useful
        when you want to silence poorly estimated parameters without retraining.

        Parameters
        ----------
        mask_dict : mapping returned by :py:meth:`inflate_variance`.
        """
        for lname, sub in mask_dict.items():
            layer = self._named_modules_dict().get(lname)
            if layer is None:
                continue
            for var_buf_name, mask in sub.items():
                mu_buf_name = var_buf_name.replace("var", "mu")
                if not hasattr(layer, mu_buf_name):
                    continue

                mu_buf = getattr(layer, mu_buf_name)
                mu = _to_numpy(mu_buf)
                mu_new = mu.copy()
                mu_new[mask] = 0.0  # hard prune

                if _HAS_TORCH and isinstance(mu_buf, torch.Tensor):
                    updated = torch.as_tensor(
                        mu_new, dtype=mu_buf.dtype, device=mu_buf.device
                    )
                else:
                    updated = mu_new
                setattr(layer, mu_buf_name, updated)

    # ------------------------------------------------------------------ #
    #  4) Reverse‑KL regulariser                                         #
    # ------------------------------------------------------------------ #
    def reverse_kl_penalty(
        self,
        snapshot: Dict[str, np.ndarray],
        *,
        eps: float = 1e-9,
    ) -> float:
        """
        Compute the *reverse* KL divergence  KL(current‖snapshot)  summed over
        all (mu, var) pairs.

        Use this as an additional loss term to discourage variances from
        collapsing back after you deliberately inflated them.

        Parameters
        ----------
        snapshot : dict
            Output of :py:meth:`snapshot_state` captured **before** training.
        eps : float, default=1e‑9
            Numerical jitter to keep variances strictly positive.

        Returns
        -------
        float
            Scalar KL value (sum over all parameters).
        """
        total = 0.0
        for lname, layer, var_name, var_buf in self._iter_buffers(which="var"):
            mu_name = var_name.replace("var", "mu")
            key_var = f"{lname}/{var_name}"
            key_mu = f"{lname}/{mu_name}"
            if key_var not in snapshot or key_mu not in snapshot:
                continue

            var_cur = _to_numpy(var_buf)
            mu_cur = _to_numpy(getattr(layer, mu_name))
            var_old = np.maximum(snapshot[key_var], eps)
            mu_old = snapshot[key_mu]

            var_cur = np.maximum(var_cur, eps)

            kl = (
                np.log(np.sqrt(var_old) / np.sqrt(var_cur))
                + (var_cur + (mu_cur - mu_old) ** 2) / (2.0 * var_old)
                - 0.5
            )
            total += kl.sum()

        return float(total)

    # ------------------------------------------------------------------ #
    #  Some handy extras (feel free to remove or extend)
    # ------------------------------------------------------------------ #
    def variance_summary(self) -> Dict[str, Tuple[float, float, float]]:
        """Return {layer_name: (min, mean, max)} across *all* variance buffers."""
        stats = {}
        for lname, *_, buf in self._iter_buffers(which="var"):
            arr = _to_numpy(buf)
            stats[lname] = (float(arr.min()), float(arr.mean()), float(arr.max()))
        return stats

    def snapshot_state(self) -> Dict[str, np.ndarray]:
        """Quick NumPy snapshot of *all* mu/var buffers (for reversible edits)."""
        snap = {}
        for lname, layer, buf_name, buf in self._iter_buffers(which=None):
            snap[f"{lname}/{buf_name}"] = _to_numpy(buf).copy()
        return snap

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _named_modules_dict(self) -> Dict[str, Any]:
        return dict(self._named_modules())

    def _named_modules(self):
        if hasattr(self.model, "named_modules"):
            yield from ((n, m) for n, m in self.model.named_modules() if n)
        elif hasattr(self.model, "layers"):
            for idx, lyr in enumerate(self.model.layers):
                yield str(idx), lyr

    def _iter_buffers(self, *, which: str | None = "var"):
        """
        Iterate over (layer_name, layer, buffer_name, buffer).

        If `which=None`, returns *every* mu/var pair it can find.
        """
        suffixes = ("w", "b")
        kinds = ("mu", "var") if which is None else (which,)
        for lname, layer in self._named_modules():
            for kind in kinds:
                for sfx in suffixes:
                    attr = f"{kind}_{sfx}"
                    if hasattr(layer, attr):
                        buf = getattr(layer, attr)
                        if buf is not None:
                            yield lname, layer, attr, buf
