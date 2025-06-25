"""
view_param.py ― cuTAGI / PyTorch parameter visualiser
-----------------------------------------------------
• Understands TAGI buffers (mu_w / var_w) plus classic .weight tensors.
• Robust to raw Python lists.
• Handles standard Dense/Conv layers and custom cuTAGI LSTM layers.
• Provides:
    - CuTAGIParameterViewer.heatmap(layer_name, which={"mean","var"})
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Union
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio  # for GIF creation


# ------------------------------------------------------------------ #
#  Utils
# ------------------------------------------------------------------ #
def _to_numpy(x) -> np.ndarray:
    """
    Convert torch.Tensor | np.ndarray | list-like to a NumPy array on CPU.
    (import kept local to avoid a hard torch dependency.)
    """
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


# ------------------------------------------------------------------ #
#  Main viewer class
# ------------------------------------------------------------------ #
class ParameterViewer:
    """
    Quick visualisation helper for cuTAGI (or PyTorch) models.

    Parameters
    ----------
    model      : pytagi.nn.Sequential | torch.nn.Module
    out_dir    : str | Path | None   – if given, save figs there instead of showing
    default_v  : {"mean","var","both"} – default family of params to plot

    Additional:
        • set_epoch(epoch) ― store current training epoch
        • heatmap(..., epoch=None, cmap=None) ― optional epoch stamp + custom cmap
        • toggle saving with save_figs=True|False on construction, or per-call save=
        • group frames with gif_id=<name>; save via save_gif(key=<name>)
    """

    def __init__(
        self,
        model,
        out_dir: Union[str, Path, None] = "nn_param_plots",
        default_v: Literal["mean", "var", "both"] = "mean",
        *,
        save_figs: bool = True,
    ):
        self.model = model
        self.default_v = default_v
        self.save_figs = bool(save_figs)
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.current_epoch: int | None = None
        # collected RGB frames for GIF export, grouped by key (e.g. layer name)
        self._frames: dict[str, list[np.ndarray]] = defaultdict(list)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def heatmap(
        self,
        layer_name: str,
        which: Literal["mean", "var"] = "mean",
        *,
        epoch: int | None = None,
        cmap: str | None = None,
        return_img: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        layer_type: str | None = None,
        gif_id: str | None = None,
        save: bool | None = None,
    ):
        """
        Visualise the weight matrix / conv filter bank of one layer.

        layer_name – name reported by Sequential/named_modules ("0", "fc1", …)
        which      – "mean" for μ, "var" for σ²
        epoch      – optional int, appended to filename as "ep###"
        cmap       – Matplotlib colour‑map name (e.g. "viridis", "plasma")
        return_img – if True, returns the RGB image as a NumPy array
        save       – override global saving behaviour for this call
        vmin, vmax – optional floats to set a fixed color scale for the heatmap
        """
        epoch = int(epoch) if epoch is not None else self.current_epoch
        layer = self._locate_layer(layer_name)
        attr = f"{'mu' if which == 'mean' else 'var'}_w"
        if not hasattr(layer, attr):
            raise ValueError(f"Layer '{layer_name}' has no {attr} buffer.")
        w = _to_numpy(getattr(layer, attr))
        fig = None

        # ---- LSTM Layer: 4 gates stacked in 2D ------------------------
        if layer_type == "LSTM":
            out_f = layer.output_size
            in_f = layer.input_size + layer.output_size
            expected_shape = (4 * out_f, in_f)

            # Reshape if needed (1D vector)
            if w.ndim == 1:
                if w.size == expected_shape[0] * expected_shape[1]:
                    w = w.reshape(expected_shape)
                    print(
                        f"[INFO] Reshaped flat LSTM weights to {expected_shape} for visualization."
                    )
                else:
                    raise ValueError(
                        f"LSTM layer '{layer_name}' has flat weights of shape {w.shape}, "
                        f"but cannot reshape to expected {expected_shape} "
                        f"(4 * {out_f}, {in_f}) = {4*out_f*in_f} elements required)."
                    )

            if w.shape == expected_shape:
                gate_names = ["Forget", "Input", "Cell State", "Output"]
                gate_colors = ["red", "green", "blue", "purple"]

                vmin_plot = vmin if vmin is not None else float(np.min(w))
                vmax_plot = vmax if vmax is not None else float(np.max(w))
                if vmin_plot == vmax_plot:
                    eps = 1e-6 or abs(vmin_plot) * 1e-3
                    vmin_plot -= eps
                    vmax_plot += eps

                fig, ax = plt.subplots(figsize=(10, 8))
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")
                im = ax.imshow(
                    w,
                    aspect="auto",
                    interpolation="nearest",
                    vmin=vmin_plot,
                    vmax=vmax_plot,
                    cmap=cmap,
                )
                im.set_clim(vmin=vmin_plot, vmax=vmax_plot)
                fig.colorbar(
                    im,
                    ax=ax,
                    shrink=0.8,
                    pad=0.02,
                    ticks=np.linspace(vmin_plot, vmax_plot, 5),
                )

                ax.set_title(
                    f"{which.capitalize()} weights – LSTM Layer '{layer_name}'"
                )
                ax.set_title(ax.get_title(), fontsize=14, fontweight="bold", pad=15)
                ax.set_xlabel("Input + Recurrent Weights")
                ax.set_xlabel(ax.get_xlabel(), fontsize=12)
                ax.xaxis.set_label_position("bottom")
                ax.xaxis.set_ticks_position("bottom")
                ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
                ax.tick_params(labelsize=10)
                ax.tick_params(
                    which="both",
                    bottom=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False,
                )

                # Horizontal lines between gates
                for i in range(1, 4):
                    ax.axhline(
                        i * out_f - 0.5, color="gray", linestyle="--", linewidth=1
                    )

                # Gate name annotations
                for i, name in enumerate(gate_names):
                    ax.text(
                        -0.5,
                        i * out_f + out_f / 2,
                        name,
                        color=gate_colors[i],
                        fontsize=10,
                        verticalalignment="center",
                        horizontalalignment="right",
                        fontweight="bold",
                    )
            else:
                raise ValueError(
                    f"LSTM layer '{layer_name}' has unexpected shape {w.shape}, expected {expected_shape}. "
                    f"Ensure weights are stored as a 2D matrix with shape [4*out_f, in_f + out_f]."
                )

        # ---- Dense layer heat-map --------------------
        else:
            if w.ndim == 1:
                out_f = None
                for a in ("out_features", "out_dim", "output_features", "output_size"):
                    if hasattr(layer, a):
                        out_f = getattr(layer, a)
                        break
                if out_f is None:
                    for b in (f"{'mu' if which == 'mean' else 'var'}_b", "bias"):
                        if hasattr(layer, b) and getattr(layer, b) is not None:
                            out_f = _to_numpy(getattr(layer, b)).size or None
                            if out_f:
                                break
                w = (
                    w.reshape((out_f, w.size // out_f))
                    if out_f and w.size % out_f == 0
                    else w.reshape(1, -1)
                )

            vmin_plot = vmin if vmin is not None else float(np.min(w))
            vmax_plot = vmax if vmax is not None else float(np.max(w))

            if vmin_plot == vmax_plot:
                eps = 1e-6 or abs(vmin_plot) * 1e-3
                vmin_plot -= eps
                vmax_plot += eps

            fig, ax = plt.subplots()
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            im = ax.imshow(
                w,
                aspect="auto",
                interpolation="nearest",
                vmin=vmin_plot,
                vmax=vmax_plot,
                cmap=cmap,
            )
            im.set_clim(vmin=vmin_plot, vmax=vmax_plot)
            fig.colorbar(
                im,
                ax=ax,
                shrink=0.8,
                pad=0.02,
                ticks=np.linspace(vmin_plot, vmax_plot, 5),
            )
            ax.set_title(f"{which.capitalize()} weight matrix – {layer_name}")
            ax.set_title(ax.get_title(), fontsize=14, fontweight="bold", pad=15)
            ax.set_xlabel("Input dim")
            ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            ax.set_ylabel("Output dim")
            ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
            ax.tick_params(labelsize=10)
            ax.tick_params(
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )

        # Determine whether this call should write a PNG
        save = self.save_figs if save is None else bool(save)

        parts = ["heat", layer_name, which]
        if epoch is not None:
            parts.append(f"ep{epoch:03d}")

        img_arr = None
        if return_img and fig:
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            arr = np.frombuffer(buf, dtype=np.uint8)
            reported_w, reported_h = fig.canvas.get_width_height()
            total_pixels = arr.size // 3
            scale = int(round((total_pixels / (reported_w * reported_h)) ** 0.5)) or 1
            img_arr = arr.reshape(reported_h * scale, reported_w * scale, 3)

        gif_key = gif_id or layer_name

        # store frame for potential GIF export
        if return_img and img_arr is not None:
            self._frames[gif_key].append(img_arr)

        if fig:
            self._save_or_show(fig, "_".join(parts), return_img, save)
        return img_arr

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _save_or_show(self, fig, fname: str, return_img: bool, save: bool):
        should_show_interactive = (self.out_dir is None or not save) and not return_img
        if save and self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.out_dir / f"{fname}.png", dpi=150, bbox_inches="tight")
        if should_show_interactive:
            plt.show()
        plt.close(fig)

    def _iter_named_modules(self):
        if hasattr(self.model, "named_modules"):
            for name, mod in self.model.named_modules():
                if name:
                    yield name, mod
        elif hasattr(self.model, "layers"):
            for idx, layer in enumerate(self.model.layers):
                yield str(idx), layer

    def _locate_layer(self, layer_name: str):
        for n, m in self._iter_named_modules():
            if n == layer_name:
                return m
        raise KeyError(f"Layer '{layer_name}' not found.")

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    # ------------------------------------------------------------------
    #  GIF export helpers
    # ------------------------------------------------------------------
    def clear_frames(self, key: str | None = None):
        """
        Discard stored frames.

        Parameters
        ----------
        key : str | None
            If given, clear only frames for that GIF key (layer);
            otherwise wipe everything.
        """
        if key is None:
            self._frames.clear()
        else:
            self._frames.pop(key, None)

    def save_gif(
        self,
        filepath: str | Path,
        *,
        key: str,
        fps: int = 2,
        loop: int = 0,
    ):
        """
        Save the accumulated frames as an animated GIF.

        Parameters
        ----------
        key : str
            Identifier whose frames will be rendered (usually the layer name
            you passed as `gif_id` or defaulted to in `heatmap`).
        filepath : str | Path
            Output file path (should end with '.gif').
        fps : int, default=2
            Frames per second in the resulting GIF.
        loop : int, default=0
            Number of animation loops (0 = infinite).
        """
        if key not in self._frames or not self._frames[key]:
            raise RuntimeError(
                f"No frames stored under key '{key}'. "
                "Make sure to call heatmap(..., return_img=True, gif_id=key) first."
            )

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # imageio expects duration *per frame* in seconds
        duration = 1.0 / fps if fps > 0 else 0.5
        imageio.mimsave(
            filepath,
            self._frames[key],
            format="GIF",
            duration=duration,
            loop=loop,
        )
