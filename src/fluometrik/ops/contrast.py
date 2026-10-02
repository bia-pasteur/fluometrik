from dataclasses import dataclass
from typing import Literal

import numpy as np

from .base import Op


@dataclass(frozen=True, kw_only=True)
class ContrastLimits(Op):
    """Fiji/Icy-style contrast limits: values below `vmin` map to 0, above
    `vmax` to 1, linear in between, clipped to [0, 1] at the end. Limits may
    lie outside [0, 1] — a window narrower than the data saturates (boosts
    contrast), a window wider than it compresses contrast. Defaults to
    per-channel windows (`channel_mode="independent"`, one drawn pair per
    channel); with `channel_mode="joint"` a single pair is shared by all
    channels. A drawn pair is sorted so vmin < vmax (with a minimum window
    of `eps`)."""

    channel_mode: Literal["joint", "independent"] = "independent"
    vmin: tuple[float, float] = (-0.2, 0.2)
    vmax: tuple[float, float] = (0.8, 1.2)
    eps: float = 1e-3

    def apply(self, image, rng, vmin: float, vmax: float, eps: float) -> np.ndarray:
        # rng unused; in independent mode the per-channel draw happened in
        # the base __call__.
        lo, hi = sorted((vmin, vmax))
        return np.clip((image - lo) / max(hi - lo, eps), 0.0, 1.0).astype(image.dtype)


@dataclass(frozen=True, kw_only=True)
class Equalize(Op):
    """Per-channel histogram equalization on 256 bins, blended with the
    original by `strength` (0 = identity, 1 = full equalization). Full
    equalization forces a flat histogram and is very destructive on
    microscopy images; small strengths act as a mild contrast
    redistribution."""

    channel_mode: Literal["joint", "independent"] = "independent"
    strength: tuple[float, float] = (0.0, 0.5)

    def apply(self, image: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
        # rng unused; in independent mode the per-channel draw happened in
        # the base __call__.
        out = np.empty_like(image)
        for ch in range(image.shape[-1]):
            q = np.clip((image[..., ch] * 255).round().astype(np.int16), 0, 255)
            hist = np.bincount(q.ravel(), minlength=256).astype(np.float64)
            cdf = hist.cumsum()
            # standard equalize formula (OpenCV): shift by the CDF at the first
            # occupied bin so the peak bin isn't forced to 1.0
            occupied = np.nonzero(hist)[0]
            cmin = cdf[occupied[0]] if len(occupied) else 0
            lut = (cdf - cmin) / max(cdf[-1] - cmin, 1.0)
            eq = lut[q]
            out[..., ch] = (1.0 - strength) * image[..., ch] + strength * eq
        return out.astype(image.dtype)


@dataclass(frozen=True, kw_only=True)
class Invert(Op):
    """Invert the image: `img -> 1 - img`."""

    channel_mode: Literal["joint", "independent"] = "independent"

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return 1.0 - image


@dataclass(frozen=True, kw_only=True)
class Gamma(Op):
    """Detector response / exposure gamma: `img ** gamma`. gamma > 1 darkens
    midtones, gamma < 1 brightens them. Must be positive; the image is
    clipped to [0, 1] first so the power stays finite."""

    channel_mode: Literal["joint", "independent"] = "independent"
    gamma: tuple[float, float] = (0.5, 2.0)

    def apply(self, image: np.ndarray, rng: np.random.Generator, gamma: float) -> np.ndarray:
        # rng unused; in independent mode the per-channel draw happened in
        # the base __call__.
        return np.clip(image, 0.0, 1.0) ** gamma
