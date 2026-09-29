from dataclasses import dataclass
from typing import Literal

import numpy as np

from .base import Op


@dataclass(frozen=True, kw_only=True)
class Noise(Op):
    """Additive: img + N(0, s); multiplicative: img * (1 + N(0, s))."""

    channel_mode: Literal["joint", "independent"] = "independent"
    sigma: tuple[float, float] = (0.05, 0.15)
    mode: Literal["multiplicative", "additive"] = "multiplicative"

    def apply(self, image: np.ndarray, rng: np.random.Generator, sigma: float, mode: str) -> np.ndarray:
        n = rng.normal(0.0, sigma, size=image.shape).astype(image.dtype)
        if mode == "multiplicative":
            return np.clip(image * (1.0 + n), 0.0, 1.0)
        return np.clip(image + n, 0.0, 1.0)


@dataclass(frozen=True, kw_only=True)
class ShotNoise(Op):
    """Poisson photon shot noise: each pixel is a Poisson draw with mean
    `image * peak`, rescaled back by `1/peak`. `peak` is the photon budget —
    higher means brighter counts and proportionally less relative noise."""

    channel_mode: Literal["joint", "independent"] = "independent"
    peak: tuple[float, float] = (10.0, 1000.0)

    def apply(self, image: np.ndarray, rng: np.random.Generator, peak: float) -> np.ndarray:
        noisy = rng.poisson(np.clip(image, 0.0, None) *
                            peak).astype(image.dtype) / peak
        return np.clip(noisy, 0.0, 1.0)
