from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter

## Base

@dataclass(frozen=True, kw_only=True)
class Op(ABC):
    """Base op. `p` is the probability of applying.

    Sampling and application are separated: `sample()` draws concrete params
    from configured ranges, `apply()` is deterministic given them (its rng use
    is reserved for intrinsic randomness, e.g. noise fields). `__call__`
    returns `(image, params)` with the drawn values, or `(image, None)` when
    skipped by `p`.

    Inherits from a normal class (ABC) so jsonargparse accepts subclass
    specs (class_path/init_args) for dataclass subclasses of Op."""

    p: float = 1.0

    def sample(self, rng: np.random.Generator) -> dict:
        """Draw concrete params for one application. Params held at ranges
        (tuples) are sampled uniformly; scalars pass through."""
        return {
            k: rng.uniform(*v) if isinstance(v, tuple) else v
            for k, v in vars(self).items()
            if k != "p"
        }

    @abstractmethod
    def apply(self, image: np.ndarray, rng: np.random.Generator, **params) -> np.ndarray:
        """Apply the op to `image` given concrete params. May use `rng` for
        intrinsic randomness (e.g. the noise field itself)."""

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict | None]:
        if rng.random() >= self.p:
            return image, None
        params = self.sample(rng)
        return self.apply(image, rng, **params), params

## OPS 

@dataclass(frozen=True, kw_only=True)
class Noise(Op):
    """Additive: img + N(0, s); multiplicative: img * (1 + N(0, s))."""

    sigma: tuple[float, float] = (0.05, 0.15)
    mode: Literal["multiplicative", "additive"] = "multiplicative"

    def apply(self, image: np.ndarray, rng: np.random.Generator, sigma: float, mode: str) -> np.ndarray:
        n = rng.normal(0.0, sigma, size=image.shape).astype(image.dtype)
        if mode == "multiplicative":
            return np.clip(image * (1.0 + n), 0.0, 1.0)
        return np.clip(image + n, 0.0, 1.0)


@dataclass(frozen=True, kw_only=True)
class Blur(Op):
    """Per-channel gaussian blur. `truncate = kernel_size / (2*sigma)` keeps
    the effective kernel at ~kernel_size taps for the sampled sigma."""

    sigma: tuple[float, float] = (0.5, 2.0)

    def apply(self, image: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
        # rng unused: blur is deterministic once sigma is drawn
        return gaussian_filter(image, sigma=(sigma, sigma, 0.0)).astype(
            image.dtype
        )


@dataclass(frozen=True, kw_only=True)
class ShotNoise(Op):
    """Poisson photon shot noise: each pixel is a Poisson draw with mean
    `image * peak`, rescaled back by `1/peak`. `peak` is the photon budget —
    higher means brighter counts and proportionally less relative noise."""

    peak: tuple[float, float] = (10.0, 1000.0)

    def apply(self, image: np.ndarray, rng: np.random.Generator, peak: float) -> np.ndarray:
        noisy = rng.poisson(np.clip(image, 0.0, None) * peak).astype(image.dtype) / peak
        return np.clip(noisy, 0.0, 1.0)


@dataclass(frozen=True, kw_only=True)
class ContrastLimits(Op):
    """Per-channel contrast limits (Fiji/Icy-style display window): values
    below `vmin` map to 0, above `vmax` to 1, linear in between, clipped to
    [0, 1] at the end. Limits may lie outside [0, 1] — a window narrower
    than the data saturates (boosts contrast), a window wider than it
    compresses contrast. Bounds are drawn independently per channel from the
    two ranges; a drawn pair is sorted so vmin < vmax (with a minimum window
    of `eps`)."""

    vmin: tuple[float, float] = (-0.2, 0.2)
    vmax: tuple[float, float] = (0.8, 1.2)
    eps: float = 1e-3

    def sample(self, rng: np.random.Generator) -> dict:
        # Draw per-channel bounds here so apply stays deterministic given
        # its params; eps passes through as a scalar.
        n_lo, n_hi = rng.uniform(*self.vmin), rng.uniform(*self.vmax)
        return {"vmin": n_lo, "vmax": n_hi, "eps": self.eps}

    def apply(self, image: np.ndarray, rng: np.random.Generator, vmin: float, vmax: float, eps: float) -> np.ndarray:
        # rng unused: the per-channel draw happened in sample()
        lo, hi = sorted((vmin, vmax))
        return np.clip(
            (image - lo) / max(hi - lo, eps), 0.0, 1.0
        ).astype(image.dtype)
