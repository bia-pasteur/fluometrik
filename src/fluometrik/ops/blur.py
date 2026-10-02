from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.ndimage import gaussian_filter

from .base import Op


@dataclass(frozen=True, kw_only=True)
class Blur(Op):
    """Per-channel gaussian blur. `truncate = kernel_size / (2*sigma)` keeps
    the effective kernel at ~kernel_size taps for the sampled sigma; with
    `kernel_size=None` scipy's default truncation (4.0 sigma) is used."""

    supported_modes: ClassVar[tuple[str, ...]] = ("joint",)
    sigma: tuple[float, float] = (0.5, 2.0)
    kernel_size: int | None = None

    def apply(self, image: np.ndarray, rng: np.random.Generator, sigma: float, kernel_size: int | None) -> np.ndarray:
        # rng unused: blur is deterministic once sigma is drawn
        truncate = 4.0 if kernel_size is None else kernel_size / (2.0 * sigma)
        return gaussian_filter(image, sigma=(sigma, sigma, 0.0), truncate=truncate).astype(
            image.dtype
        )
