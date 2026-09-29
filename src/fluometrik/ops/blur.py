from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.ndimage import gaussian_filter

from .base import Op


@dataclass(frozen=True, kw_only=True)
class Blur(Op):
    """Per-channel gaussian blur. `truncate = kernel_size / (2*sigma)` keeps
    the effective kernel at ~kernel_size taps for the sampled sigma."""

    supported_modes: ClassVar[tuple[str, ...]] = ("joint",)
    sigma: tuple[float, float] = (0.5, 2.0)

    def apply(self, image: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
        # rng unused: blur is deterministic once sigma is drawn
        return gaussian_filter(image, sigma=(sigma, sigma, 0.0)).astype(
            image.dtype
        )
