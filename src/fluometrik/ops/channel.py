from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .base import Op


@dataclass(frozen=True, kw_only=True)
class ChannelPermutation(Op):
    """Random permutation of the channel axis (may be the identity). Only
    meaningful for images with more than one channel."""

    supported_modes: ClassVar[tuple[str, ...]] = ("joint",)

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        perm = rng.permutation(image.shape[-1])
        return image[..., perm].astype(image.dtype, copy=False)
