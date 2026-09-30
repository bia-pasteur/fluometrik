from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import zoom

from .base import Op


@dataclass(frozen=True, kw_only=True)
class Background(Op):
    """Additive smooth low-frequency background, modeling out-of-focus
    fluorescence and sample autofluorescence: a random field drawn on a coarse
    grid (`scale` = feature size as a fraction of the smaller spatial dim) is
    bilinearly upsampled, normalized to [0, 1], and added with weight
    `amplitude`. Small `scale` values yield fine structure, large ones a
    near-constant offset."""

    channel_mode: Literal["joint", "independent"] = "independent"
    amplitude: tuple[float, float] = (0.05, 0.3)
    scale: tuple[float, float] = (0.05, 0.5)

    def apply(self, image: np.ndarray, rng: np.random.Generator, amplitude: float, scale: float) -> np.ndarray:
        h, w = image.shape[:2]
        corr = max(scale * min(h, w), 1.0)
        gh = max(int(round(h / corr)), 1)
        gw = max(int(round(w / corr)), 1)
        field = rng.random((gh, gw))
        field = zoom(field, (h / gh, w / gw), order=1, mode="reflect")
        field = field[:h, :w]
        if field.shape != (h, w):  # zoom can round up by a pixel
            field = np.pad(field, ((0, h - field.shape[0]), (0, w - field.shape[1])), mode="edge")
        return np.clip(image + amplitude * field[..., None], 0.0, 1.0).astype(image.dtype)
