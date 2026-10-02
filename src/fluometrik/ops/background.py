from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from scipy.ndimage import gaussian_filter, rotate, zoom

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


@dataclass(frozen=True, kw_only=True)
class OutOfFocusBackground(Op):
    """Additive out-of-focus background: a heavily blurred copy of the image
    itself, geometrically misaligned (random rotation, flip and scale, so it
    never lines up with the in-focus layer), added on top with weight
    `amplitude`. Models out-of-focus fluorescence from structures above/below
    the focal plane — blur destroys their alignment with the sharp layer, and
    geometric decorrelation avoids an aligned-ghost look. Applied jointly on
    all channels (one geometry per frame) so channels stay coherent."""

    supported_modes: ClassVar[tuple[str, ...]] = ("joint",)
    amplitude: tuple[float, float] = (0.05, 0.4)
    scale: tuple[float, float] = (1.05, 1.3)
    blur_sigma: tuple[float, float] = (8.0, 20.0)

    def apply(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
        amplitude: float,
        scale: float,
        blur_sigma: float,
    ) -> np.ndarray:
        h, w = image.shape[:2]

        layer = image
        if rng.random() < 0.5:  # flip side and rotation angle are intrinsic randomness
            layer = layer[:, ::-1]
        layer = rotate(layer, rng.uniform(0.0, 360.0), axes=(0, 1), reshape=False, order=1, mode="reflect")
        if scale != 1.0:
            layer = zoom(layer, (scale, scale, 1.0), order=1, mode="reflect")
        layer = layer[:h, :w]
        if layer.shape[:2] != (h, w):
            layer = np.pad(layer, ((0, h - layer.shape[0]), (0, w - layer.shape[1]), (0, 0)), mode="edge")
        if layer.shape[-1] != image.shape[-1]:
            layer = layer[..., : image.shape[-1]]

        layer = gaussian_filter(layer, sigma=(blur_sigma, blur_sigma, 0.0))

        # normalize to peak 1 so `amplitude` is comparable to Background's,
        # regardless of the source image's histogram
        peak = layer.max()
        if peak > 0:
            layer = layer / peak

        return np.clip(image + amplitude * layer, 0.0, 1.0).astype(image.dtype)
