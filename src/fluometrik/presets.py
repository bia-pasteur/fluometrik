from collections.abc import Callable
from dataclasses import replace

from .ops import (
    Background,
    Blur,
    ChannelPermutation,
    ContrastLimits,
    Equalize,
    Gamma,
    Invert,
    Noise,
    ShotNoise,
)
from .pipeline import Pipeline


# Pixel size the preset ranges were calibrated on (samples/sample_0 metadata).
REFERENCE_UM_PER_PX = 0.11


def default(resolution: float = REFERENCE_UM_PER_PX) -> Pipeline:
    """The standard augmentation pipeline, for images at `resolution`
    (µm/px; default = sample_0's calibration).

    Only pixel-unit parameters depend on it, and the ranges are calibrated in
    pixels at the reference pixel size, so the conversion is inverse: at
    coarser sampling (larger µm/px) the same physical structures span fewer
    pixels. Applies to blur sigma and Background's feature size (a fraction
    of image size). Everything else (gamma, contrast window, noise and
    background amplitude) is dimensionless and unchanged.
    """
    scale = REFERENCE_UM_PER_PX / resolution

    #TODO tune values
    return Pipeline(steps=(
        Background(
            p=0.2,
            scale=(0.15 * scale, 1.5 * scale)
        ),
        Blur(
            p=0.4,
            sigma=(1.5 * scale, 6.0 * scale)
        ),
        ShotNoise(
            p=0.5,
            peak=(5.0, 100.0)
        ),
        Noise(
            p=0.1,
            sigma=(0.05, 0.25)
        ),
        Gamma(
            p=0.5,
            channel_mode="independent",
            gamma=(0.3, 2.5)
        ),
        Equalize(
                    p=0.3,
                    strength=(0.0, 0.2)
                ),
        ContrastLimits(
            p=0.5,
            vmin=(-0.5, 0.2),
            vmax=(0.8, 1.5)
        ),
    ))


def heavy(resolution: float = REFERENCE_UM_PER_PX) -> Pipeline:
    """Like `default`, with every step more likely and stronger ranges."""
    scale = REFERENCE_UM_PER_PX / resolution
    return Pipeline(steps=(
        Background(
            p=0.4,
            scale=(0.15 * scale, 1.5 * scale)
        ),
        Blur(
            p=0.5,
            sigma=(1.5 * scale, 6.0 * scale)
        ),
        ShotNoise(
            p=0.8,
            peak=(5.0, 50.0)
        ),
        Noise(
            p=0.5,
            sigma=(0.05, 0.25)
        ),
        Gamma(
            p=0.8,
            channel_mode="independent",
            gamma=(0.3, 2.5)
        ),
        Equalize(
                    p=0.5,
                    strength=(0.0, 0.5)
                ),
        ContrastLimits(
            p=0.8,
            vmin=(-0.5, 0.2),
            vmax=(0.8, 1.5)
        ),
    ))


PRESETS: dict[str, Callable[..., Pipeline]] = {
    "default": default,
    "heavy": heavy,
}


def load_pipeline(
    name: str = "default",
    resolution: float = REFERENCE_UM_PER_PX,
    overrides: dict[int, dict] | None = None,
) -> Pipeline:
    """Build a registered preset for images at `resolution` (µm/px),
    optionally overriding op fields.

    Overrides address steps by index, e.g.
    `load_pipeline("default", 0.226, {1: {"p": 1.0}})`.
    """
    pipeline = PRESETS[name](resolution=resolution)
    steps = list(pipeline.steps)
    for idx, patch in (overrides or {}).items():
        steps[idx] = replace(steps[idx], **patch)
    return replace(pipeline, steps=tuple(steps))
