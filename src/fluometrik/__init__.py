from importlib.metadata import version as _version

from .ops import (
    Background,
    Blur,
    ChannelPermutation,
    ContrastLimits,
    Equalize,
    Gamma,
    Invert,
    Noise,
    Op,
    OutOfFocusBackground,
    ShotNoise,
)
from .ops import OPS
from .pipeline import Pipeline
from .presets import PRESETS, load_pipeline

__version__ = _version("fluometrik")
