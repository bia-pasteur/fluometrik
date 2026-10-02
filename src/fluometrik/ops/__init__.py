from .base import Op
from .background import Background
from .blur import Blur
from .channel import ChannelPermutation
from .contrast import ContrastLimits, Equalize, Gamma, Invert
from .noise import Noise, ShotNoise

OPS = [
    Noise, Blur, ShotNoise, ContrastLimits, Gamma, Background,
    Equalize, Invert, ChannelPermutation,
]
