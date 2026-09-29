from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np


@dataclass(frozen=True, kw_only=True)
class Op(ABC):
    """Base op. `p` is the probability of applying.

    Sampling and application are separated: `sample()` draws concrete params
    from configured ranges, `apply()` is deterministic given them (its rng use
    is reserved for intrinsic randomness, e.g. noise fields). `__call__`
    returns `(image, params)` with the drawn values, or `(image, None)` when
    skipped by `p`.

    `channel_mode` decides how multi-channel (H, W, C) images are treated:
    "joint" samples once and applies to the full image (the op decides how
    channels interact); "independent" splits the image into (H, W, 1)
    slices, samples and applies per channel, and reassembles — so the op's
    `apply` sees a single-channel image. In "independent" mode the params
    record is a list of per-channel dicts (one entry per channel); in
    "joint" mode it is a single dict.

    The `apply` contract: it receives either the full (H, W, C) image
    ("joint") or one (H, W, 1) channel ("independent"), must return an
    array of the same shape and dtype with values in [0, 1], and must
    handle any channel count it is given (a (H, W, 1) image reaches both
    modes identically). Params are the dataclass fields minus `p` and
    `channel_mode`, with tuple ranges already resolved to scalars.

    An op may restrict its channel treatment with `supported_modes` —
    constructing it with an unsupported `channel_mode` raises.

    Inherits from a normal class (ABC) so jsonargparse accepts subclass
    specs (class_path/init_args) for dataclass subclasses of Op."""

    p: float = 1.0
    channel_mode: Literal["joint", "independent"] = "joint"
    supported_modes: ClassVar[tuple[str, ...]] = ("joint", "independent")

    def __post_init__(self) -> None:
        if self.channel_mode not in self.supported_modes:
            raise ValueError(
                f"{type(self).__name__} does not support channel_mode={self.channel_mode!r}; "
                f"supported: {self.supported_modes}"
            )

    def sample(self, rng: np.random.Generator) -> dict:
        """Draw concrete params for one application. Params held at ranges
        (tuples) are sampled uniformly; scalars pass through."""
        return {
            k: rng.uniform(*v) if isinstance(v, tuple) else v
            for k, v in vars(self).items()
            if k not in ("p", "channel_mode")
        }

    @abstractmethod
    def apply(self, image: np.ndarray, rng: np.random.Generator, **params) -> np.ndarray:
        """Apply the op to `image` given concrete params. May use `rng` for
        intrinsic randomness (e.g. the noise field itself)."""

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict | list[dict] | None]:
        if rng.random() >= self.p:
            return image, None
        if self.channel_mode == "independent":
            outs, records = [], []
            for c in range(image.shape[-1]):
                params = self.sample(rng)
                outs.append(self.apply(image[..., c: c + 1], rng, **params))
                records.append(params)
            return np.concatenate(outs, axis=-1), records
        elif self.channel_mode == "joint":
            params = self.sample(rng)
            return self.apply(image, rng, **params), params
        else:
            raise ValueError(self.channel_mode)
