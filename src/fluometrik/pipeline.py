from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .ops import Op


@dataclass(frozen=True, kw_only=True)
class Pipeline:
    """A sequence of ops applied in order. Returns the image and one params
    record per step (None when the step was skipped by its probability)."""

    steps: tuple[Op, ...]

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[dict | None]]:
        records = []
        for step in self.steps:
            image, params = step(image, rng)
            records.append(params)
        return image, records

    def __iter__(self) -> Iterator[Op]:
        return iter(self.steps)
