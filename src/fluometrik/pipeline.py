from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from .ops import Op


@dataclass(frozen=True, kw_only=True)
class Pipeline:
    """A sequence of ops applied in order. Returns the image and one params
    record per step (None when the step was skipped by its probability)."""

    steps: list[Op]

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[dict | None]]:
        records = []
        for step in self.steps:
            image, params = step(image, rng)
            records.append(params)
        return image, records

    def __iter__(self) -> Iterator[Op]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, i: int) -> Op:
        return self.steps[i]

    def drop(self, *ops: type[Op]) -> "Pipeline":
        """A copy of the pipeline without steps of the given op type(s),
        identified by class, e.g. `pipeline.drop(Background, Blur)`."""
        return replace(self, steps=[s for s in self.steps if type(s) not in ops])
