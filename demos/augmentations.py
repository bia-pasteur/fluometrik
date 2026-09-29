"""Demo: apply each augmentation several times to a sample image.

Two figures are written next to this script:
- `demo(...)`: one column per op, one row per independent sample of that op.
- `demo_pipeline_grid(...)`: samples_per_op x samples_per_op cells, each an
  independent sample of the whole op list applied as one Pipeline.

    pixi run -e dev python demos/augmentations.py --help
    pixi run -e dev python demos/augmentations.py --config demos/demo.yaml
"""

import logging
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import auto_cli

from fluometrik import Op, Pipeline
from fluometrik.utils import normalize_percentile

logger = logging.getLogger(__name__)


def load_image(path: str | Path) -> np.ndarray:
    img = np.asarray(iio.imread(str(path)))
    if img.ndim == 2:  # (H, W) -> (H, W, 1)
        img = img[..., None]
        img = normalize_percentile(img).astype(np.float32)
    return img


def to_display(img: np.ndarray) -> np.ndarray:
    """Collapse to what imshow can show: 2D grayscale, or first 3 channels."""
    if img.ndim == 2:
        return img
    if img.shape[-1] == 1:
        return img[..., 0]
    return img[..., :3]


def format_params(params: dict | list[dict] | None) -> str:
    """Render a params record for the tile title: joint ops get k=v pairs,
    independent ops one compact line per channel."""
    if params is None:
        return "skipped"
    if isinstance(params, dict):
        kv = lambda k, v: f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
        return ", ".join(kv(k, v) for k, v in params.items())
    lines = []
    for c, record in enumerate(params):
        pairs = ", ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in record.items()
        )
        lines.append(f"c{c}: {pairs}")
    return "\n".join(lines)


def demo(
    img: np.ndarray,
    ops: list[Op] | None = None,
    samples_per_op: int = 4,
    out: Path = Path(__file__).parent / "demo_augmentations.png",
    show: bool = True,
) -> None:
    """Render one column per op, one row per independent sample."""
    n_ops = len(ops)
    fig, axes = plt.subplots(
        samples_per_op,
        n_ops + 1,
        figsize=(5 * (n_ops + 1), 5 * samples_per_op),
        squeeze=False,
    )

    for col, ax in enumerate(axes[0]):
        ax.set_title("input" if col == 0 else type(ops[col - 1]).__name__, fontsize=10)
        ax.axis("off")

    disp = to_display(img)
    for row in range(samples_per_op):
        axes[row][0].imshow(disp, cmap="gray" if disp.ndim == 2 else None)
        axes[row][0].axis("off")

    for col, op in enumerate(ops, start=1):
        name = type(op).__name__
        for row in range(samples_per_op):
            out_img, params = op(img, np.random.default_rng(row))
            disp = to_display(out_img)
            ax = axes[row][col]
            label = format_params(params)
            ax.set_title(f"{name}\n{label}")
            ax.imshow(disp, cmap="gray" if disp.ndim == 2 else None,vmin=0,vmax=1)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    logger.info("wrote %s", out)
    if show:
        plt.show()


def demo_pipeline_grid(
    img: np.ndarray,
    ops: list[Op] | None = None,
    samples_per_op: int = 4,
    out: Path = Path(__file__).parent / "demo_pipeline_grid.png",
    show: bool = True,
) -> None:
    """Render a samples_per_op x samples_per_op grid of the full op list
    applied as one Pipeline — one independent pipeline sample per cell."""
    pipeline = Pipeline(steps=tuple(ops))
    fig, axes = plt.subplots(
        samples_per_op,
        samples_per_op,
        figsize=(5 * samples_per_op, 5 * samples_per_op),
        squeeze=False,
    )

    for row in range(samples_per_op):
        for col in range(samples_per_op):
            out_img, _ = pipeline(img, np.random.default_rng(row * samples_per_op + col))
            disp = to_display(out_img)
            ax = axes[row][col]
            ax.imshow(disp, cmap="gray" if disp.ndim == 2 else None, vmin=0, vmax=1)
            ax.axis("off")

    fig.suptitle(f"Pipeline: {' -> '.join(type(op).__name__ for op in ops)}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    logger.info("wrote %s", out)
    if show:
        plt.show()


def main(
    image: str = "samples/sample_0.tif",
    ops: list[Op] | None = None,
    patch_size: int = 256,
    samples_per_op: int = 4,
    show: bool = True,
) -> None:
    """Load the sample once, then render both demo figures."""
    img = load_image(image)[:patch_size, :patch_size]
    demo(img, ops=ops, samples_per_op=samples_per_op, show=False)
    demo_pipeline_grid(img, ops=ops, samples_per_op=samples_per_op, show=show)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    auto_cli(main)
