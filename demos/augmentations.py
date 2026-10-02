"""Demo: apply a preset pipeline to several images.

For each (image, channels, preset) entry, two figures are written next to
this script, named `<stem>[_c<idx>...]_<preset>_...`:
- `demo(...)`: one column per op, one row per independent sample of that op.
- `demo_pipeline_grid(...)`: samples_per_op x samples_per_op cells, each an
  independent sample of the whole pipeline.

Each image is cropped to its biggest centered square (no padding) before the
demo runs on it.

    pixi run -e dev python demos/augmentations.py --help
    pixi run -e dev python demos/augmentations.py --config demos/demo.yaml
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import auto_cli

from fluometrik import Op, Pipeline, load_pipeline
from fluometrik.utils import normalize_percentile

logger = logging.getLogger(__name__)


@dataclass
class DemoEntry:
    """One demo run: an image, an optional channel subset and a preset name.

    `channels` restricts the ops to a channel subset (e.g. [0, 1] = R, G for
    an RGB image); dropped channels are shown un-augmented in the figures.
    `preset` names a registered preset (see `fluometrik.presets.PRESETS`),
    built for `resolution` (µm/px)."""

    image: str
    channels: list[int] | None = None
    preset: str = "default"
    resolution: float = 0.1132496161970507


def load_image(path: str | Path) -> np.ndarray:
    img = np.asarray(iio.imread(str(path)))
    if img.ndim == 2:  # (H, W) -> (H, W, 1)
        img = img[..., None]
    return normalize_percentile(img.astype(np.float32))


def crop_biggest_square(img: np.ndarray) -> np.ndarray:
    """Crop to the largest centered square the image contains (no padding)."""
    side = min(img.shape[0], img.shape[1])
    h0 = (img.shape[0] - side) // 2
    w0 = (img.shape[1] - side) // 2
    return img[h0: h0 + side, w0: w0 + side]


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
        def kv(k, v): return f"{k}={v:.3g}" if isinstance(
            v, float) else f"{k}={v}"
        return ", ".join(kv(k, v) for k, v in params.items())
    lines = []
    for c, record in enumerate(params):
        pairs = ", ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in record.items()
        )
        lines.append(f"c{c}: {pairs}")
    return "\n".join(lines)


def splice(processed: np.ndarray, base: np.ndarray, channels: list[int] | None) -> np.ndarray:
    """Re-embed processed channel slices into the full base image at their
    original slots, so dropped channels show un-augmented."""
    if channels is None or processed.shape[-1] == base.shape[-1]:
        return processed
    out = base.copy()
    for i, c in enumerate(channels):
        out[..., c] = processed[..., i]
    return out


def demo(
    img: np.ndarray,
    ops: list[Op] | None = None,
    samples_per_op: int = 4,
    out: Path = Path(__file__).parent / "demo_augmentations.png",
    show: bool = True,
    base: np.ndarray | None = None,
    channels: list[int] | None = None,
) -> None:
    """Render one column per op, one row per independent sample.

    If `channels` restricts the ops to a channel subset, `base` is the full
    original image and dropped channels are shown un-augmented."""
    n_ops = len(ops)
    fig, axes = plt.subplots(
        samples_per_op,
        n_ops + 1,
        figsize=(5 * (n_ops + 1), 5 * samples_per_op),
        squeeze=False,
    )

    for col, ax in enumerate(axes[0]):
        ax.set_title("input" if col == 0 else type(
            ops[col - 1]).__name__, fontsize=10)
        ax.axis("off")

    disp = to_display(splice(img, base, channels) if channels else img)
    for row in range(samples_per_op):
        axes[row][0].imshow(disp, cmap="gray" if disp.ndim == 2 else None)
        axes[row][0].axis("off")

    for col, op in enumerate(ops, start=1):
        name = type(op).__name__
        for row in range(samples_per_op):
            out_img, params = op(img, np.random.default_rng(row))
            disp = to_display(splice(out_img, base, channels)
                              if channels else out_img)
            ax = axes[row][col]
            label = format_params(params)
            ax.set_title(f"{name}\n{label}")
            ax.imshow(disp, cmap="gray" if disp.ndim ==
                      2 else None, vmin=0, vmax=1)
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
    base: np.ndarray | None = None,
    channels: list[int] | None = None,
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
            out_img, _ = pipeline(
                img, np.random.default_rng(row * samples_per_op + col))
            disp = to_display(splice(out_img, base, channels)
                              if channels else out_img)
            ax = axes[row][col]
            ax.imshow(disp, cmap="gray" if disp.ndim ==
                      2 else None, vmin=0, vmax=1)
            ax.axis("off")

    fig.suptitle(
        f"Pipeline: {' -> '.join(type(op).__name__ for op in ops)}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    logger.info("wrote %s", out)
    if show:
        plt.show()


def run_entry(
    entry: DemoEntry,
    samples_per_op: int,
    show: bool,
) -> None:
    """Crop to the biggest square, select channels, build the preset and
    render both figures."""
    img = load_image(entry.image)
    img = crop_biggest_square(img)
    if entry.channels is not None:
        base, img = img, img[..., entry.channels]
    else:
        base = img

    pipeline = load_pipeline(entry.preset, resolution=entry.resolution)
    ops = list(pipeline)

    stem = Path(entry.image).stem
    out_dir = Path(__file__).parent
    # single-augs figure: force p=1 so every column shows an applied op
    demo(
        img, ops=[replace(op, p=1.0) for op in ops],
        samples_per_op=samples_per_op,
        show=False, base=base, channels=entry.channels,
        out=out_dir / f"{stem}_{entry.preset}_single-augs.png")
    demo_pipeline_grid(
        img, ops=ops, samples_per_op=samples_per_op,
        show=show, base=base, channels=entry.channels,
        out=out_dir / f"{stem}_{entry.preset}_augs-grid.png")


def main(
    entries: list[DemoEntry],
    samples_per_op: int = 4,
    show: bool = True,
) -> None:
    """Run the same demo process on every (image, channels, preset) entry."""
    for entry in entries:
        logger.info(
            "demo %s channels=%s preset=%s", entry.image, entry.channels,
            entry.preset)
        run_entry(entry, samples_per_op=samples_per_op, show=show)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    auto_cli(main)
