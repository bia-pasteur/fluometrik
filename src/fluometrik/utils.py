import numpy as np


def normalize_percentile(
    image: np.ndarray, low: float = 1.0, high: float = 99.0
) -> np.ndarray:
    """Clamp to the (low, high) percentiles and rescale so low -> 0.0, high -> 1.0.

    Works on (H, W) or (H, W, C) arrays (a single gain/offset for all channels).
    A degenerate image (hi == lo, e.g. constant) maps to all zeros.
    """
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, (low, high))
    if hi > lo:
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image)
    return np.clip(image, 0.0, 1.0)