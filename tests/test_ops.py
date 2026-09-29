
import dataclasses

import numpy as np
import pytest

import fluometrik
from fluometrik import Blur, Noise, Pipeline

SHAPES = [(8, 8, 3), (5, 7, 4), (4, 4, 6), (8, 8, 1), (16, 16, 2)]
DTYPES = [np.float32, np.float64]

rng_stream = lambda seed: np.random.default_rng(seed)


def default_op(cls):
    """Instantiate an op with its dataclass defaults (p=1 so it always fires)."""
    return cls(p=1.0)


ALL_OPS = [default_op(cls) for cls in fluometrik.OPS]


def ids(op):
    return type(op).__name__


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("op", ALL_OPS, ids=ids)
def test_op_preserves_shape_and_dtype(shape, dtype, op):
    rng = rng_stream(0)
    img = rng.random(shape).astype(dtype)
    out, params = op(img, rng)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert isinstance(params, dict)


@pytest.mark.parametrize("op", ALL_OPS, ids=ids)
def test_op_p0_returns_same_image(op):
    """p=0 must return the input unchanged with a None record."""
    rng = rng_stream(42)
    img = rng.random((8, 8, 3)).astype(np.float32)
    op0 = dataclasses.replace(op, p=0.0)
    out, params = op0(img, rng)
    np.testing.assert_array_equal(out, img)
    assert params is None


@pytest.mark.parametrize("op", ALL_OPS, ids=ids)
def test_op_p1_is_deterministic(op):
    """Same seed + p=1 => identical output; and output actually differs from input."""
    img = rng_stream(1).random((8, 8, 3)).astype(np.float32)
    out1, _ = op(img, rng_stream(7))
    out2, _ = op(img, rng_stream(7))
    np.testing.assert_array_equal(out1, out2)
    assert not np.array_equal(out1, img), f"{type(op).__name__} with p=1 changed nothing"


@pytest.mark.parametrize("op", ALL_OPS, ids=ids)
def test_op_output_in_valid_range(op):
    """Outputs stay in [0, 1] for [0, 1] inputs (Float ops clamp)."""
    rng = rng_stream(3)
    img = rng.random((8, 8, 3)).astype(np.float32)
    out, _ = op(img, rng)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


@pytest.mark.parametrize("op", ALL_OPS, ids=ids)
def test_op_stochastic(op):
    """Different seeds give different results (with p=1)."""
    img = rng_stream(2).random((8, 8, 3)).astype(np.float32)
    out1, _ = default_op(type(op))(img, rng_stream(10))
    out2, _ = default_op(type(op))(img, rng_stream(11))
    assert not np.array_equal(out1, out2)


@pytest.mark.parametrize("op", ALL_OPS, ids=ids)
def test_op_reports_sampled_params(op):
    """__call__ returns the concrete params actually used, sampled within ranges."""
    rng = rng_stream(5)
    img = rng_stream(2).random((8, 8, 3)).astype(np.float32)
    out, params = op(img, rng)
    assert isinstance(params, dict) and params, f"{type(op).__name__} returned empty params"
    for key, value in params.items():
        field = getattr(type(op), key, None)
        if isinstance(field, tuple) and len(field) == 2 and all(np.isscalar(v) for v in field):
            lo, hi = sorted(field)
            assert lo <= value <= hi, f"{type(op).__name__}.{key}={value} outside {field}"
        else:
            assert value == field, f"{type(op).__name__}.{key}={value} != default {field}"


def test_pipeline_applies_in_order():
    """Pipeline output == manual sequential application with the same rng stream."""
    img = rng_stream(5).random((8, 8, 3)).astype(np.float32)
    steps = [Noise(sigma=(0.05, 0.15)), Blur(kernel_size=3, sigma=(0.5, 1.0))]

    rng = rng_stream(9)
    out, records = Pipeline(steps=tuple(steps))(img, rng)

    rng = rng_stream(9)
    manual, manual_records = img, []
    for s in steps:
        manual, params = s(manual, rng)
        manual_records.append(params)
    np.testing.assert_array_equal(out, manual)
    assert records == manual_records


def test_pipeline_empty():
    img = rng_stream(0).random((8, 8, 3)).astype(np.float32)
    out, records = Pipeline(steps=())(img, rng_stream(1))
    np.testing.assert_array_equal(out, img)
    assert records == []
