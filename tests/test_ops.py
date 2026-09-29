
import dataclasses

import numpy as np
import pytest

import fluometrik
from fluometrik import Blur, ContrastLimits, Noise, Pipeline

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
    assert isinstance(params, dict | list)


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
    """__call__ returns the concrete params actually used, sampled within
    ranges. Independent ops report a list of per-channel dicts; joint ops a
    single dict."""
    rng = rng_stream(5)
    img = rng_stream(2).random((8, 8, 3)).astype(np.float32)
    out, params = op(img, rng)
    n_channels = img.shape[-1]
    if isinstance(params, list):
        assert len(params) == n_channels
        records = params
    else:
        records = [params]
    for record in records:
        assert record, f"{type(op).__name__} returned empty params"
        for key, value in record.items():
            field = getattr(type(op), key, None)
            if isinstance(field, tuple) and len(field) == 2 and all(np.isscalar(v) for v in field):
                lo, hi = sorted(field)
                assert lo <= value <= hi, f"{type(op).__name__}.{key}={value} outside {field}"
            else:
                assert value == field, f"{type(op).__name__}.{key}={value} != default {field}"


@pytest.mark.parametrize("op", ALL_OPS, ids=ids)
def test_op_channel_mode_record_shape(op):
    """Ops report a single dict in joint mode and a list of per-channel dicts
    in independent mode (where supported); skipped ops report None."""
    img = rng_stream(4).random((8, 8, 3)).astype(np.float32)
    if "joint" in op.supported_modes:
        joint, params_j = dataclasses.replace(op, channel_mode="joint")(img, rng_stream(6))
        assert isinstance(params_j, dict)
    if "independent" in op.supported_modes:
        indep, params_i = dataclasses.replace(op, channel_mode="independent")(img, rng_stream(6))
        assert isinstance(params_i, list) and len(params_i) == img.shape[-1]
        assert all(isinstance(r, dict) for r in params_i)


def test_unsupported_channel_mode_raises():
    """An op may restrict its modes: constructing with an unsupported one raises."""
    with pytest.raises(ValueError, match="does not support"):
        Blur(channel_mode="independent")


def test_independent_mode_draws_per_channel():
    """An independent op with narrow, distinct ranges must yield params that
    actually differ channel to channel."""
    op = ContrastLimits(vmin=(0.0, 0.01), vmax=(0.5, 0.51), p=1.0)
    img = rng_stream(3).random((8, 8, 4)).astype(np.float32)
    _, params = op(img, rng_stream(8))
    assert len(params) == 4
    vmins = [r["vmin"] for r in params]
    assert len(set(vmins)) == 4, f"expected 4 distinct draws, got {vmins}"


def test_joint_mode_shares_params_across_channels():
    """A joint range op samples once: params are scalars. Intrinsic randomness
    (the noise field) stays per-element, so channels are not correlated."""
    op = Noise(channel_mode="joint", sigma=(0.05, 0.15), mode="additive", p=1.0)
    img = rng_stream(3).random((8, 8, 4)).astype(np.float32)
    out, params = op(img, rng_stream(9))
    assert isinstance(params, dict) and np.isscalar(params["sigma"])
    noise = out - img  # additive noise, per channel
    assert not np.allclose(noise[..., 0], noise[..., 1])


def test_pipeline_applies_in_order():
    """Pipeline output == manual sequential application with the same rng stream."""
    img = rng_stream(5).random((8, 8, 3)).astype(np.float32)
    steps = [Noise(sigma=(0.05, 0.15)), Blur(sigma=(0.5, 1.0))]

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
