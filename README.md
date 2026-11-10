# 🦠 Fluometrik

[![PyPI version](https://img.shields.io/pypi/v/fluometrik)](https://pypi.org/project/fluometrik/)
[![Python versions](https://img.shields.io/pypi/pyversions/fluometrik)](https://pypi.org/project/fluometrik/)
[![License](https://img.shields.io/pypi/l/fluometrik)](https://github.com/bia-pasteur/fluometrik/blob/main/LICENSE)

Photometric augmentation for fluorescence (and biological) images.

Ops are composed into a `Pipeline` and applied to `(H, W, C)` float images
with values in `[0, 1]`. Each op samples concrete parameters from configured
ranges (tuple = uniform range, scalar = fixed), then applies them
deterministically.

## Install
```bash
pip install fluometrik
```
Or the latest source:
```bash
pip install git+https://github.com/bia-pasteur/fluometrik.git
```

## Usage

```python
import numpy as np
from fluometrik import Pipeline, Noise, Blur, ShotNoise, ContrastLimits

img = np.random.random((64, 64, 3)).astype(np.float32)  # (H, W, C) float in [0, 1]

pipeline = Pipeline(steps=[
    Blur(sigma=(1.0, 4.0)),
    Noise(sigma=(0.1, 1.0), mode="multiplicative"),
    ShotNoise(peak=(5, 50)),
    ContrastLimits(vmin=(-0.2, 0.2), vmax=(0.8, 1.2)),
])
out, records = pipeline(img, np.random.default_rng(0))
```

`Pipeline` also supports `drop(OpType)` (a copy without the steps of the
given op types) and plain indexing/iteration over its steps. Presets are
built for a pixel size via `load_pipeline("default", resolution)` (µm/px);
registered presets: `default`, `fast`, `heavy` (see
`fluometrik.presets.PRESETS`).

Or declaratively with jsonargparse (`class_path`/`init_args`):

```yaml
ops:
  - class_path: fluometrik.Blur
    init_args: {p: 1.0, sigma: [1.0, 4.0]}
  - class_path: fluometrik.Noise
    init_args: {p: 1.0, sigma: [0.1, 1.0], mode: multiplicative}
```

See `demos/demo.yaml` and `pixi run -e dev python demos/augmentations.py
--config demos/demo.yaml` for a rendered grid of every op.

## Ops by acquisition stage

Ops are grouped by where they act in a fluorescence image acquisition
pipeline: the optics and sample first, then the detector electronics and
photon statistics, then the display/lookup-table stage.

### Optics / sample

| Op | Description | Params |
|---|---|---|
| `Blur` | Isotropic gaussian blur per channel (the channel axis is left unfiltered). | `sigma` (px), `kernel_size` |
| `Background` | Additive smooth low-frequency field (out-of-focus fluorescence / autofluorescence): a random coarse grid upsampled bilinearly, added with weight `amplitude`. | `amplitude`, `scale` |
| `OutOfFocusBackground` | A heavily blurred copy of the image itself, geometrically misaligned (random flip/rotation/scale) and added back: out-of-focus structures never line up with the sharp layer. Joint across channels. | `amplitude`, `scale`, `blur_sigma` |

### Detector electronics

| Op | Description | Params |
|---|---|---|
| `Noise` | Additive (`img + N(0, s)`) or multiplicative (`img * (1 + N(0, s))`) gaussian noise. | `sigma`, `mode` |
| `ShotNoise` | Poisson photon shot noise: each pixel is a Poisson draw with mean `img * peak`, rescaled back by `1/peak`. Higher `peak` = brighter counts, proportionally less relative noise. | `peak` |

### Photometry / display

| Op | Description | Params |
|---|---|---|
| `Gamma` | Detector response / exposure gamma: `img ** gamma` (per channel by default). | `gamma` |
| `Equalize` | Per-channel histogram equalization blended with the original by `strength` (0 = identity, 1 = full flat histogram). | `strength` |
| `ContrastLimits` | Fiji/Icy-style contrast window: values below `vmin` → 0, above `vmax` → 1, linear in between, clipped to `[0, 1]`. A drawn pair is sorted so `vmin < vmax`. | `vmin`, `vmax`, `eps` |
| `Invert` | `img -> 1 - img`. | — |
| `ChannelPermutation` | Random permutation of the channel axis (may be the identity). | — |

## Channel handling

Every op takes a `channel_mode`:

- **`"joint"`** (default) — one parameter draw, applied to the full
  `(H, W, C)` image. The op decides how channels interact (e.g. `Blur`
  filters each channel with the same sigma; `OutOfFocusBackground` uses one
  geometry for all channels so they stay coherent).
- **`"independent"`** — the image is split into `(H, W, 1)` slices; each
  channel gets its own parameter draw and its own `apply`, then channels are
  reassembled. The params record is a list of per-channel dicts.

An op may restrict which modes it supports (`supported_modes`); constructing
it with an unsupported mode raises. `Noise`, `ShotNoise`, `Background`,
`Gamma`, `Equalize` and `ContrastLimits` default to `"independent"`;
`Blur`, `OutOfFocusBackground`, `Invert` (joint/independent equivalent) and
`ChannelPermutation` are joint-only or joint-defaulting.

## Writing an op

Subclass `Op` (a frozen kw-only dataclass) and implement `apply`. Fields
become configurable params: tuple fields are sampled uniformly in `sample()`,
scalars pass through. `apply` receives either the full image (`"joint"`) or
one `(H, W, 1)` channel (`"independent"`), must return the same shape and
dtype with values in `[0, 1]`, and may use `rng` only for intrinsic
randomness (e.g. the noise field itself).

```python
@dataclass(frozen=True, kw_only=True)
class Gamma(Op):
    """img ** gamma, drawn per channel."""

    channel_mode: Literal["joint", "independent"] = "independent"
    gamma: tuple[float, float] = (0.5, 2.0)

    def apply(self, image, rng, gamma) -> np.ndarray:
        return np.clip(image ** gamma, 0.0, 1.0).astype(image.dtype)
```

## Todo
- [ ] bleed-through
- [ ] anisotropic PSF
- [ ] bloom

## Citation

If you use fluometrik in your research, please cite it (see
[CITATION.cff](CITATION.cff)):

```bibtex
@software{mabon_fluometrik,
  author  = {Mabon, Julie},
  title   = {fluometrik: physically-motivated photometric augmentation for fluorescence images},
  url     = {https://github.com/bia-pasteur/fluometrik},
  version = {1.0.1},
  year    = {2026},
}
```

## License

BSD 3-Clause — see [LICENSE](LICENSE).
