from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from kitebench import perturb


@pytest.fixture()
def frame() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))


@pytest.mark.parametrize("name", sorted(perturb.registry()))
def test_shape_and_dtype_preserved(frame: Image.Image, name: str) -> None:
    out = perturb.registry()[name](frame)
    assert out.size == (224, 224)
    assert out.mode == "RGB"


def test_brightness_identity(frame: Image.Image) -> None:
    out = perturb.brightness(frame, 1.0)
    assert np.array_equal(np.asarray(out), np.asarray(frame))


def test_rotation_identity(frame: Image.Image) -> None:
    out = perturb.rotate(frame, 0.0)
    assert np.array_equal(np.asarray(out), np.asarray(frame))


def test_random_perturbations_are_seeded(frame: Image.Image) -> None:
    for name in ("noise", "occlude_rect"):
        fn = perturb.registry()[name]
        first = np.asarray(fn(frame, seed=7))
        second = np.asarray(fn(frame, seed=7))
        other = np.asarray(fn(frame, seed=8))
        assert np.array_equal(first, second)
        assert not np.array_equal(first, other)


def test_occlude_arm_covers_bottom(frame: Image.Image) -> None:
    out = np.asarray(perturb.occlude_arm(frame))
    bottom = out[int(224 * 0.7) :]
    assert bottom.mean() < np.asarray(frame)[int(224 * 0.7) :].mean()
