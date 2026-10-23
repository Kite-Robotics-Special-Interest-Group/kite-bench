from __future__ import annotations

import numpy as np
import pytest

from kitebench import metrics


def make_chunk(pos=0.0, rot6=(1, 0, 0, 1, 0, 0), grip=0.0, scale=1.0, steps=30) -> np.ndarray:
    chunk = np.zeros((steps, 20))
    chunk[:, 0:3] = pos
    chunk[:, 3:9] = rot6
    chunk[:, 9] = grip
    chunk[:, 10:13] = pos * scale
    return chunk


def test_identical_chunks_have_zero_drift() -> None:
    a = make_chunk(pos=0.1, grip=0.2)
    drift = metrics.action_drift(a, a.copy())
    assert drift == {"pos_mm": 0.0, "rot_deg": 0.0, "grip_flip": 0.0}


def test_position_drift_in_millimetres() -> None:
    a = make_chunk(pos=0.0)
    b = make_chunk(pos=0.0)
    b[:, 0] = 0.01  # 1 cm along x
    assert metrics.action_drift(a, b)["pos_mm"] == pytest.approx(10.0, abs=1e-6)


def test_rotation_drift_for_ninety_degrees() -> None:
    # 6D layout is interleaved: [b1x, b2x, b1y, b2y, b1z, b2z]
    a = make_chunk(rot6=(1, 0, 0, 1, 0, 0))          # b1=(1,0,0), b2=(0,1,0)
    b = make_chunk(rot6=(0, -1, 1, 0, 0, 0))         # b1=(0,1,0), b2=(-1,0,0)
    assert metrics.action_drift(a, b)["rot_deg"] == pytest.approx(90.0, abs=1e-3)


def test_gripper_flip_rate() -> None:
    a = make_chunk(grip=0.1)
    b = make_chunk(grip=0.1)
    b[::2, 9] = 0.99  # every other step flips to closed
    assert metrics.action_drift(a, b)["grip_flip"] == pytest.approx(0.5)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        metrics.action_drift(np.zeros((30, 20)), np.zeros((10, 20)))


def test_success_and_score() -> None:
    assert metrics.success_rate([True, False, True, True]) == pytest.approx(0.75)
    assert metrics.success_rate([]) == 0.0
    assert metrics.robust_score(0.0) == 1.0
    assert metrics.robust_score(50.0) == 0.0

