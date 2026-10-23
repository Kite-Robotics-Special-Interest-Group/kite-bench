"""Drift and success metrics for action chunks.

Action chunks follow the WidowX ``ee6d`` layout: 20 dims per step, where
``0:3`` is the end-effector position, ``3:9`` a 6D rotation and ``9`` the gripper
probability. Two arms are packed in one chunk (``10:13`` / ``13:19`` / ``19``); the
metrics below use the first arm by default.
"""

from __future__ import annotations

import numpy as np


def rot6d_to_matrix(v6: np.ndarray) -> np.ndarray:
    """Convert the two 3-vectors of a 6D rotation into a rotation matrix."""
    a1 = np.asarray(v6, dtype=float)[0::2]
    a2 = np.asarray(v6, dtype=float)[1::2]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, in degrees."""
    cos = (np.trace(r_a.T @ r_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def action_drift(clean: np.ndarray, perturbed: np.ndarray, gripper_threshold: float = 0.91) -> dict:
    """Compare two ``[T, 20]`` action chunks.

    Returns position drift in millimetres, rotation drift in degrees and the
    fraction of steps where the gripper open/close decision flipped.
    """
    clean = np.asarray(clean, dtype=float)
    perturbed = np.asarray(perturbed, dtype=float)
    if clean.shape != perturbed.shape:
        raise ValueError(f"shape mismatch: {clean.shape} vs {perturbed.shape}")

    pos_mm = float(np.mean(np.linalg.norm(perturbed[:, 0:3] - clean[:, 0:3], axis=1)) * 1000.0)

    rots = [
        rotation_angle_deg(rot6d_to_matrix(clean[t, 3:9]), rot6d_to_matrix(perturbed[t, 3:9]))
        for t in range(clean.shape[0])
    ]
    clean_closed = clean[:, 9] >= gripper_threshold
    perturbed_closed = perturbed[:, 9] >= gripper_threshold
    grip_flip = float(np.mean(perturbed_closed != clean_closed))
    return {
        "pos_mm": round(pos_mm, 2),
        "rot_deg": round(float(np.mean(rots)), 2),
        "grip_flip": round(grip_flip, 3),
    }


def success_rate(results: list[bool]) -> float:
    if not results:
        return 0.0
    return float(np.mean([bool(r) for r in results]))


def robust_score(pos_mm: float, scale_mm: float = 50.0) -> float:
    """Normalised drift score: 1.0 is identical output, lower is worse."""
    return float(max(0.0, 1.0 - pos_mm / scale_mm))


def summarise(rows: list[dict]) -> dict:
    """Aggregate per-perturbation drift rows into a single report dict."""
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "pos_mm_mean": round(float(np.mean([r["pos_mm"] for r in rows])), 2),
        "rot_deg_mean": round(float(np.mean([r["rot_deg"] for r in rows])), 2),
        "grip_flip_mean": round(float(np.mean([r["grip_flip"] for r in rows])), 3),
    }

