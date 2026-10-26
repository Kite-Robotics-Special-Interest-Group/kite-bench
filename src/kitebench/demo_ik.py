"""Nearest-demo joint targets: map an end-effector position to the joint config the
demonstrations used at (nearly) that position.

The IK solver can land on a different elbow branch for the same EE pose, which
changes the approach angle and breaks grasping (see docs/roadmap.md). The demo
manifold is single-valued in practice, so a nearest-neighbour lookup keeps the
policy on the same branch the demonstrations used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class DemoIK:
    def __init__(self, episodes: Path = Path("out/episodes_v4"), stride: int = 1):
        ees, joints = [], []
        for ep in sorted(p for p in episodes.iterdir() if p.is_dir()):
            a_path, p_path = ep / "action.npy", ep / "proprio.npy"
            if not (a_path.exists() and p_path.exists()):
                continue
            actions = np.load(a_path)[::stride]
            proprio = np.load(p_path)[::stride]
            ees.append(actions[:, 0:3])
            joints.append(proprio[:, 0:6])
        self.ee = np.concatenate(ees).astype(np.float32)
        self.q = np.concatenate(joints).astype(np.float32)
        print(f"[demo-ik] {len(self.ee)} reference points from {episodes}")

    def solve(self, target: np.ndarray, q_prev: np.ndarray | None = None) -> np.ndarray:
        d = np.linalg.norm(self.ee - np.asarray(target, dtype=np.float32), axis=1)
        k = 5
        idx = np.argpartition(d, k)[:k]
        if q_prev is None:
            return self.q[idx[np.argmin(d[idx])]].copy()
        # among the closest points prefer the one closest to the current posture,
        # so the arm does not jump between phases
        best, best_cost = None, None
        for i in idx:
            cost = d[i] * 50.0 + np.linalg.norm(self.q[i] - q_prev) * 0.5
            if best_cost is None or cost < best_cost:
                best, best_cost = i, cost
        return self.q[best].copy()
