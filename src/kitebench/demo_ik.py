"""Nearest-demo joint targets, matched on a short trajectory window.

A single EE position is ambiguous (approaching vs retreating pass through the same
point), so the lookup matches the *next few* predicted positions against demo
windows. That keeps the arm on the branch the demonstrations used and preserves
the direction of travel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WINDOW = 3


class DemoIK:
    def __init__(self, episodes: Path = Path("out/episodes_v4"), stride: int = 1,
                 window: int = WINDOW):
        self.window = window
        windows: list[np.ndarray] = []
        joints: list[np.ndarray] = []
        for ep in sorted(p for p in episodes.iterdir() if p.is_dir()):
            a_path, p_path = ep / "action.npy", ep / "proprio.npy"
            if not (a_path.exists() and p_path.exists()):
                continue
            actions = np.load(a_path)[::stride]
            proprio = np.load(p_path)[::stride]
            n = len(actions)
            if n < window:
                continue
            ees = actions[:, 0:3].astype(np.float32)
            qs = proprio[:, 0:6].astype(np.float32)
            for i in range(n - window + 1):
                windows.append(ees[i : i + window])
                joints.append(qs[i])
        self.windows = np.stack(windows).astype(np.float32) if windows else np.zeros((0, window, 3))
        self.q = np.stack(joints).astype(np.float32) if joints else np.zeros((0, 6))
        print(f"[demo-ik] {len(self.q)} reference windows from {episodes}")

    def solve(self, target: np.ndarray, q_prev: np.ndarray | None = None) -> np.ndarray:
        """``target`` is either one EE point or the next ``window`` predicted points."""
        pts = np.asarray(target, dtype=np.float32)
        if pts.ndim == 1:
            pts = np.repeat(pts[None, :], self.window, axis=0)
        if len(pts) < self.window:
            pad = np.repeat(pts[-1:], self.window - len(pts), axis=0)
            pts = np.concatenate([pts, pad], axis=0)
        pts = pts[: self.window]

        if not len(self.q):
            raise RuntimeError("empty demo reference set")
        d = np.linalg.norm(self.windows - pts[None, :, :], axis=2).sum(axis=1)
        k = min(5, len(d))
        idx = np.argpartition(d, k - 1)[:k]
        if q_prev is None:
            return self.q[idx[np.argmin(d[idx])]].copy()
        best, best_cost = None, None
        for i in idx:
            cost = d[i] * 50.0 + np.linalg.norm(self.q[i] - q_prev) * 0.5
            if best_cost is None or cost < best_cost:
                best, best_cost = i, cost
        return self.q[best].copy()
