"""Randomised episode collection for imitation learning.

Each episode rebuilds the scene with a jittered cube position, table/cube colours,
light intensity and camera offset, runs the scripted oracle, then dumps states,
actions and a sparse set of frames.

Usage::

    python -m kitebench.sim.collect --episodes 5 --out episodes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kitebench.sim.oracle import run_pick
from kitebench.sim.scene import CUBE_SIZE, CUBE_XY, IMAGE_SIZE, build

INSTRUCTION = "pick up the red block"


def vary(rng: np.random.Generator) -> tuple[tuple[float, float], dict]:
    """Sample a cube position and an appearance variant."""
    cube_xy = (
        CUBE_XY[0] + float(rng.uniform(-0.05, 0.05)),
        CUBE_XY[1] + float(rng.uniform(-0.05, 0.05)),
    )
    mat = rng.uniform(0.30, 0.60, 3)
    cube = np.clip(np.array([0.78, 0.16, 0.14]) + rng.normal(0, 0.08, 3), 0, 1)
    light = float(rng.uniform(0.75, 1.0))
    variant = {
        "mat_rgba": f"{mat[0]:.3f} {mat[1]:.3f} {mat[2]:.3f} 1",
        "cube_rgba": f"{cube[0]:.3f} {cube[1]:.3f} {cube[2]:.3f} 1",
        "light": f"{light:.3f} {light:.3f} {light:.3f}",
        "cam_offset": tuple(float(v) for v in rng.uniform(-0.04, 0.04, 3)),
    }
    return cube_xy, variant


def rot_to_6d(r: np.ndarray) -> np.ndarray:
    """First two columns, interleaved as [b1x, b2x, b1y, b2y, b1z, b2z]."""
    b1, b2 = r[:, 0], r[:, 1]
    return np.array([b1[0], b2[0], b1[1], b2[1], b1[2], b2[2]])


def collect_episode(seed: int, out_dir: Path, frame_every: int = 20,
                    steps_per_phase: int = 200) -> dict:
    import mujoco

    rng = np.random.default_rng(seed)
    cube_xy, variant = vary(rng)
    model = build(cube_xy, variant)
    data = mujoco.MjData(model)
    episode = run_pick(model, data, steps_per_phase=steps_per_phase)

    ep_dir = out_dir / f"{seed:04d}"
    (ep_dir / "obs").mkdir(parents=True, exist_ok=True)

    actions = np.stack([
        np.concatenate([s["ee"], rot_to_6d(s["ee_rot"]), [s["grip"]]]) for s in episode.states
    ]).astype(np.float32)
    proprio = np.stack(episode.proprio).astype(np.float32)
    np.save(ep_dir / "action.npy", actions)
    np.save(ep_dir / "proprio.npy", proprio)

    from kitebench.sim.scene import render_with, wrist_camera

    renderer = mujoco.Renderer(model, IMAGE_SIZE, IMAGE_SIZE)
    frames = []
    for i in range(0, len(episode.states), frame_every):
        data.qpos[:] = episode.states[i]["qpos"]
        mujoco.mj_forward(model, data)
        render_with(renderer, data, "front").save(ep_dir / "obs" / f"{i:04d}_front.png")
        render_with(renderer, data, wrist_camera(model, data)).save(
            ep_dir / "obs" / f"{i:04d}_wrist.png"
        )
        frames.append(i)
    renderer.close()

    meta = {
        "seed": seed,
        "instruction": INSTRUCTION,
        "cube_xy": list(cube_xy),
        "cube_size": CUBE_SIZE,
        "variant": variant,
        "success": bool(episode.success),
        "cube_z_final": round(float(episode.cube_z), 4),
        "steps": len(episode.states),
        "frames": frames,
        "action_layout": "[ee_xyz(3), rot6d(6), grip(1)]",
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="collect randomised pick episodes")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("episodes"))
    parser.add_argument("--frame-every", type=int, default=20)
    parser.add_argument("--steps-per-phase", type=int, default=200)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i in range(args.start, args.start + args.episodes):
        meta = collect_episode(i, args.out, args.frame_every, args.steps_per_phase)
        ok += int(meta["success"])
        print(json.dumps({"seed": i, "success": meta["success"],
                          "cube_z": meta["cube_z_final"]}, ensure_ascii=False))
    print(f"collected {args.episodes}, success {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
