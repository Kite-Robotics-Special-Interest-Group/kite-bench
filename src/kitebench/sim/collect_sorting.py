"""Collect sorting demonstrations (3 object classes -> 2 pads).

Usage::

    python -m kitebench.sim.collect_sorting --episodes 300 --out out/sorting
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kitebench.sim.sorting import INSTRUCTIONS, OBJECTS, build, run_place, sample_positions


def rot_to_6d(r: np.ndarray) -> np.ndarray:
    b1, b2 = r[:, 0], r[:, 1]
    return np.array([b1[0], b2[0], b1[1], b2[1], b1[2], b2[2]])


def collect_episode(seed: int, out_dir: Path, frame_every: int = 4,
                    steps_per_phase: int = 400, record_every: int = 25,
                    attempts: int = 4) -> dict | None:
    import mujoco

    from kitebench.sim.scene import IMAGE_SIZE, render_with, wrist_camera

    rng = np.random.default_rng(seed)
    for attempt in range(attempts):
        positions = sample_positions(rng)
        target = list(OBJECTS)[(seed + attempt) % len(OBJECTS)]
        instruction = INSTRUCTIONS[target]
        model = build(positions)
        data = mujoco.MjData(model)
        episode = run_place(model, data, instruction, record_every, steps_per_phase)
        if episode.success:
            break
    else:
        return None

    ep_dir = out_dir / f"{seed:04d}"
    (ep_dir / "obs").mkdir(parents=True, exist_ok=True)
    actions = np.stack([
        np.concatenate([s["ee"], rot_to_6d(s["ee_rot"]), [s["grip"]]]) for s in episode.states
    ]).astype(np.float32)
    proprio = np.stack([
        np.concatenate([s["qpos"][0:6], [s["grip"]]]) for s in episode.states
    ]).astype(np.float32)
    np.save(ep_dir / "action.npy", actions)
    np.save(ep_dir / "proprio.npy", proprio)

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
        "instruction": instruction,
        "target": target,
        "positions": {k: list(v) for k, v in positions.items()},
        "success": True,
        "distance": round(episode.distance, 4),
        "steps": len(episode.states),
        "frames": frames,
        "action_layout": "[ee_xyz(3), rot6d(6), grip(1)]",
    }
    (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="collect sorting demonstrations")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("out/sorting"))
    parser.add_argument("--frame-every", type=int, default=4)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for seed in range(args.start, args.start + args.episodes):
        meta = collect_episode(seed, args.out, args.frame_every)
        if meta is None:
            print(json.dumps({"seed": seed, "kept": False}), flush=True)
            continue
        kept += 1
        print(json.dumps({"seed": seed, "kept": True, "target": meta["target"],
                          "dist": meta["distance"]}), flush=True)
    print(f"kept {kept}/{args.episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


