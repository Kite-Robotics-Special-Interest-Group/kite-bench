"""Closed-loop rollout in MuJoCo (the same renderer/camera used for collection).

The policy predicts absolute end-effector targets; IK turns them into joint
targets for the position servos. Success = cube lifted above 6 cm.

Usage::

    python -m kitebench.rollout --model <weights> --head out/head_ft2.pt --episodes 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kitebench.sim.oracle import GRIP_CLOSED, GRIP_OPEN, LIFT_OK, ik, reset
from kitebench.sim.scene import CUBE_XY, IMAGE_SIZE, build, render

INSTRUCTION = "pick up the red block"


def rollout(model, data, policy, chunks: int, steps_per_action: int = 25,
            exec_steps: int = 10, demo_ik=None) -> dict:
    import mujoco

    reset(model, data)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wx250s/gripper_link")
    best = 0.0

    prev_grip_closed = False
    hold = 0
    for _ in range(chunks):
        image = render(model, data, "front", IMAGE_SIZE)
        proprio = np.zeros(20, dtype=np.float32)
        proprio[0:3] = data.xpos[gid]
        proprio[3:10] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        chunk = policy.act(image, proprio, INSTRUCTION)
        for step in range(min(exec_steps, chunk.shape[0])):
            target = np.clip(chunk[step, 0:3], -0.5, 0.6)
            target[2] = max(target[2], 0.01)
            closing = chunk[step, 9] >= 0.5
            if closing and not prev_grip_closed:
                hold = 5  # let the fingers settle before the arm moves away
            prev_grip_closed = closing
            if hold > 0:
                hold -= 1
                target = data.xpos[gid].copy()
            if demo_ik is not None:
                q = demo_ik.solve(target, data.qpos[0:6].copy())
            else:
                q = ik(model, data, target, data.qpos[0:6].copy(), iters=200)
            data.ctrl[0:6] = q
            data.ctrl[6] = GRIP_CLOSED if closing else GRIP_OPEN
            for _ in range(steps_per_action):
                mujoco.mj_step(model, data)
            best = max(best, float(data.qpos[10]))

    return {
        "success": best > LIFT_OK,
        "cube_z_max": round(best, 4),
        "cube_xy": [round(float(CUBE_XY[0]), 3), round(float(CUBE_XY[1]), 3)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="closed-loop MuJoCo evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--head", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start", type=int, default=300, help="first episode seed (unseen)")
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--demo-source", type=Path, default=Path("out/episodes_v4"),
                        help="episodes used by --demo-ik")
    parser.add_argument("--demo-ik", action="store_true",
                        help="map EE targets to the nearest demo joint configuration")
    parser.add_argument("--exec", type=int, default=10,
                        help="execute only the first N actions of each chunk")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import mujoco

    from kitebench.eval import XVLAAdapter

    policy = XVLAAdapter(args.model, steps=5, seed=0, head=args.head)
    demo_ik = None
    if args.demo_ik:
        from kitebench.demo_ik import DemoIK
        demo_ik = DemoIK(args.demo_source)
    results = []
    for seed in range(args.start, args.start + args.episodes):
        rng = np.random.default_rng(seed)
        cube_xy = (CUBE_XY[0] + float(rng.uniform(-0.05, 0.05)),
                   CUBE_XY[1] + float(rng.uniform(-0.05, 0.05)))
        model = build(cube_xy)
        data = mujoco.MjData(model)
        res = rollout(model, data, policy, args.chunks, args.steps, args.exec, demo_ik)
        res["cube_xy"] = [round(cube_xy[0], 3), round(cube_xy[1], 3)]
        results.append(res)
        print(json.dumps({"seed": seed, **res}), flush=True)

    rate = float(np.mean([r["success"] for r in results]))
    print(f"success {sum(r['success'] for r in results)}/{len(results)} = {rate:.0%}")
    if args.out:
        args.out.write_text(json.dumps({"rate": rate, "results": results}, indent=2),
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())








