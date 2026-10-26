"""Closed-loop evaluation of the sorting task (pick the named object, place on its pad)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kitebench.demo_ik import DemoIK
from kitebench.sim.oracle import GRIP_CLOSED, GRIP_OPEN, reset
from kitebench.sim.sorting import OBJECTS, PAD_LEFT, PAD_RIGHT, build, sample_positions


def rollout(model, data, policy, instruction: str, chunks: int = 10,
            steps_per_action: int = 25, exec_steps: int = 10, demo_ik=None) -> dict:
    import mujoco

    reset(model, data)
    names = list(OBJECTS)
    target_idx = names.index(next(n for n in names if n in instruction))
    pad = PAD_LEFT if "left" in instruction else PAD_RIGHT
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wx250s/gripper_link")
    obj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"body{target_idx}")

    prev_closed = False
    hold = 0
    for _ in range(chunks):
        from kitebench.sim.scene import IMAGE_SIZE, render

        image = render(model, data, "front", IMAGE_SIZE)
        proprio = np.zeros(20, dtype=np.float32)
        proprio[0:3] = data.xpos[gid]
        proprio[3:10] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        chunk = policy.act(image, proprio, instruction)
        for step in range(min(exec_steps, chunk.shape[0])):
            target = np.clip(chunk[step, 0:3], -0.5, 0.6)
            target[2] = max(target[2], 0.01)
            closing = chunk[step, 9] >= 0.5
            if closing and not prev_closed:
                hold = 5
            prev_closed = closing
            if hold > 0:
                hold -= 1
                target = data.xpos[gid].copy()
            if demo_ik is not None:
                q = demo_ik.solve(target, data.qpos[0:6].copy())
            else:
                from kitebench.sim.oracle import ik

                q = ik(model, data, target, data.qpos[0:6].copy(), iters=200)
            data.ctrl[0:6] = q
            data.ctrl[6] = GRIP_CLOSED if closing else GRIP_OPEN
            for _ in range(steps_per_action):
                mujoco.mj_step(model, data)

    final = data.xpos[obj_id]
    distance = float(np.linalg.norm(final[0:2] - np.array(pad)))
    return {"success": distance < 0.07, "distance": round(distance, 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description="closed-loop sorting evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--head", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start", type=int, default=3000)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--exec", type=int, default=10)
    parser.add_argument("--demo-ik", action="store_true")
    parser.add_argument("--demo-source", type=Path, default=Path("out/sorting"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import mujoco

    from kitebench.eval import XVLAAdapter
    from kitebench.sim.sorting import INSTRUCTIONS

    policy = XVLAAdapter(args.model, steps=5, seed=0, head=args.head)
    demo_ik = DemoIK(args.demo_source) if args.demo_ik else None

    results = []
    for seed in range(args.start, args.start + args.episodes):
        rng = np.random.default_rng(seed)
        positions = sample_positions(rng)
        names = list(OBJECTS)
        target = names[seed % len(names)]
        instruction = INSTRUCTIONS[target]
        model = build(positions)
        data = mujoco.MjData(model)
        res = rollout(model, data, policy, instruction, args.chunks, demo_ik=demo_ik)
        res.update({"seed": seed, "target": target, "instruction": instruction})
        results.append(res)
        print(json.dumps(res, ensure_ascii=False), flush=True)

    ok = sum(r["success"] for r in results)
    print(f"success {ok}/{len(results)}")
    if args.out:
        args.out.write_text(json.dumps({"success": ok, "total": len(results), "results": results},
                                       indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
