"""Scripted pick-and-lift oracle used to generate imitation-learning demos."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kitebench.sim.scene import CUBE_SIZE

READY = np.array([0.0, -0.70, 0.96, 0.0, -0.44, 0.0])
GRIP_OPEN = 0.037
GRIP_CLOSED = 0.016
LIFT_OK = 2 * CUBE_SIZE + 0.02
POSTURE_WEIGHT = 0.35
TOP_DOWN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


@dataclass
class Episode:
    success: bool
    cube_z: float
    actions: list[np.ndarray] = field(default_factory=list)
    proprio: list[np.ndarray] = field(default_factory=list)
    states: list[dict] = field(default_factory=list)


def _gid(model):
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wx250s/gripper_link")


def _rotvec(r: np.ndarray) -> np.ndarray:
    angle = np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return axis / (2.0 * np.sin(angle)) * angle


def ik(model, data, target: np.ndarray, q0: np.ndarray, iters: int = 400) -> np.ndarray:
    """6-DoF damped least squares: position target plus a top-down gripper."""
    import mujoco

    q = q0.copy()
    bid = _gid(model)
    for _ in range(iters):
        data.qpos[0:6] = q
        mujoco.mj_forward(model, data)
        ep = target - data.xpos[bid]
        er = _rotvec(TOP_DOWN @ data.xmat[bid].reshape(3, 3).T)
        err = np.concatenate([ep, 0.5 * er])
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, bid)
        j = np.vstack([jacp[:, 0:6], 0.5 * jacr[:, 0:6]])
        if np.linalg.norm(err) < 1e-3:
            break
        # bias toward the seed posture: the same EE pose has several IK branches
        # (elbow up/down) and picking a different one changes the whole approach
        err = err + POSTURE_WEIGHT * (j @ (q0 - q))
        dq = j.T @ np.linalg.solve(j @ j.T + 0.01 * np.eye(6), err)
        q = np.clip(q + 0.4 * dq, model.jnt_range[0:6, 0], model.jnt_range[0:6, 1])
    return q


def reset(model, data) -> None:
    """Start from the shipped keyframe: the fingers otherwise jam at qpos 0."""
    import mujoco

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[0:6] = READY
    data.ctrl[6] = GRIP_OPEN
    for _ in range(300):
        mujoco.mj_step(model, data)


def run_pick(model, data, steps_per_phase: int = 250, record_every: int = 1) -> Episode:
    """Run the scripted pick.

    ``record_every`` decouples the control rate from the physics rate: one recorded
    step means ``record_every`` simulation steps. With dt=2 ms, ``record_every=25``
    gives a 20 Hz control signal, so a 30-step chunk spans ~1.5 s like Bridge.
    """
    import mujoco

    reset(model, data)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube = data.xpos[cid].copy()
    q = data.qpos[0:6].copy()

    hover = np.array([cube[0], cube[1], cube[2] + 0.22])
    grasp = np.array([cube[0], cube[1], cube[2] + 0.056])  # 1 cm deeper than the fingertip offset
    # lift diagonally (not vertically) so the ascent never retraces the descent
    lift = np.array([cube[0] + 0.01, cube[1] + 0.07, cube[2] + 0.18])
    # carry the cube sideways after lifting: without this the trajectory revisits the
    # same poses on the way down and up, so a single-frame policy cannot tell whether
    # it is approaching or retreating and settles into a down-up cycle
    carry = np.array([cube[0] + 0.02, cube[1] + 0.14, cube[2] + 0.26])

    q_hover = ik(model, data, hover, q)
    q_grasp = ik(model, data, grasp, q_hover)
    q_lift = ik(model, data, lift, q_grasp)
    q_carry = ik(model, data, carry, q_lift)

    # IK leaves the simulation sitting at the last solved pose; without this reset the
    # episode would start at the carry pose and every demo would be a closed loop
    reset(model, data)
    cube = data.xpos[cid].copy()

    episode = Episode(success=False, cube_z=float(cube[2]))

    def drive(joints: np.ndarray, grip: float, n: int) -> None:
        data.ctrl[0:6] = joints
        data.ctrl[6] = grip
        gid = _gid(model)
        for step in range(n):
            mujoco.mj_step(model, data)
            if step % record_every != 0:
                continue
            episode.proprio.append(np.concatenate([data.qpos[0:6].copy(), [grip]]))
            episode.states.append({
                "qpos": data.qpos.copy(),
                "ee": data.xpos[gid].copy(),
                "ee_rot": data.xmat[gid].reshape(3, 3).copy(),
                "grip": float(grip),  # commanded, not the blocked finger position
            })

    drive(q_hover, GRIP_OPEN, steps_per_phase)
    drive(q_grasp, GRIP_OPEN, steps_per_phase)
    drive(q_grasp, GRIP_CLOSED, steps_per_phase)
    drive(q_lift, GRIP_CLOSED, steps_per_phase)
    drive(q_carry, GRIP_CLOSED, steps_per_phase)

    episode.cube_z = float(data.qpos[10])
    episode.success = episode.cube_z > LIFT_OK
    return episode





