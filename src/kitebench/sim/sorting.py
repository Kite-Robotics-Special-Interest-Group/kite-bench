"""Multi-object sorting scene and scripted pick-and-place oracle.

Three object classes (red cube, blue cylinder, green sphere) sit on the table with
two target pads. The instruction names both the object and the pad, e.g.
``put the red cube on the left pad``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kitebench.sim.assets import ensure_model

IMAGE_SIZE = 224
PAD_LEFT = (-0.02, 0.22)
PAD_RIGHT = (0.30, -0.16)
PAD_RADIUS = 0.055
GRIP_OPEN = 0.037
GRIP_CLOSED = 0.016
INSTRUCTIONS = {
    "red cube": "put the red cube on the left pad",
    "blue cylinder": "put the blue cylinder on the right pad",
    "green box": "put the green box on the left pad",
}

OBJECTS = {
    "red cube": {"type": "box", "size": (0.02, 0.02, 0.02), "rgba": "0.80 0.16 0.14 1"},
    "blue cylinder": {"type": "cylinder", "size": (0.02, 0.02, 0.0), "rgba": "0.16 0.32 0.80 1"},
    "green box": {"type": "box", "size": (0.017, 0.017, 0.017), "rgba": "0.18 0.66 0.30 1"},
}

_TEMPLATE = """<mujoco model="kite_sorting">
  <include file="wx250s.xml"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="640" offheight="480"/>
  </visual>
  <asset>
    <material name="mat" rgba="0.38 0.52 0.47 1"/>
    <material name="table_side" rgba="0.62 0.62 0.60 1"/>
    <material name="pad_left" rgba="0.95 0.85 0.30 1"/>
    <material name="pad_right" rgba="0.35 0.75 0.95 1"/>
{object_materials}
  </asset>
  <worldbody>
    <light pos="0.2 0 1.4" dir="0 0 -1" directional="true" diffuse="0.9 0.9 0.9"/>
    <geom name="table" type="box" pos="0 0 -0.03" size="0.45 0.35 0.03" material="table_side"/>
    <geom name="mat" type="box" pos="0 0 -0.003" size="0.40 0.30 0.004" material="mat"/>
    <geom name="pad_left" type="box" pos="{plx} {ply} 0.001"
          size="{pr} {pr} 0.001" material="pad_left"/>
    <geom name="pad_right" type="box" pos="{prx} {pry} 0.001"
          size="{pr} {pr} 0.001" material="pad_right"/>
{objects}
    <camera name="front" pos="-0.55 -0.62 0.62" xyaxes="{xyaxes}" fovy="55"/>
  </worldbody>
</mujoco>
"""


def xyaxes(pos, target) -> str:
    z = np.array(pos, dtype=float) - np.array(target, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return " ".join(f"{v:.4f}" for v in np.concatenate([x, y]))


def scene_xml(positions: dict[str, tuple[float, float]]) -> str:
    mats, objs = [], []
    for i, (name, spec) in enumerate(OBJECTS.items()):
        mats.append(f'    <material name="m{i}" rgba="{spec["rgba"]}"/>')
        x, y = positions[name]
        size = spec["size"]
        if spec["type"] == "box":
            geom = (
                f'<geom name="obj{i}" type="box" size="{size[0]} {size[1]} {size[2]}"'
            )
            z = size[2] + 0.002
        elif spec["type"] == "cylinder":
            geom = (
                f'<geom name="obj{i}" type="cylinder" size="{size[0]} {size[1]}"'
            )
            z = size[1] + 0.002
        else:
            geom = f'<geom name="obj{i}" type="sphere" size="{size[0]}"'
            z = size[0] + 0.002
        objs.append(
            f'    <body name="body{i}" pos="{x:.3f} {y:.3f} {z:.3f}">\n'
            f'      <freejoint name="free{i}"/>\n'
            f'      {geom} material="m{i}" mass="0.02" friction="1.2 0.02 0.002"/>\n'
            f"    </body>"
        )
    return _TEMPLATE.format(
        object_materials="\n".join(mats),
        objects="\n".join(objs),
        plx=PAD_LEFT[0], ply=PAD_LEFT[1],
        prx=PAD_RIGHT[0], pry=PAD_RIGHT[1],
        pr=PAD_RADIUS,
        xyaxes=xyaxes((-0.55, -0.62, 0.62), (0.13, 0.05, 0.08)),
    )


def build(positions: dict[str, tuple[float, float]], model_dir: Path | None = None):
    import mujoco

    model_dir = Path(model_dir) if model_dir else ensure_model()
    composed = model_dir / "_kite_sorting.xml"
    composed.write_text(scene_xml(positions), encoding="utf-8")
    return mujoco.MjModel.from_xml_path(str(composed))


def sample_positions(rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    """Three objects spread over the workspace, kept apart from each other."""
    out: dict[str, tuple[float, float]] = {}
    while len(out) < len(OBJECTS):
        name = list(OBJECTS)[len(out)]
        candidate = (float(rng.uniform(0.17, 0.34)), float(rng.uniform(-0.10, 0.12)))
        if all(np.hypot(candidate[0] - v[0], candidate[1] - v[1]) > 0.07 for v in out.values()):
            out[name] = candidate
    return out


@dataclass
class PlaceEpisode:
    success: bool
    distance: float
    states: list[dict] = field(default_factory=list)


def _body_id(model, name: str) -> int:
    import mujoco

    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def run_place(model, data, instruction: str, record_every: int = 25,
              steps_per_phase: int = 450) -> PlaceEpisode:
    import mujoco

    from kitebench.sim.oracle import GRIP_CLOSED as GC
    from kitebench.sim.oracle import GRIP_OPEN as GO
    from kitebench.sim.oracle import ik, reset

    reset(model, data)
    names = list(OBJECTS)
    target_name = next((n for n in names if n in instruction), names[0])
    target_idx = names.index(target_name)
    pad = PAD_LEFT if "left" in instruction else PAD_RIGHT
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wx250s/gripper_link")
    obj = data.xpos[_body_id(model, f"body{target_idx}")].copy()

    hover = np.array([obj[0], obj[1], obj[2] + 0.20])
    grasp = np.array([obj[0], obj[1], obj[2] + 0.050])
    lift = np.array([obj[0] + 0.01, obj[1] + 0.05, obj[2] + 0.18])
    above_pad = np.array([pad[0], pad[1], obj[2] + 0.16])
    place = np.array([pad[0], pad[1], obj[2] + 0.030])
    retreat = np.array([pad[0] - 0.02, pad[1] + 0.04, obj[2] + 0.20])

    q0 = data.qpos[0:6].copy()
    q_hover = ik(model, data, hover, q0)
    q_grasp = ik(model, data, grasp, q_hover)
    q_lift = ik(model, data, lift, q_grasp)
    q_above = ik(model, data, above_pad, q_lift)
    q_place = ik(model, data, place, q_above)
    q_retreat = ik(model, data, retreat, q_place)

    reset(model, data)
    episode = PlaceEpisode(success=False, distance=1.0)

    def drive(joints, grip, n):
        data.ctrl[0:6] = joints
        data.ctrl[6] = grip
        for step in range(n):
            mujoco.mj_step(model, data)
            if step % record_every:
                continue
            episode.states.append({
                "qpos": data.qpos.copy(),
                "ee": data.xpos[gid].copy(),
                "ee_rot": data.xmat[gid].reshape(3, 3).copy(),
                "grip": float(grip),
            })

    drive(q_hover, GO, steps_per_phase)
    drive(q_grasp, GO, steps_per_phase)
    drive(q_grasp, GC, steps_per_phase)
    drive(q_lift, GC, steps_per_phase)
    drive(q_above, GC, steps_per_phase)
    drive(q_place, GC, steps_per_phase)
    drive(q_place, GO, steps_per_phase // 2)
    drive(q_retreat, GO, steps_per_phase // 2)

    final = data.xpos[_body_id(model, f"body{target_idx}")]
    episode.distance = float(np.linalg.norm(final[0:2] - np.array(pad)))
    episode.success = episode.distance < 0.07
    return episode



