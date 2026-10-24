"""MuJoCo workspace for the WidowX-250 pick task.

ROS/Bridge frame convention (x forward, y left, z up) so the rendered observations
match the frame the VLA policies were trained in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from kitebench.sim.assets import ensure_model

CUBE_XY = (0.26, 0.02)
CUBE_SIZE = 0.02
TABLE_TOP = 0.0
IMAGE_SIZE = 224

FRONT_POS = (-0.55, -0.62, 0.62)
FRONT_TARGET = (0.13, 0.05, 0.08)

_TEMPLATE = """<mujoco model="kite_wx250s">
  <include file="wx250s.xml"/>

  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="640" offheight="480"/>
  </visual>

  <asset>
    <material name="mat" rgba="{mat_rgba}"/>
    <material name="cube" rgba="{cube_rgba}"/>
    <material name="table_side" rgba="{table_rgba}"/>
  </asset>

  <worldbody>
    <light pos="0.2 0 1.4" dir="0 0 -1" directional="true" diffuse="{light}"/>
    <geom name="table" type="box" pos="0 0 -0.03" size="0.45 0.35 0.03" material="table_side"/>
    <geom name="mat" type="box" pos="0 0 -0.003" size="0.40 0.30 0.004" material="mat"/>

    <body name="cube" pos="{cx} {cy} {cz}">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="{size} {size} {size}" material="cube"
            mass="0.02" friction="1.2 0.02 0.002"/>
    </body>

    <camera name="front" pos="{fx} {fy} {fz}" xyaxes="{xyaxes}" fovy="55"/>
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


DEFAULT_VARIANT = {
    "mat_rgba": "0.38 0.52 0.47 1",
    "cube_rgba": "0.78 0.16 0.14 1",
    "table_rgba": "0.62 0.62 0.60 1",
    "light": "0.9 0.9 0.9",
    "cam_offset": (0.0, 0.0, 0.0),
}


def scene_xml(cube_xy: tuple[float, float] = CUBE_XY, variant: dict | None = None) -> str:
    v = {**DEFAULT_VARIANT, **(variant or {})}
    cam = (FRONT_POS[0] + v["cam_offset"][0],
           FRONT_POS[1] + v["cam_offset"][1],
           FRONT_POS[2] + v["cam_offset"][2])
    return _TEMPLATE.format(
        cx=cube_xy[0],
        cy=cube_xy[1],
        cz=CUBE_SIZE + 0.002,
        size=CUBE_SIZE,
        fx=cam[0],
        fy=cam[1],
        fz=cam[2],
        xyaxes=xyaxes(cam, FRONT_TARGET),
        mat_rgba=v["mat_rgba"],
        cube_rgba=v["cube_rgba"],
        table_rgba=v["table_rgba"],
        light=v["light"],
    )


def build(cube_xy: tuple[float, float] = CUBE_XY, variant: dict | None = None,
          model_dir: Path | None = None):
    """Instantiate the scene. Needs ``mujoco`` (``pip install -e .[sim]``)."""
    import mujoco

    model_dir = Path(model_dir) if model_dir else ensure_model()
    composed = model_dir / "_kite_scene.xml"
    composed.write_text(scene_xml(cube_xy, variant), encoding="utf-8")
    return mujoco.MjModel.from_xml_path(str(composed))


def wrist_camera(model, data):
    """Free camera following the gripper: behind/above, looking forward-down."""
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wx250s/gripper_link")
    pos = data.xpos[bid].copy()
    fwd = data.xmat[bid].reshape(3, 3)[:, 0]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = pos + fwd * 0.10 - np.array([0.0, 0.0, 0.04])
    cam.distance = 0.30
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    cam.elevation = -42.0
    return cam


def render(model, data, camera, size: int = IMAGE_SIZE) -> Image.Image:
    import mujoco

    renderer = mujoco.Renderer(model, size, size)
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    renderer.close()
    return Image.fromarray(pixels)


def render_with(renderer, data, camera) -> Image.Image:
    """Reuse an existing renderer (much faster when capturing many frames)."""
    renderer.update_scene(data, camera=camera)
    return Image.fromarray(renderer.render())


def observe(model, data, size: int = IMAGE_SIZE) -> dict:
    """Front + wrist frames plus the end-effector pose and gripper state."""
    import mujoco

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wx250s/gripper_link")
    return {
        "front": render(model, data, "front", size),
        "wrist": render(model, data, wrist_camera(model, data), size),
        "ee": data.xpos[gid].copy(),
        "ee_rot": data.xmat[gid].reshape(3, 3).copy(),
        "grip": float(data.qpos[6]),
    }
