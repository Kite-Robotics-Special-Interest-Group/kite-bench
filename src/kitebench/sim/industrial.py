"""Industrial dressing for the pick cell: conveyor, bin, striped floor.

The task geometry (cube position, camera, teal work mat) is unchanged so the
fine-tuned policy still sees what it was trained on; this only adds factory
context around it.
"""

from __future__ import annotations

from pathlib import Path

from kitebench.sim.assets import ensure_model
from kitebench.sim.scene import CUBE_SIZE, FRONT_POS, FRONT_TARGET, xyaxes

_TEMPLATE = """<mujoco model="kite_industrial">
  <include file="wx250s.xml"/>
  <visual>
    <headlight diffuse="0.65 0.65 0.65" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global offwidth="640" offheight="480"/>
  </visual>
  <asset>
    <material name="mat" rgba="0.38 0.52 0.47 1"/>
    <material name="table_side" rgba="0.45 0.46 0.48 1"/>
    <material name="cube" rgba="0.78 0.16 0.14 1"/>
    <material name="steel" rgba="0.55 0.57 0.60 1"/>
    <material name="dark" rgba="0.18 0.19 0.21 1"/>
    <material name="belt" rgba="0.12 0.13 0.14 1"/>
    <material name="bin" rgba="0.85 0.55 0.12 1"/>
    <material name="hazard" rgba="0.92 0.75 0.10 1"/>
  </asset>
  <worldbody>
    <light pos="0.2 0 1.4" dir="0 0 -1" directional="true" diffuse="0.9 0.9 0.9"/>

    <geom name="floor" type="plane" pos="0 0 -0.90" size="4 4 0.05" material="steel"/>
    <geom name="table" type="box" pos="0 0 -0.03" size="0.45 0.35 0.03" material="table_side"/>
    <geom name="mat" type="box" pos="0 0 -0.003" size="0.40 0.30 0.004" material="mat"/>

    <geom name="stripe_f" type="box" pos="0 -0.352 0.001"
          size="0.45 0.008 0.002" material="hazard"/>
    <geom name="stripe_b" type="box" pos="0 0.352 0.001"
          size="0.45 0.008 0.002" material="hazard"/>

    <geom name="conv_frame" type="box" pos="0.0 -0.30 0.02" size="0.34 0.05 0.02" material="dark"/>
    <geom name="conv_belt" type="box" pos="0.0 -0.30 0.045"
          size="0.32 0.045 0.006" material="belt"/>
    <geom name="roller0" type="cylinder" pos="-0.30 -0.30 0.045"
          size="0.03 0.045" euler="1.5708 0 0" material="steel"/>
    <geom name="roller1" type="cylinder" pos="0.30 -0.30 0.045"
          size="0.03 0.045" euler="1.5708 0 0" material="steel"/>

    <geom name="bin_floor" type="box" pos="-0.30 0.24 0.012" size="0.08 0.07 0.012" material="bin"/>
    <geom name="bin_x0" type="box" pos="-0.38 0.24 0.06" size="0.008 0.07 0.05" material="bin"/>
    <geom name="bin_x1" type="box" pos="-0.22 0.24 0.06" size="0.008 0.07 0.05" material="bin"/>
    <geom name="bin_y0" type="box" pos="-0.30 0.17 0.06" size="0.08 0.008 0.05" material="bin"/>
    <geom name="bin_y1" type="box" pos="-0.30 0.31 0.06" size="0.08 0.008 0.05" material="bin"/>

    <body name="cube" pos="{cx} {cy} {cz}">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="{size} {size} {size}" material="cube"
            mass="0.02" friction="1.2 0.02 0.002"/>
    </body>

    <camera name="front" pos="{fx} {fy} {fz}" xyaxes="{xyaxes}" fovy="55"/>
  </worldbody>
</mujoco>
"""


def build(cube_xy: tuple[float, float] = (0.26, 0.02), model_dir: Path | None = None):
    import mujoco

    model_dir = Path(model_dir) if model_dir else ensure_model()
    xml = _TEMPLATE.format(
        cx=cube_xy[0],
        cy=cube_xy[1],
        cz=CUBE_SIZE + 0.002,
        size=CUBE_SIZE,
        fx=FRONT_POS[0],
        fy=FRONT_POS[1],
        fz=FRONT_POS[2],
        xyaxes=xyaxes(FRONT_POS, FRONT_TARGET),
    )
    composed = model_dir / "_kite_industrial.xml"
    composed.write_text(xml, encoding="utf-8")
    return mujoco.MjModel.from_xml_path(str(composed))

