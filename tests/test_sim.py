from __future__ import annotations

import importlib.util

import pytest

mujoco_missing = importlib.util.find_spec("mujoco") is None
pytestmark = pytest.mark.skipif(
    mujoco_missing, reason="mujoco not installed (pip install -e .[sim])"
)


def test_scene_builds_and_cube_settles() -> None:
    import mujoco

    from kitebench.sim.scene import CUBE_XY, build

    model = build()
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    assert abs(data.xpos[cid][0] - CUBE_XY[0]) < 0.01
    assert abs(data.xpos[cid][1] - CUBE_XY[1]) < 0.01


def test_oracle_picks_the_cube() -> None:
    import mujoco

    from kitebench.sim.oracle import run_pick
    from kitebench.sim.scene import build

    model = build()
    data = mujoco.MjData(model)
    episode = run_pick(model, data, steps_per_phase=200)
    assert episode.success, f"cube only reached z={episode.cube_z:.3f}"


def test_observation_shapes() -> None:
    import mujoco

    from kitebench.sim.oracle import reset
    from kitebench.sim.scene import build, observe

    model = build()
    data = mujoco.MjData(model)
    reset(model, data)
    obs = observe(model, data)
    assert obs["front"].size == (224, 224)
    assert obs["wrist"].size == (224, 224)
    assert obs["ee"].shape == (3,)
    assert obs["ee_rot"].shape == (3, 3)

