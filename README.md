# kite-bench

> VLA 的仿真环境和输入鲁棒性评测台。物理用 MuJoCo，外观扰动可以接 Unity / Moer Engine。

一个 VLA 模型在训练分布里能抓住方块，不代表它在真实车间里还能抓住——换一盏灯、相机被碰歪 5°、目标被机械臂自己挡住，输出就可能完全跑偏。**kite-bench 就是为这件事准备的**：给同一批观测和同一批 episode 加上可控扰动，量化模型的动作漂移和成功率下降。

## 为什么需要它

- 现有 VLA 论文大多只报分布内成功率，扰动下的退化没有统一口径
- 真实部署的第一个问题不是"能不能做"，而是"输入变一点还能不能做"
- 消费级显卡（8 GB）也应该能跑完整评测

## 两层评测

| 层 | 扰动来源 | 工具 | 成本 |
| --- | --- | --- | --- |
| 输入层 | 光度、几何、遮挡、语言、时序 | 纯 Python（PIL/numpy） | 低，可跑上千组 |
| 外观层 | 光照、材质、背景、相机机位、遮挡物 | Unity / Moer Engine 重渲染 | 中，几十组 |
| 闭环层 | 上述扰动下的完整 episode | MuJoCo（物理正确、可批量） | 中，每类 20–50 episode |

动作漂移是便宜的代理指标，用来扫参数；**结论级指标是闭环成功率**。

## 指标

- `Δpos`：末端位置差的 L2（毫米）
- `Δrot`：姿态测地距离（度）
- `grip_flip`：夹爪开合判定翻转率
- `success@perturb`：扰动下 episode 成功率，与干净基线对比
- `robust_score = 1 - mean(Δpos / scale)`（归一化漂移，越大越稳）

公式与口径见 [`docs/design.md`](docs/design.md)。

## 目录

```
src/kitebench/
  perturb.py     # 光度/几何/遮挡扰动（纯 PIL，无仿真依赖）
  metrics.py     # 漂移与成功率指标
  eval.py        # 扰动扫描（任意策略适配）
  train.py       # 动作头微调（特征预计算 + 后 8 层）
  rollout.py     # MuJoCo 闭环评测（抓取）
  demo_ik.py     # 示教流形最近邻执行器（消除 IK 分支歧义）
  sim/
    assets.py    # 拉取/定位 MuJoCo WidowX-250 模型
    scene.py     # 场景、两路 224×224 相机、观测
    oracle.py    # 脚本化拾取（IK + 位置伺服）
    sorting.py   # 3 类物体分拣场景与 pick-and-place
    industrial.py# 工业外观（传送带/料筐/安全线）
unity/KiteWidowX/  # Unity 工程源码（场景/脚本/URDF），见 docs/unity-setup.md
docs/
  design.md            # 设计与指标口径
  roadmap.md           # 里程碑
  perturb-baseline.md  # 输入扰动基线表
  unity-setup.md       # Unity 安装 → 配置 → MCP → 使用
  mujoco-sim.md        # MuJoCo 场景与采集
  unity-sim.md         # Unity 桥协议与踩坑（历史版本）
  official-env-setup.md# 官方 SimplerEnv 复现现场
tests/
```

文档索引：

| 想了解 | 看 |
| --- | --- |
| 为什么做这个、指标怎么定义 | [`docs/design.md`](docs/design.md) |
| 现在做到哪、下一步 | [`docs/roadmap.md`](docs/roadmap.md) |
| 模型对输入扰动有多敏感 | [`docs/perturb-baseline.md`](docs/perturb-baseline.md) |
| Unity 怎么装、怎么跑 | [`docs/unity-setup.md`](docs/unity-setup.md) |
| MuJoCo 场景与数据采集 | [`docs/mujoco-sim.md`](docs/mujoco-sim.md) |
| 官方 SimplerEnv 复现进度 | [`docs/official-env-setup.md`](docs/official-env-setup.md) |

## 快速开始

```bash
pip install -e ".[sim,dev]"   # sim 额外装 mujoco
python -m pytest -q
```

MuJoCo 的 WidowX-250 模型不随仓库分发：首次构建场景时会从 `mujoco_menagerie` 稀疏克隆到 `~/.cache/kitebench/models`，也可以用 `KITEBENCH_WX250S` 指向已有目录。没有 mujoco 时仿真测试自动跳过，CI 只跑不依赖仿真的部分。

```python
from kitebench import perturb, metrics

img = ...  # 224x224 RGB PIL image
for name, fn in perturb.registry().items():
    print(name, fn(img, seed=0).size)
```

采集示教（带域随机化）：

```bash
python -m kitebench.sim.collect --episodes 20 --out episodes
# episodes/0000/{meta.json, action.npy, proprio.npy, obs/*_front.png, *_wrist.png}
```

每条 episode 随机化方块位置（±5 cm）、桌垫/方块颜色、光照强度与相机机位；动作标签是绝对末端位姿 `[ee_xyz(3), rot6d(6), grip(1)]`。

## 与 Kite 的关系

- `Kite`：端侧 VLA 模型与运行时（机器人本体）
- `kite-bench`：评测台，**不包含模型权重**，也不依赖厂商工具链
- 两者通过"观测 → 动作"的接口解耦：任何能吐 30×20 动作块的策略都能接进来

## 许可

MIT，见 [`LICENSE`](LICENSE)。
