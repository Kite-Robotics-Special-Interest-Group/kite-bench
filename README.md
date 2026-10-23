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
docs/
  design.md      # 设计与指标口径
  roadmap.md     # 里程碑
tests/
```

## 快速开始

```bash
pip install -e .[dev]
python -m pytest -q
```

```python
from PIL import Image
from kitebench import perturb, metrics

img = Image.open("frame.png")
for name, fn in perturb.registry().items():
    out = fn(img, seed=0)
    print(name, out.size)
```

## 与 Kite 的关系

- `Kite`：端侧 VLA 模型与运行时（机器人本体）
- `kite-bench`：评测台，**不包含模型权重**，也不依赖厂商工具链
- 两者通过"观测 → 动作"的接口解耦：任何能吐 30×20 动作块的策略都能接进来

## 许可

MIT，见 [`LICENSE`](LICENSE)。
