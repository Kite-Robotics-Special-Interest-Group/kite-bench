# 设计

## 1. 目标与非目标

**目标**

1. 给定一个策略（任何把观测映射到动作块的模型），能量化它在输入扰动下的退化
2. 扰动可复现：固定 seed、固定参数，同一份配置在任何机器上跑出同一组图
3. 便宜：输入层扰动不依赖 GPU、不依赖仿真器
4. 结论可追溯到具体扰动族和幅度

**非目标**

- 不训练/不微调模型
- 不追求物理真实到可以替代真机（那是 MuJoCo/Isaac 的事）
- 不包含任何模型权重

## 2. 评测单元

鲁棒性评测的最小单元是 **episode**，不是单帧：

```
for each initial_state in validation_set:
    for each perturb_family in families:
        run_policy(initial_state, perturb_family) -> success, action_drift_trace
```

但 episode 很贵（一次闭环要几十次推理），所以日常用**单帧代理**扫参数：

```
obs_clean -> policy -> a_clean
obs_perturbed(obs_clean, p) -> policy -> a_p
drift = distance(a_clean, a_p)
```

代理指标只用于筛扰动强度；最终报告必须给闭环成功率。

## 3. 扰动族

| 族 | 参数 | 实现 |
| --- | --- | --- |
| `brightness` | ±40% | 亮度缩放 |
| `contrast` | 0.6–1.4 | 对比度缩放 |
| `gamma` | 0.5–2.0 | 幂律变换 |
| `noise` | σ = 5–25 | 高斯噪声 |
| `blur` | r = 1–3 px | 高斯模糊 |
| `jpeg` | q = 20–80 | JPEG 重压缩 |
| `translate` | ±10% 视场 | 平移并补边 |
| `scale` | 0.9–1.1 | 中心缩放 |
| `rotate` | ±5° | 旋转并补边 |
| `downsample` | 112/168 → 224 | 降采样再放大 |
| `occlude_rect` | 面积 5–25% | 随机矩形遮挡 |
| `occlude_arm` | 固定区域 | 模拟机械臂挡住镜头 |
| `delay` | 1–3 帧 | 时序错位（需要 episode） |
| `paraphrase` | 人工列表 | 指令改写（需要模型） |

约定：所有图像扰动作用在 **224×224 RGB** 上，输出同尺寸、同 dtype；随机扰动必须接受 `seed`。

## 4. 指标口径

设干净动作块 `a`、扰动动作块 `a'`，形状 `[T=30, D=20]`，其中 `xyz = a[:, 0:3]`、`rot6d = a[:, 3:9]`、`grip = a[:, 9]`。

- **Δpos**：`mean_t ||a'[:, 0:3] - a[:, 0:3]||₂ × 1000`（毫米）
- **Δrot**：把 6D 转成旋转矩阵，取 `angle(R'ᵀR)` 的均值（度）
- **grip_flip**：`mean_t [ (a'[t,9] >= 0.91) != (a[t,9] >= 0.91) ]`
- **success@perturb**：episode 末尾 `cube_z > 0.06` 的比例
- **robust_score**：`1 - mean(Δpos) / 50`，50 mm 作为参考尺度，仅用于横向比较

注意：Δpos 对绝对位置动作是"目标点漂移"，对增量动作是"位移漂移"，两者不可混报——报告里必须写明动作空间。

## 5. 数据流

**控制频率是数据格式的一部分。** MuJoCo 的 `dt = 2 ms`，如果按物理步记录动作，一个 30 步动作块只覆盖 0.06 s，模型会学成"下一帧几乎不动"。所以采集时用 `record_every=25`（20 Hz 控制），30 步对应约 1.5 s，与 Bridge 量级一致；闭环回放时每个动作也走 25 个物理步。

```
MuJoCo/MoJoCo+Unity ──► episodes (obs, proprio, action, success)
        │
        ├─► 单帧集（obs）──► perturb ──► policy ──► metrics ──► 表
        │
        └─► 闭环回放（Unity）──► 成功率
```

数据集格式（第一阶段）：

```
episodes/
  0000/
    meta.json        # instruction, initial_state, success, seed
    obs/0000.png ... # front 224x224
    proprio.npy      # [T, 20]
    action.npy       # [T, 30, 20]
```

## 6. 与 Kite 主仓的边界

- `kite-bench` 只管"评测"，不引入 X-VLA 的建模代码；模型通过适配器接入
- 不 vendor 权重和厂商 SDK；MuJoCo 模型从 `mujoco_menagerie` 拉取
- 评测结果是可公开的数字，不包含内部路径、板端参数或客户信息
