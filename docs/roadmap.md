# 路线图

每个里程碑都要有可验收的数字，没有数字不算完成。

## M0 · 场景与专家轨迹（已完成）

- MuJoCo 场景：WidowX-250 + 桌面 + 自由方块 + 两路 224×224 相机
- 脚本化专家：6-DoF IK，自上而下抓取，抬升 > 6 cm
- 验收：`success=True`，单条 ≤ 600 步
- 状态：`tests/test_sim.py::test_oracle_picks_the_cube` 已覆盖

## M1 · 数据采集（进行中）

- 随机化：方块位置 ±5 cm、桌垫/方块颜色、光照强度、相机机位 ±4 cm
- 采集 200–500 条，存 episode 格式（两路图像 + proprio + 动作 + success）
- 命令：`python -m kitebench.sim.collect --episodes 20 --out episodes`
- 验收：成功率 ≥ 90%，随机种子固定可复现
- 状态：采集器已完成，烟测 3/3 成功；正在跑批量统计

## M2 · 输入层扰动评测（不依赖 GPU）

- 实现 §3 的 12 类图像扰动，全部带 seed
- 对单帧集跑"干净 vs 扰动"的动作漂移，输出 md/csv 表
- 验收：每类扰动 ≥ 5 档，表格含 Δpos / Δrot / grip_flip

## M3 · 轻量微调

- 冻结主干，只训练动作头 + soft prompt
- 输入：M1 的数据集；输出：可加载的 checkpoint
- 验收：闭环成功率相对原始权重提升 ≥ 20 个百分点

## M4 · 闭环鲁棒性评测

- 同一批初始状态，干净 vs 扰动各跑 N 次
- Unity 侧加 `PERTURB` 命令（光照/材质/机位/遮挡物）
- 验收：给出"扰动族 × 成功率"矩阵，并附动作漂移相关性

## 未来

- 时序扰动（延迟/丢帧）与语言改写
- 残差动作 + PPO 精修（MuJoCo 批量环境）
- Moer Engine 离线渲染，生成对外演示视频
