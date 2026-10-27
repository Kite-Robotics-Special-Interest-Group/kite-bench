# 官方环境复现（SimplerEnv）— 现场状态

停机会话前的进度，下次从这里继续。

## 目标

用 X-VLA 官方评测口径（SimplerEnv / ManiSkill2_real2sim）拿到 WidowX 任务的基线成功率，
用来回答"源权重到底行不行"，并给后续微调提供对照。

## 已完成

| 项 | 状态 |
| --- | --- |
| WSL Ubuntu | 26.04，GPU 直通正常（`nvidia-smi` 可见 RTX 5060） |
| Python 3.10 | `uv python install 3.10` → 3.10.22，venv 在 `~/official/venv` |
| X-VLA 仓库 | `~/official/X-VLA`，已读通 `evaluation/simpler/README.md` |
| SimplerEnv 主仓库 | `D:\JLU_tools\official\SimplerEnv`（Windows 侧，走 `ssh.github.com:443`） |
| 子模块对象 | `ManiSkill2_real2sim` 已下 44 MB 对象，**checkout 为空** |

## 卡点（都是网络，不是技术）

1. **子模块 checkout 失败**：反复报 `Fetched ... but did not contain 54ae2e0...`（shallow 抓不到该 commit）
   和 `shallow.lock` 冲突。解决思路：先 `rm -rf .git/modules/ManiSkill2_real2sim` 与工作目录，
   再做**非 shallow** 的 `git submodule update --init`；或在有稳定代理时一次完成。
2. **WSL 内 pip 下载不稳定**：pytorch.org 直连会中途断；阿里云 `pytorch-wheels/cu121` 可用，
   但需要多次重试（uv 先下到 cache 再安装，中途超时会导致 venv 仍是空的）。

## 下次的最小步骤

```bash
# 1) WSL 装 torch（可重复执行，断了就重跑）
cd ~/official && source venv/bin/activate
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cu121/ \
  'torch==2.5.1+cu121' 'torchvision==0.20.1+cu121'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2) 子模块（Windows 侧，确保 git 走 ssh.github.com:443）
cd D:/JLU_tools/official/SimplerEnv
rm -rf .git/modules/ManiSkill2_real2sim ManiSkill2_real2sim
git -c url."ssh://git@ssh.github.com:443/".insteadOf="https://github.com/" \
  submodule update --init          # 不要加 --depth

# 3) 安装（WSL 里，路径指向 /mnt/d）
pip install -e /mnt/d/JLU_tools/official/SimplerEnv/ManiSkill2_real2sim
pip install -e /mnt/d/JLU_tools/official/SimplerEnv
```

## 备选：Webots（第三环境，不是官方口径）

`D:\ProgramFiles\Webots` 是 **R2025a**，自带 43 个机器人（含 `franka_emika`、`universal_robots`、`abb`、`kuka`），
没有 WidowX。要接 WidowX 需 `pip install urdf2webots`，把已有的 Interbotix URDF + STL 转成 PROTO。

用途：作为"换引擎"的鲁棒性横轴（比 Unity 轻），**不能替代**官方 SimplerEnv 基线。

## 2026-10-27 更新：环境装好了，卡在 SAPIEN 的 WSL 渲染

已完成（WSL Ubuntu，Python 3.10）：

| 项 | 状态 |
| --- | --- |
| torch 2.9.1+cu128 | ✅ `cc (12, 0)`，sm_120 可用（2.5.1+cu121 不支持 Blackwell，已升级） |
| ManiSkill2_real2sim @54ae2e0 | ✅ 646 个文件 / 221.6 MB，用 `raw.githubusercontent.com` 逐文件下载（codeload 只有 20 KB/s） |
| simpler_env | ✅ 已装（含 sapien 2.2.2） |
| 评测服务端 | ✅ `deploy.py` 能加载官方权重并在 :8010 提供服务 |

**硬阻塞**：SAPIEN 的 Vulkan 渲染器在 WSL 里无法创建设备：

```
sapien.core.SapienRenderer() ->
RuntimeError: vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent
```

- sapien 的 `__init__` 明确警告 `SAPIEN renderer is not supported on WSL`
- sapien 2.2.2 只有 manylinux 轮子，**Windows 装不了**（PyPI 只列出 cp37–cp311 linux）
- 所以 SimplerEnv 在本机（Windows + WSL）都无法渲染观测

**要跑通只有三条路**：

1. 一台独立 Linux + NVIDIA 机器（或云 GPU），把 `~/official` 整个搬过去（依赖已就绪）
2. 双系统 / 原生 Linux 分区
3. 放弃官方口径，用 MuJoCo（已有 8/8）与 Unity（演示）代替
