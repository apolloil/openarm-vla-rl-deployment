# Conda 安装 Isaac Lab + OpenArm

## 前提条件

- Ubuntu 22.04 / 24.04（或满足 Isaac Sim pip 包对 glibc 的要求）
- 与 Isaac Sim 版本匹配的 NVIDIA 驱动与 GPU
- 空闲磁盘足够（Isaac Sim 与缓存常需数十 GB；pip 缓存建议放在大盘）
- 已安装：`cmake`、`build-essential`
- 已安装：Miniforge3 或 Miniconda（可用 mamba）

## 实施步骤

以下路径请按本机修改（示例：`/path/to/miniforge3`、`/path/to/large_disk`、`/path/to/IsaacLab`、`/path/to/openarm_isaac_lab`）。

**1. 克隆 Isaac Lab（与 README 一致的分支）**

```bash
git clone --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git /path/to/IsaacLab
```

**2. 新建 Conda 环境并设置 pip 缓存**

```bash
source /path/to/miniforge3/etc/profile.d/conda.sh
mamba create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
mkdir -p /path/to/large_disk/pip_cache
export PIP_CACHE_DIR=/path/to/large_disk/pip_cache
```

**3. 安装 Isaac Lab 依赖（在 Isaac Lab 仓库根目录）**

```bash
export TERM=xterm
cd /path/to/IsaacLab
./isaaclab.sh --install
```

若其中从 GitHub 安装 `rl-games` 失败，可单独执行：

```bash
pip install "git+https://github.com/isaac-sim/rl_games.git@python3.11"
```

**4. 安装 Isaac Sim（耗时长）**

```bash
conda activate env_isaaclab
export PIP_CACHE_DIR=/path/to/large_disk/pip_cache
pip install "isaacsim[all,extscache]==5.1.0.0" --extra-index-url https://pypi.nvidia.com
```

**5. 安装 Isaac Lab 核心包 `isaaclab`**

若 `pip install -e .../source/isaaclab` 时 `flatdict` 报缺少 `pkg_resources`，先执行 `pip install "setuptools<80"`，再重试：

```bash
pip install -e /path/to/IsaacLab/source/isaaclab --no-build-isolation
```

**6. 与 Isaac Sim 冲突时的版本固定（按需）**

装完 `rl_games` 后若 `packaging` 被拉高，可执行：`pip install packaging==23.0 --no-deps`。若 `click` 被拉高，可执行：`pip install "click==8.1.7" --no-deps`。

**7. 安装本仓库 OpenArm 扩展**

```bash
conda activate env_isaaclab
cd /path/to/openarm_isaac_lab
pip install -e source/openarm --no-build-isolation
```

**8. 运行前环境变量与验证**

```bash
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=Y
cd /path/to/openarm_isaac_lab
python -u ./scripts/tools/list_envs.py --headless
```

`OMNI_KIT_ACCEPT_EULA=Y` 用于无图形界面时非交互接受 Omniverse/Isaac Sim 的许可。
