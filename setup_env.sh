#!/bin/bash
# ============================================================
# Isaac RM 训练环境一键安装脚本
# 依赖：Isaac Lab 2.3.2 + Isaac Sim 5.1.0 + skrl 1.4.x + Python 3.11
# 使用方式：bash setup_env.sh
# ============================================================

set -e

ENV_NAME="isaac_rm"
PYTHON_VER="3.11"
ISAACLAB_DIR="$HOME/IsaacLab"

echo "===== [1/5] 创建 conda 环境：$ENV_NAME (Python $PYTHON_VER) ====="
conda create -n "$ENV_NAME" python="$PYTHON_VER" -y

# 激活 conda 环境（兼容不同 shell）
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "===== [2/5] 安装 Isaac Sim 5.1.0（NVIDIA PyPI，约 5~10 GB，请耐心等待）====="
OMNI_KIT_ACCEPT_EULA=YES pip install 'isaacsim[all]==5.1.0.0' \
    --extra-index-url https://pypi.nvidia.com

echo "===== [3/5] 克隆 IsaacLab 并切换到 v2.3.2 ====="
if [ -d "$ISAACLAB_DIR" ]; then
    echo "  检测到 $ISAACLAB_DIR 已存在，跳过 clone，直接 checkout tag"
    cd "$ISAACLAB_DIR" && git fetch --tags && git checkout v2.3.2
else
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
    cd "$ISAACLAB_DIR" && git checkout v2.3.2
fi

echo "===== [4/5] 安装 IsaacLab 扩展（skrl 后端）====="
# isaaclab.sh --install 会把 isaaclab_rl / isaaclab_tasks 等包装到当前 conda 环境
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh --install skrl

echo "===== [4.5/5] 钉定 skrl 版本（v2.3.2 与 skrl>=2 不兼容）====="
pip install 'skrl>=1.4.3,<2'

# 若 flatdict 因 uv 构建隔离报 pkg_resources 缺失，取消下一行注释
# pip install --no-build-isolation flatdict==4.0.1

echo "===== [5/5] 验证关键包版本 ====="
python - <<'EOF'
import isaacsim; print(f"isaacsim  : {isaacsim.__version__}")
import isaaclab; print(f"isaaclab  : {isaaclab.__version__}")
import skrl;     print(f"skrl      : {skrl.__version__}")
import torch;    print(f"torch     : {torch.__version__}, CUDA: {torch.cuda.is_available()}")
EOF

echo ""
echo "===== 安装完成！====="
echo ""
echo "【URDF】URDF 和 mesh 已内置在项目 assets/ 目录，路径由 chassis.py 动态推导，无需额外配置。"
echo ""
echo "训练命令（无头模式）："
echo "  conda activate $ENV_NAME"
echo "  cd ~/isaac_rm"
echo "  OMNI_KIT_ACCEPT_EULA=YES python scripts/train.py \\"
echo "      --task Isaac-Chassis-Approach-v0 --num_envs 4096 --headless"
