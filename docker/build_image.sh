#!/usr/bin/env bash
# ============================================================
# 在【本机】(有 sudo + GPU) 构建 isaac_rm 训练镜像，并导出 tar.gz 供平台上传。
# 用法：sudo bash docker/build_image.sh
# 说明：构建上下文放在 isaac_rm 【外面】(/home/shihao/.isaac_rm_build_ctx)，
#       否则 cp -al 把 isaac_rm 拷进它自己的子目录会报"复制到自身内"。
# ============================================================
set -euo pipefail

IMAGE="isaac-rm:5.1"
HERE="$(cd "$(dirname "$0")" && pwd)"          # .../isaac_rm/docker
CTX="/home/shihao/.isaac_rm_build_ctx"          # 与 /home/shihao/* 同一文件系统，硬链接可用
HS="$CTX/home/shihao"                            # 镜像内 /home/shihao 的布局

echo "===== [1/4] 组装构建上下文（cp -al 硬链接：瞬间完成、不额外占盘）====="
rm -rf "$CTX" /home/shihao/isaac_rm/docker/_ctx   # 清理本次与历史残留
mkdir -p "$HS/.local/share/uv/python" \
         "$HS/pistar" \
         "$HS/XRTeleop/assets/robot_discription/overseas_65_b_v_description"

# uv 自带 python（venv 的 bin/python 软链指向它）
cp -al /home/shihao/.local/share/uv/python/cpython-3.11.9-linux-x86_64-gnu \
       "$HS/.local/share/uv/python/"
# venv 本体（~14G，含 isaacsim / torch / skrl）
cp -al /home/shihao/pistar/.venv      "$HS/pistar/.venv"
# openpi editable 源码（venv 的 .pth 会引用，带上以免 import 时报路径缺失）
cp -al /home/shihao/pistar/src        "$HS/pistar/src"
cp -al /home/shihao/pistar/packages   "$HS/pistar/packages"
# isaaclab editable 源码（venv 的 __editable__ finder 指向这里）
cp -al /home/shihao/IsaacLab          "$HS/IsaacLab"
# 项目本体（代码 + assets + URDF）
cp -al /home/shihao/isaac_rm          "$HS/isaac_rm"
# URDF 里写死的 XRTeleop 网格（绝对路径必须存在，否则机器人网格解析失败）
cp -al /home/shihao/XRTeleop/assets/robot_discription/overseas_65_b_v_description/meshes \
       "$HS/XRTeleop/assets/robot_discription/overseas_65_b_v_description/meshes"

# 排除进镜像的体积/垃圾（.dockerignore 必须在上下文根下）
cat > "$CTX/.dockerignore" <<'EOF'
**/.git
**/__pycache__
**/*.pyc
home/shihao/isaac_rm/wandb
home/shihao/isaac_rm/logs
home/shihao/isaac_rm/outputs
home/shihao/isaac_rm/docker/_ctx
home/shihao/isaac_rm/docker/isaac-rm-*.tar.gz
EOF

echo "===== [2/4] docker build（首次较慢，需读完 ~14G 上下文）====="
sudo docker build -f "$HERE/Dockerfile" -t "$IMAGE" "$CTX"

echo "===== [3/4] 清理构建上下文硬链接 ====="
rm -rf "$CTX"

echo "===== [4/4] 导出镜像为 tar.gz（用于平台「镜像包」上传）====="
sudo docker save "$IMAGE" | gzip > "$HERE/isaac-rm-5.1.tar.gz"
# 用 sudo 跑时把产物归还给原用户（否则 tar.gz 属 root，浏览器上传/移动不便）
OWNER="${SUDO_USER:-$(id -un)}"
sudo chown "$OWNER:$OWNER" "$HERE/isaac-rm-5.1.tar.gz"
ls -lh "$HERE/isaac-rm-5.1.tar.gz"
echo ""
echo "完成 ✅  产物：$HERE/isaac-rm-5.1.tar.gz"
echo "（若要推到镜像仓库改用 docker tag + docker push，见 docker/README.md）"
