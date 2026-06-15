# syntax=docker/dockerfile:1
# ============================================================
# Isaac RM 训练镜像
# 思路：不在容器里重装 Isaac Sim，而是把本机【已验证可用】的 uv venv
#       连同它依赖的几棵目录，按【相同绝对路径】原样搬进镜像。
#       venv 的 python 软链、isaaclab 的 editable 源码、URDF 写死的
#       XRTeleop 网格路径都依赖绝对路径，所以必须落在 /home/shihao/... 下。
# 基础镜像：CUDA 运行时；GPU 驱动 / Vulkan ICD 由平台的 NVIDIA 容器运行时注入。
# ============================================================
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Isaac Sim / Omniverse Kit 无头运行所需的系统库（GL / Vulkan / X / 字体等）
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        libgl1 libglu1-mesa libegl1 libgles2 libglvnd0 libglx0 \
        libgomp1 libatomic1 libglib2.0-0 libdbus-1-3 libgssapi-krb5-2 \
        libsm6 libice6 libx11-6 libxext6 libxrender1 libxrandr2 libxi6 \
        libxcursor1 libxinerama1 libxkbcommon0 libxkbcommon-x11-0 \
        libxcb1 libxcb-glx0 libxcb-keysyms1 libxcb-image0 libxcb-shm0 \
        libxcb-icccm4 libxcb-sync1 libxcb-xfixes0 libxcb-shape0 \
        libxcb-randr0 libxcb-render-util0 libxcb-util1 libxcb-xinerama0 libxcb-cursor0 \
        libfontconfig1 libfreetype6 libnss3 libnspr4 libasound2 libpulse0 \
        libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# 让平台的 NVIDIA 运行时把 GPU + 图形能力（含 Vulkan ICD，PhysX-GPU/渲染需要）注入容器
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# 一次性把组装好的目录树搬到镜像 /home（构建上下文根下已镜像 home/shihao/... 布局）
# --chmod=0777：平台容器以非 root 的随机用户(如 liupuze)运行，文件属主是构建时的
#   uid 1000，否则 Isaac Sim 写不了它的扩展缓存/日志(PermissionError)。改 777 让任意用户可写，
#   且在 COPY 阶段设权限【不会增加镜像体积】(不像 RUN chmod -R 会复制整层)。
COPY --chmod=0777 home /home

# 与开发机保持一致的运行环境
ENV HOME=/home/shihao
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV PATH=/home/shihao/pistar/.venv/bin:$PATH
WORKDIR /home/shihao/isaac_rm

# 默认进 bash；真正的训练命令在平台「启动命令」里填（见 docker/README.md）
CMD ["/bin/bash"]
