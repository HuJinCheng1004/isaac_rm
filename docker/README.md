# isaac_rm 训练镜像：打包并在托管平台上跑

目标：把 `scripts/train.py`（Isaac Sim 5.1 + IsaacLab v2.3.2 + skrl）打成 Docker 镜像，
交给网页端训练平台运行——**平台帮你跑容器并提供 GPU，所以你完全不需要在目标服务器上 sudo。**

镜像策略：**复用本机已验证可用的 venv**（不在容器里重装 Isaac Sim）。
镜像里这几棵目录落在与本机【相同的绝对路径】下，缺一不可：

| 路径 | 作用 |
|---|---|
| `/home/shihao/.local/share/uv/python/cpython-3.11.9-...` | venv 的 `bin/python` 软链目标 |
| `/home/shihao/pistar/.venv` | isaacsim / torch / skrl |
| `/home/shihao/IsaacLab` | isaaclab editable 源码（`.pth` 指向） |
| `/home/shihao/isaac_rm` | 训练代码 + assets + URDF |
| `/home/shihao/XRTeleop/.../meshes` | URDF 写死的网格绝对路径 |

---

## 第 1 步：本机构建镜像（在有 sudo + GPU 的这台机器上）

```bash
bash docker/build_image.sh
```

产物：`docker/isaac-rm-5.1.tar.gz`（~8–11G）。

> 注意：脚本用 `sudo docker`（当前用户不在 docker 组）。需要约 25–30G 空闲磁盘。

## 第 2 步：本机冒烟测试（强烈建议，省得上传后才发现问题）

```bash
# --gpus all 需要本机装了 nvidia-container-toolkit；没装可 sudo 装，或跳过直接上平台测
sudo docker run --rm --gpus all isaac-rm:5.1 \
    python /home/shihao/isaac_rm/scripts/train.py \
    --task Isaac-Chassis-Approach-v0 --num_envs 32 --headless --max_iterations 2
```

能跑完 2 个迭代（约十几秒）就说明镜像 OK。
若报缺少某个 `libXXX.so`：把对应包名加到 `Dockerfile` 的 apt 列表里重建即可。

## 第 3 步：上传到平台（网页「构建镜像」页，按下面三选一）

通用必选项：
- **适用加速卡 = GPU**（Isaac Sim 必须有 N 卡）
- **选择镜像类型 = Base**（自带完整环境，不用平台的 PyTorch 等基础镜像）

### 方式 A（推荐）：外部镜像仓库 —— 适合大镜像
镜像 8G+，浏览器直传容易超时/超限。若你有平台能访问的镜像仓库
（阿里云 ACR / 腾讯 TCR / 自建 Harbor，注意国内多半访问不了 Docker Hub）：

```bash
sudo docker tag isaac-rm:5.1 <仓库地址>/<命名空间>/isaac-rm:5.1
sudo docker login <仓库地址>
sudo docker push <仓库地址>/<命名空间>/isaac-rm:5.1
```

网页里「构建方式」选 **外部镜像仓库**，填上面的镜像地址（私有仓库再填账号密码/凭证）。

### 方式 B：镜像包 —— 没有仓库时用
网页「构建方式」选 **镜像包**，「文件路径」选刚才的 `docker/isaac-rm-5.1.tar.gz` 上传。
（平台对单文件大小有上限，太大就走方式 A。）

### 方式 C：Dockerfile —— 不推荐
让平台自己联网 pip 装 Isaac Sim。只有当平台构建机能访问 `pypi.nvidia.com` 和 GitHub
时才可行，且无法本地预先验证、Isaac Sim 容器化坑多。本仓库的 Dockerfile 是「烤 venv」式的，
不适用于这种方式（需另写从头安装的 Dockerfile）。

## 第 4 步：在平台上填启动命令

```bash
OMNI_KIT_ACCEPT_EULA=YES python /home/shihao/isaac_rm/scripts/train.py \
    --task Isaac-Chassis-Approach-v0 --num_envs 4096 --headless
```

（镜像已设 `OMNI_KIT_ACCEPT_EULA=YES` 和 venv 的 PATH，命令里再写一遍也无妨。）

> **Checkpoint 落盘**：训练输出在 `/home/shihao/isaac_rm/logs/skrl/...`。容器是临时的，
> 务必把日志/权重写到平台的**持久化存储/挂载目录**（如平台给的 `/workspace` 或数据卷），
> 否则容器回收后丢失。可在启动命令里 `cd` 到挂载目录再跑，或软链 `logs` 到挂载点。

---

## 常见问题

- **CUDA / 驱动不匹配**：基础镜像是 CUDA 12.6。若目标平台 GPU 驱动较老报 CUDA 错，
  把 Dockerfile 的 `FROM` 换成更低版本（如 `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`）或
  `ubuntu:22.04`（torch/isaacsim 轮子自带 CUDA）后重建。
- **Vulkan 看不到 GPU / PhysX-GPU 报错**：确认平台用的是 NVIDIA 容器运行时，
  且 `NVIDIA_DRIVER_CAPABILITIES` 含 `graphics`（镜像已设 `all`）。
- **权限**：镜像内文件属 uid 1000。若平台以其他非 root 用户跑、写缓存失败，
  在启动命令前加 `export HOME=/tmp` 或让平台以 root 运行。
