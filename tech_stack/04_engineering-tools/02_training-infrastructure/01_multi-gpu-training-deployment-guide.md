# 多 GPU 训练部署与恢复

## 1. 概述

训练部署将代码、模型、仿真环境和计算资源组成可恢复的实验。流程从资源与版本核对开始，经过单卡更新、多卡同步和目标负载验证，再进入正式运行、监控与检查点归档。

## 2. 基础与定义

### 2.1 运行环境

宿主机提供 GPU 驱动，容器封装 CUDA 用户态依赖、PyTorch 和仿真器。镜像固定基础系统与依赖，挂载目录提供源码、数据、模型和输出。GPU 可见性由容器运行时和进程设备选择共同决定。

### 2.2 实验标识

每次实验保存唯一运行 ID、源码提交、镜像摘要、配置、初始权重校验值、随机种子、硬件和启动命令。公共日志由指定进程写入，各 Rank 保留独立错误日志。

## 3. 核心原理

### 3.1 容器与存储

Docker 和 containerd 运行元数据按主机隔离，数据和模型按权限共享。图形缓存、编译缓存与仿真缓存使用已验证的隔离方式，防止跨主机或并发进程覆盖。持久训练输出通过显式挂载保存。

镜像、驱动、GPU 架构和仿真版本共同决定运行兼容性；headless 模式仍可能需要图形驱动和渲染组件。先验证张量计算，再验证传感器生成与实际反向更新。[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### 3.2 进程与资源

每个 Rank 绑定明确设备，环境数、场景数和批量参数注明全局或局部语义。容器内编号可能不同于宿主机物理编号。GPU 显存、CPU、共享内存与输出存储分别测量。[Docker 资源配置](https://docs.docker.com/engine/containers/resource_constraints/)

### 3.3 保存与恢复

恢复状态包括模型、优化器、调度器、归一化、随机数、全局步数及任务所需的课程状态。精确延续还依赖环境与仿真状态的可恢复性；只恢复权重时按新实验记录。检查点完整写入并校验后，再作为恢复入口。

## 4. 实现与应用

### 4.1 部署流程

| 阶段 | 验证结果 |
|---|---|
| 资源审计 | 设备、存储、运行时和权限明确 |
| 单卡短测 | 环境推进、推理、反向与保存可完成 |
| 多卡短测 | 至少两轮更新完成，参数与统计一致 |
| 目标负载 | 内存、吞吐与通信保持可用 |
| 恢复验证 | 检查点可读，步数与状态按约定衔接 |

以下命令使用占位主机、路径与镜像，执行时替换为已核实的实验环境。

<details class="kt-derivation" id="resource-audit">
<summary>服务器资源审计</summary>

**连接与系统信息**

```bash
ssh USER@SERVER_IP
hostname -s
uname -a
lsblk -f
df -hT
findmnt -T /var/lib/docker
free -h
```

**GPU 与容器运行时**

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
docker version
docker info --format '{{.DockerRootDir}}'
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
docker compose version
systemctl is-active docker containerd
```

检查 NVIDIA 驱动、目标 GPU 空闲状况、Docker 权限及磁盘余量。`findmnt` 或某些查询命令对不存在的目标返回非零，不应直接判定整台服务器故障。

如果需要安装或修复驱动及 NVIDIA Container Toolkit，按服务器发行版和驱动版本采用官方安装文档，避免在已有业务的服务器上盲目升级。

**资源记录**

| 项目 | 必须记录 |
|---|---|
| 主机 | 主机名、系统版本、登录用户 |
| GPU | 型号、数量、物理编号、空闲显存 |
| 存储 | 挂载点、文件系统、可用空间 |
| 容器 | Docker 根目录、containerd 根目录、运行中容器 |
| 软件 | 驱动、镜像标签、PyTorch 与 CUDA 运行版本 |

</details>

<details class="kt-derivation" id="storage-and-runtime">
<summary>存储与容器运行环境</summary>

**目录规划**

每台服务器使用**主机专属**的系统缓存和训练输出目录。推荐结构如下：

```text
DATA_ROOT/
├── system-cache/HOST_ID/
│   └── store/
│       ├── docker/
│       ├── containerd/
│       └── runtime-cache/
└── training/HOST_ID/
    ├── project/
    ├── artifacts/
    ├── outputs/
    └── image-archives/
```

共享 NFS 等存储可以存放项目、模型文件及训练产物，但**不应让多台主机共享同一份 Docker/containerd 运行元数据**。Docker 根目录优先使用本地 ext4 或其他已验证兼容的本地文件系统。若必须借助共享盘承载本机专属 ext4 loop 文件，应由管理员创建、挂载并验证重启后的自动挂载，不得复用其他主机的 loop 文件。

**Docker 根目录迁移**

只有确认当前系统盘容量不足，并且获得维护授权时才执行迁移。迁移前必须：

1. 确认 Docker 和 containerd 的**实际**根目录与挂载类型。
2. 保存现有容器、镜像和服务配置清单。
3. 备份 `/etc/docker/daemon.json` 和 `/etc/containerd/config.toml`。
4. 确认目标目录位于主机专属、空间足够的文件系统。
5. 通知相关使用者，并停止依赖 Docker/containerd 的业务。

迁移时停止相关服务，用 `rsync -aHAX --numeric-ids` 将**已核实的实际源目录**复制到新目录，再谨慎修改配置：

- Docker 在 `daemon.json` 中使用 `data-root`。
- containerd 在与当前版本兼容的配置中设置顶层 `root`。
- 合并原配置，不要用空白模板覆盖原有运行时、Registry 或网络设置。

重启后验证：

```bash
docker info --format '{{.DockerRootDir}}'
docker ps -a
findmnt -T DATA_ROOT/system-cache/HOST_ID/store/docker
findmnt -T DATA_ROOT/system-cache/HOST_ID/store/containerd
```

出现问题时先恢复原配置和旧根目录，不要急于删除源文件。**不得在未识别的块设备上执行格式化命令。**

**运行缓存与权限**

图形仿真、渲染和数据处理框架可能会生成大量缓存。将其映射到每台服务器的专属目录，避免多个 Rank 争用同一缓存。检查容器内用户对输出目录具有写权限，不要为了解决权限问题直接放宽整个共享盘的访问权限。

</details>

<details class="kt-derivation" id="image-deployment">
<summary>项目与容器镜像部署</summary>

**源码与资源同步**

保留项目所需源码、配置文件、模型初始化权重和数据资源，避免将历史日志与大体积缓存打进镜像。

```bash
rsync -aP \
  --exclude '.git/' \
  --exclude 'outputs/' \
  --exclude '__pycache__/' \
  /path/to/local-project/ \
  USER@SERVER_IP:DATA_ROOT/training/HOST_ID/project/
```

执行前先确认目标目录。默认不要使用可能删除远端文件的 `rsync --delete`。初始化权重、数据集和检查点应单独同步至 `artifacts/`，并记录校验值与版本来源。

**镜像构建与迁移**

优先固定 PyTorch、CUDA、仿真框架和算法库版本。修改 Python 源码且使用 Bind Mount 时，通常无需重建镜像；改变系统依赖、Dockerfile 或底层仿真组件后，应生成新镜像标签并重新验证。

```bash
docker build -t IMAGE_TAG .
docker save -o training-image.tar IMAGE_TAG
# 在另一台服务器上
# docker load -i training-image.tar
```

跨主机传输使用 Registry 或 `docker save`、`docker load`，不要复制共享的 Docker 数据目录。

**GPU 容器可用性**

选择已准备好的 CUDA 测试镜像并确认容器能访问 GPU：

```bash
docker run --rm --gpus all CUDA_TEST_IMAGE nvidia-smi
```

随后检查训练镜像中 Python、PyTorch、CUDA、仿真组件和分布式后端是否均可正常导入。CUDA 编译版本、驱动支持范围与框架版本须相互兼容。

</details>

<details class="kt-derivation" id="distributed-launch">
<summary>单机多 GPU 启动</summary>

**进程与设备约定**

常用模式为每张可见 GPU 一个训练进程。`torchrun` 自动设置 `LOCAL_RANK`、`RANK`、`WORLD_SIZE` 等环境变量，程序应根据 `LOCAL_RANK` 绑定设备。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py
```

`train.py` 必须已实现分布式初始化、按 Rank 创建环境和统一结束流程。**仅添加 `torchrun` 不会自动让单卡 PPO 变成正确的多卡 PPO。**

**容器内启动示例**

下面假设当前项目目录可被容器映射，训练镜像及程序入口已经验证：

```bash
docker run --rm \
  --name training-RUN_ID \
  --gpus '"device=0,1,2,3"' \
  --shm-size=8g \
  -v "$PWD":/workspace \
  -v "DATA_ROOT/training/HOST_ID/outputs":/outputs \
  -w /workspace \
  IMAGE_TAG \
  torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py
```

容器内 GPU 编号可能被重新映射，须在容器内部检查 `nvidia-smi` 与每个进程的实际设备。不同仿真器可能还需要额外的缓存挂载、共享内存或图形运行时参数，按项目的已验证运行约定补充。

**训练配置**

配置文件必须明确：

| 字段 | 必须约定的语义 |
|---|---|
| GPU 数量 | 进程数量和可见设备范围 |
| 环境数 | 全局总数还是每 Rank 数量 |
| 场景数 | 全局总数还是每 Rank 数量 |
| Rollout 长度 | 每环境采样步数，还是每 Rank 总步数 |
| Episode 长度 | 单回合最长步数 |
| Mini-batch | 本地大小、数量与更新次数 |
| 随机种子 | 主种子及各 Rank 的派生方式 |

不能依据参数名称猜测上述含义，应检查训练脚本实际实现并在配置中记录。

**多服务器模式**

多台服务器独立运行实验时，各自设置唯一 `RUN_ID`、输出目录与缓存，不加入同一个分布式进程组。若需要跨节点联合训练，则必须额外配置主节点地址、通信端口、节点序号、网络互通及跨节点故障恢复，并单独完成正确性验收。

</details>

<details class="kt-derivation" id="staged-validation">
<summary>分阶段验收</summary>

| 阶段 | 操作 | 通过标准 |
|---|---|---|
| 静态检查 | 目录、权限、镜像、驱动、GPU 映射 | 所有依赖与目标设备可访问 |
| 单卡测试 | 运行简化任务和短 Rollout | 环境推进、推理及更新成功 |
| 多卡短测 | 小环境数，至少两轮完整 PPO 更新 | 全部 Rank 完成且参数一致 |
| 压力测试 | 目标环境数和障碍规模 | 无 OOM、死锁、持续掉速 |
| 正式训练 | 新 `RUN_ID`、固定代码版本 | 指标、日志和检查点正常生成 |

多卡验收必须进入至少一次真实 `backward` 与参数更新，不能仅以容器启动、GPU 可见或 NCCL 初始化成功为通过标准。

</details>

<details class="kt-derivation" id="experiment-records">
<summary>正式运行与实验记录</summary>

每次训练创建唯一实验目录：

```text
outputs/RUN_ID/
├── config/
├── logs/
├── rank_logs/
├── tensorboard/
├── checkpoints/
└── diagnostics/
```

建议固定并保存源代码提交号、镜像标签、随机种子、硬件信息、完整训练配置、初始化权重及数据版本。公共日志和检查点优先由 Rank 0 写入，避免多进程覆盖同一文件；各 Rank 的错误日志应单独保留。

</details>

<details class="kt-derivation" id="runtime-monitoring">
<summary>监控与日志</summary>

**服务器状态检查**

```bash
nvidia-smi
nvidia-smi pmon -c 1
docker ps
docker stats --no-stream
docker logs --tail 200 CONTAINER_NAME
```

检查训练步数是否持续推进、各 Rank 是否正常、GPU 显存是否稳定，以及是否存在某一 Rank 明显落后。

**TensorBoard 与远程访问**

在服务器上将 TensorBoard 限制为回环地址：

```bash
tensorboard --logdir /path/to/outputs/RUN_ID/tensorboard \
  --host 127.0.0.1 --port 6006
```

本地电脑通过 SSH 端口转发访问：

```bash
ssh -N -L 6006:127.0.0.1:6006 USER@SERVER_IP
```

浏览器打开 `http://127.0.0.1:6006`。多实验同时运行时使用不同端口。其他日志平台如 W&B 可设为离线模式，但离线记录不等于实时云端访问。

</details>

<details class="kt-derivation" id="recovery-and-debugging">
<summary>停止、恢复与故障排查</summary>

**安全停止与恢复**

优先使用程序提供的安全退出入口，等待完成正在执行的必要写入。恢复训练前检查是否保存：模型参数、优化器状态、学习率调度器、随机数状态、全局步数，以及任务所需的环境和课程状态。

仅有模型权重不代表能够精确恢复此前训练。恢复后先做小规模验证，再继续目标负载。

**常见故障**

| 现象 | 优先检查 |
|---|---|
| CUDA OOM | 每 Rank 环境数量、观测缓存、Rollout Buffer、Mini-batch |
| NCCL Timeout | 最早失败的 Rank、通信调用顺序、设备映射、网络状态 |
| GPU 利用率低 | 环境推进耗时、CPU 饱和、同步等待、数据传输 |
| 参数不一致 | 初始广播、优化器恢复、更新步数、同步遗漏 |
| 日志停滞 | 进程状态、Rank 错误日志、文件系统及剩余空间 |
| 恢复后异常 | 版本漂移、部分状态缺失、随机种子和配置不匹配 |

NCCL 超时可能只是某个 Rank 更早发生异常的连锁结果，应从第一处报错开始排查，不要仅靠延长超时时间掩盖问题。

</details>

## 5. 分析与总结

部署验收以完整训练更新和可读取检查点为证据。故障定位从最早异常开始，再检查跨 Rank 传播；NCCL 超时可能由某个进程先发生 OOM 或环境异常引起。训练曲线的算法解释见[训练指标与诊断](/knowledge-tree/learning-based-planning/engineering-practice/tensorboard-metrics/)。

## 6. 关联与参考

- [分布式训练架构](./02_distributed-rl-training-architecture.md)。
- [PyTorch torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)。
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)。
- [Docker 存储](https://docs.docker.com/engine/storage/)。
