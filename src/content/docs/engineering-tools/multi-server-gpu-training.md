---
title: "YOPO RL Selector：多服务器部署与单机多 GPU 训练操作手册"
sidebar:
  order: 5
---

本文总结当前项目已经采用的流程：在**每台服务器上分别部署**同一套 Docker 镜像和代码，再在该服务器内用 `torchrun` 做单机多 GPU 同步训练。多台服务器可以同时运行不同实验；当前脚本**不把不同服务器组成同一个跨节点进程组**。

操作入口以 `docker/train.sh` 为准。以下命令中的 `USER`、`SERVER_IP`、`HOST_ID`、`DATA_ROOT`、`RUN_ID` 都是占位符，执行前替换；密码在 SSH 提示时交互输入，不写进命令、脚本、`.env` 或日志。全文中的服务器检查命令可先只读执行；Docker 根目录迁移必须有维护窗口和停止现有容器的授权。

## 0. 总体流程和关键口径

```text
逐台服务器只读审计
  → 确认数据盘类型、现有容器与 GPU 占用
  → 安全迁移 Docker/containerd root（确有必要且允许停机时）
  → 主机专属 runtime-cache
  → 同步项目、YOPO 权重和镜像
  → 单卡 smoke → 多卡短测 → 目标规模压力测试
  → 正式训练 → 日志/TensorBoard 监控 → checkpoint 验收
```

每台服务器独立执行这条链。指定物理 GPU 后，容器内设备重编号为 `cuda:0..N-1`，每卡一个进程、一个 Isaac 实例和一份本地环境；Selector PPO 梯度通过 NCCL 同步，只有 rank 0 写公共日志和 checkpoint。`--num-envs` 是**该次训练的全局环境数**，必须能被 GPU 数整除；树和几何体的数量则是**每个 rank 的场景数量**，不是多卡合计。`--steps` 是每个环境并行推进的训练循环步数，训练 transition 总数约为 `steps × num-envs`；`--episode-steps` 才是单个 env 的回合 timeout 上限。

## 1. 每台服务器先做只读审计

本地电脑登录：

```bash
ssh USER@SERVER_IP
```

在远端确认主机、磁盘、挂载、GPU、容器和服务状态：

```bash
hostname -s
lsblk -f
df -hT
findmnt /diff
findmnt /mnt/fsvln-data
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
docker info --format '{{.DockerRootDir}}'
docker compose version
grep '^[[:space:]]*root' /etc/containerd/config.toml
systemctl is-active docker containerd
```

`findmnt` 对不存在的挂载点或 `grep` 对使用默认 containerd root 的配置返回非零，并不必然是故障；继续查看 `findmnt -T /var/lib/containerd` 和配置文件。还需确认 `docker info` 显示 NVIDIA runtime 可用，并核对目标 GPU 没有其他业务占用。**不要因为要训练就停掉不属于本项目的容器。**记录现有 Docker/containerd 配置及 `docker ps -a` 结果，作为迁移前的回退依据。

存储分两类：

| 场景 | 目录与文件系统要求 |
| --- | --- |
| 多台机器共享 `/diff` NATS/NFS | 每台机器使用自己的 `/diff/system-cache/HOST_ID/store`；Docker/containerd 必须位于这个主机专属的 ext4 loop 挂载里，不能直接共用 `/diff/docker_root` 或直接把 Docker 元数据放在共享文件系统上。 |
| 服务器独占的 ext4 数据盘 | 例如 `/mnt/fsvln-data`。可在此盘建立 `system-cache/HOST_ID/store`，无需再套 loop；仍保留主机名隔离。 |

先用 `findmnt -T DATA_ROOT` 确认真实文件系统和挂载来源。`HOST_ID` 可取 `hostname -s`，但同名主机仍要人为保证目录唯一。

## 2. Docker/containerd 根目录迁移

目标结构如下；`DATA_ROOT` 在共享盘机器可取 `/diff`，在独占盘机器可取实际挂载点：

```text
DATA_ROOT/system-cache/HOST_ID/store/
├── docker/
├── containerd/
└── runtime-cache/
    ├── home/
    ├── pip/
    ├── ov/
    ├── kit/{cache,data,logs}/
    ├── gl/
    ├── compute/
    └── ranks/
```

共享 `/diff` 时，先由主机管理员为**这台机器**准备独立 ext4 loop 文件并挂载到 `.../store`，确认 `findmnt -T .../store` 的 `FSTYPE=ext4`、来源为本机 loop 设备，且重启后能恢复挂载。这里不能复用另一台机器的 loop 文件或 Docker root。不要对已有文件/块设备执行 `mkfs`；如果无法确认目标为新建且空白的专属存储，停在此处。

迁移前置条件：已有容器清单已核对、相关使用者同意停机、数据盘空间足够、旧目录留作回退。若服务器仍有不可中断的业务容器，**跳过迁移并暂停正式部署**，不要边运行边复制 Docker 元数据。获得维护窗口后，以当前实际 root 为源、主机专属 ext4 路径为目标执行以下步骤；下面的 `/var/lib/...` 仅适用于只读检查确认它们确实是当前 root 的机器：

```bash
sudo systemctl stop docker.socket docker containerd
sudo mkdir -p DATA_ROOT/system-cache/HOST_ID/store/docker
sudo mkdir -p DATA_ROOT/system-cache/HOST_ID/store/containerd
sudo rsync -aHAX --numeric-ids /var/lib/docker/ DATA_ROOT/system-cache/HOST_ID/store/docker/
sudo rsync -aHAX --numeric-ids /var/lib/containerd/ DATA_ROOT/system-cache/HOST_ID/store/containerd/
```

保留 `/etc/docker/daemon.json` 和 `/etc/containerd/config.toml` 的备份。合并而非覆盖已有配置：在 `daemon.json` 设置 `"data-root": "DATA_ROOT/system-cache/HOST_ID/store/docker"`；在 `config.toml` 顶层设置 `root = "DATA_ROOT/system-cache/HOST_ID/store/containerd"`。若 containerd 配置文件原本不存在，先生成兼容本机版本的默认配置，再修改顶层 `root`；不要照搬别的机器的配置。然后：

```bash
sudo systemctl start containerd docker
docker info --format '{{.DockerRootDir}}'
grep '^[[:space:]]*root' /etc/containerd/config.toml
findmnt -T DATA_ROOT/system-cache/HOST_ID/store/docker
findmnt -T DATA_ROOT/system-cache/HOST_ID/store/containerd
docker ps -a
```

前三项应分别指向本机专属 `docker`、`containerd`、正确的 ext4 挂载；迁移前的容器/镜像也应仍可见。若失败，停止服务并用**备份的配置**恢复原 root，再启动服务排查。旧 `/var/lib/docker`、`/var/lib/containerd` 暂时保留，验收、备份和回退窗口结束后再另行决定是否清理。本步骤不安装代理、VPN 或翻墙软件。

## 3. 项目、权重、镜像与缓存部署

推荐每台机器使用自己的项目和输出目录：

```text
DATA_ROOT/yopo-runs/HOST_ID/
├── project/YOPO_RL_SELECTOR/
├── artifacts/models/
├── outputs/
└── image-archives/
```

从本地同步项目代码时排除训练输出、Git 元数据和运行缓存；不要使用可能删除远端既有数据的 `rsync --delete`。示例：

```bash
# 先在远端建立本机专属目录；不要把 HOST_ID 留成字面值
mkdir -p DATA_ROOT/yopo-runs/HOST_ID/project/YOPO_RL_SELECTOR
mkdir -p DATA_ROOT/yopo-runs/HOST_ID/artifacts/models
mkdir -p DATA_ROOT/yopo-runs/HOST_ID/outputs
mkdir -p DATA_ROOT/yopo-runs/HOST_ID/image-archives
```

回到本地电脑执行：

```bash
rsync -aP \
  --exclude '/outputs/' --exclude '/.git/' \
  --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '/docker/runtime-cache/' \
  /path/to/workspace/YOPO_RL_SELECTOR/ \
  USER@SERVER_IP:DATA_ROOT/yopo-runs/HOST_ID/project/YOPO_RL_SELECTOR/
```

将 YOPO checkpoint 及其需要的契约/配置文件同步到 `.../artifacts/models/`。当前 YOPO-swarm 无 mask 主线的一个权重例子是 `swarm/YOPO_Omni_no_mask/epoch100.pth`；实际训练务必以已验证的文件为准。不要把大量数据集、旧 outputs 或模型历史版本一并放进镜像。

镜像复用使用私有 registry 或 `docker save` → 传输 → `docker load`；**绝不通过共享 Docker root 复用镜像**。如果本地镜像已通过 smoke，可在本地导出：

```bash
docker save -o yopo-rl-selector-isaac5.1-lab2.3-skrl2.1.tar yopo-rl-selector:isaac5.1-lab2.3-skrl2.1
rsync -aP yopo-rl-selector-isaac5.1-lab2.3-skrl2.1.tar \
  USER@SERVER_IP:DATA_ROOT/yopo-runs/HOST_ID/image-archives/
```

远端在数据盘上的归档执行 `docker load -i ...tar`；若要在远端重新构建，使用项目里的 `./docker/train.sh build`。两个方案都应预留 Isaac 镜像解包和 Docker 层的充足空间。镜像环境包含 Isaac Sim 5.1、Isaac Lab 2.3.0 和 skrl 2.1.0。Python 源码由 bind mount 提供；仅修改项目 Python 代码通常无需重建镜像，修改依赖锁文件、Dockerfile 或 Isaac 环境才需要重建并重跑 smoke。

在远端项目的 `docker/.env` 配置：

```dotenv
YOPO_HOST_ID=HOST_ID
YOPO_PROJECT_ROOT=DATA_ROOT/yopo-runs/HOST_ID/project/YOPO_RL_SELECTOR
YOPO_ARTIFACT_ROOT=DATA_ROOT/yopo-runs/HOST_ID/artifacts/models
YOPO_OUTPUT_ROOT=DATA_ROOT/yopo-runs/HOST_ID/outputs
YOPO_CACHE_ROOT=DATA_ROOT/system-cache/HOST_ID/store/runtime-cache
YOPO_UID=1000
YOPO_GID=1000
YOPO_IMAGE=yopo-rl-selector:isaac5.1-lab2.3-skrl2.1
```

`YOPO_UID/GID` 应改为远端实际运行账号的 `id -u` / `id -g`，并确保输出和 cache 可写。对共享 `/diff`，`YOPO_CACHE_ROOT` 必须精确位于上述主机专属 `store/runtime-cache`，因为脚本会按此路径校验。由脚本建立 pip、OV、Kit、GL、compute 和各 run/rank 缓存目录；不要跨服务器共享 cache。`docker/.env` 不放密码或 W&B API key。可用 `./docker/train.sh init` 创建模板后逐项检查；该命令若发现 `.env` 已存在不会覆盖。

## 4. 部署验收：从静态检查到目标负载

在远端项目根目录运行：

```bash
cd DATA_ROOT/yopo-runs/HOST_ID/project/YOPO_RL_SELECTOR
./docker/train.sh paths
./docker/train.sh validate-storage
./docker/train.sh gpu-map 0,1,2,3
./docker/train.sh preflight --offline
./docker/train.sh config
```

`validate-storage` 是脚本检查，独占数据盘场景还需手工对照第 2 节的两个 root 和挂载结果。`preflight --offline` 不拉取 CUDA smoke 镜像；如果镜像/网络已具备，可以再用 `./docker/train.sh preflight` 检查 NVIDIA 容器运行时。对目标 GPU 做完整 Isaac 导入测试，例如只使用物理 GPU 0：

```bash
YOPO_GPU_DEVICE_IDS=0 YOPO_CONTAINER_CUDA_VISIBLE_DEVICES=0 ./docker/train.sh smoke
```

接下来三阶段验收，逐级推进，不以“容器启动成功”代替训练成功：

1. **多卡链路短测**：例如 4 GPU、全局 8 env、`--steps 128 --rollouts 8 --epochs 1 --mini-batches 2`，低障碍物数量和唯一的临时 `RUN_ID`。确认每 rank 的 GPU、至少两次 PPO update、NCCL 梯度同步、只有 rank 0 写 output，且 TensorBoard/W&B 离线文件产生。短测结束后用 `status`、日志及产物验收。
2. **目标场景压力测试**：先保持实际 GPU 数和目标障碍物数量，使用目标 `--num-envs`，但只运行数百到数千 step。核对首个完整 PPO update、持续推进、CPU/RAM/显存余量、是否有单 rank 拖慢、reward 是否有限值、是否出现 OOM。某些场景在首次 PPO 更新前就很慢，不能只看 Isaac 启动日志。
3. **正式训练**：压力测试通过后换**新的 RUN_ID**，再设置计划中的总步数和 checkpoint 周期。`train.sh` 拒绝覆盖已有输出，避免误删训练记录。

容量计算用于选择 `mini-batches`：若全局 512 env、4 GPU、`rollouts=64`，则每 rank 有 128 env 和 `128×64=8192` transitions；`mini-batches=64` 时本地每批 128，全局等效约 512。环境数、rollout、PPO epoch 与场景物体数量都会影响吞吐和内存，先实测后放大；OOM 时优先检查每 rank 的环境和场景数量，再调整 rollout/minibatch。`--steps` 必须写整数（例如 `2000000`），不要传 `1e10` 字符串。

## 5. 正式分布式训练示例

以下是 YOPO-swarm 无 mask 权重、4 张指定 GPU、box-tree 的**模板**，不是所有服务器的固定最优参数。先确认权重位于 `YOPO_ARTIFACT_ROOT` 下、四张 GPU 空闲、输出盘容量充足；物理 GPU 编号和 `RUN_ID` 按服务器修改。

```bash
cd DATA_ROOT/yopo-runs/HOST_ID/project/YOPO_RL_SELECTOR

./docker/train.sh run-distributed --gpus 0,1,2,3 RUN_ID \
  --map box-tree --curriculum true --uav-obs-other false \
  --depth-backend analytic --depth-preprocess none \
  --selector-weight-mode yopo-init --selector-downsample area \
  --yopo-checkpoint DATA_ROOT/yopo-runs/HOST_ID/artifacts/models/swarm/YOPO_Omni_no_mask/epoch100.pth \
  --num-envs 512 \
  --static-count 512 --dynamic-count 256 \
  --dynamic-cube-count 256 --dynamic-sphere-count 256 \
  --steps 2000000 --episode-steps 3200 \
  --rollouts 64 --epochs 5 --mini-batches 64 \
  --learning-rate 0.0001 --discount-factor 0.999 \
  --curriculum-success-thresholds 0.8 0.8 0.8 \
  --logger wandb --wandb-mode offline --wandb-artifacts none \
  --tensorboard true --tensorboard-flush-rollouts 1 \
  --log-every-rollouts 1 --save-check 50000 \
  --runtime-speed 3 --seed 42
```

`run-distributed` 自动添加 `--headless --mode train --output outputs/RUN_ID`，所以不要再手传这些参数中的 `--output`。宿主机上位于 `YOPO_ARTIFACT_ROOT` 内的 checkpoint 路径会自动转换成容器路径。若只用一张卡，改为 `./docker/train.sh run --gpu GPU_ID RUN_ID ...`，其余参数仍按单卡资源调整。多台服务器同时训练时，每台服务器各自运行 `run` 或 `run-distributed`，使用独立 `RUN_ID`、输出目录和缓存；当前不提供跨服务器联合梯度同步。

## 6. 查看进度、资源和训练曲线

远端在项目根目录：

```bash
./docker/train.sh status RUN_ID
./docker/train.sh logs RUN_ID
./docker/train.sh monitor RUN_ID
```

`logs` 会把 tqdm 的回车刷新转换成逐行输出，便于通过 SSH 看 step/s。原生命令是 `docker logs --tail 200 -f yopo-RUN_ID`。还可用 `nvidia-smi` 查看 GPU，`docker stats --no-stream yopo-RUN_ID` 查看容器资源。周期 checkpoint 在 `outputs/RUN_ID/checkpoints/selector_<step>steps.pt`，rank 异常报告在 `outputs/RUN_ID/rank_logs/`。若日志停止推进，检查容器状态、所有 rank 日志、首个异常、OOM 和 GPU/CPU 负载；NCCL timeout 可能只是某个 rank 先失败后的连锁现象。

训练时启用 W&B **offline** 和 TensorBoard 双记录。W&B offline 不上传云端、无需远端登录，但自身不能直接提供实时网页。实时查看使用 TensorBoard：

```bash
# 远端：仅监听远端回环地址，不向公网开放 6006
./docker/train.sh tensorboard RUN_ID 6006
```

在本地**另开终端**：

```bash
ssh -N -L 6006:127.0.0.1:6006 USER@SERVER_IP
```

本地浏览器访问 `http://127.0.0.1:6006`。多台服务器或多个 run 同时监控时，各 sidecar 与本地 SSH 隧道使用不同端口。若 SSH 出现 `connect failed: Connection refused`，先在远端检查 `yopo-tensorboard-RUN_ID` 是否运行、日志目录是否存在、端口是否已被占用；只建立隧道不会自动启动 TensorBoard。结束查看可运行 `./docker/train.sh tensorboard-stop RUN_ID`。离线 W&B 原始数据保存在 `outputs/RUN_ID/wandb/`，需要时再通过 `rsync` 拉回本地分析，不要把 `wandb sync` 当作严格离线实时查看方式。

## 7. 更新、停止、恢复和交付检查

- **只改超参数**：用新 `RUN_ID` 启动；不重建镜像。
- **修改项目 Python 代码**：同步新的项目快照，正常停止旧训练后重启新 run；源码是只读 bind mount，通常不重建镜像。运行中的进程不会自动采用新代码。
- **修改依赖、Dockerfile、Isaac 或 CUDA 环境**：构建/传输新镜像，重新 `preflight`、`smoke` 和短测，记录镜像标签和代码版本。
- **停止项目训练**：`./docker/train.sh stop RUN_ID` 会向容器发 SIGINT 并等待；先看状态与最后 checkpoint，再决定是否进一步处理。不要用停止 Docker 服务代替停止训练。
- **恢复训练**：先核对当前脚本是否提供与该 checkpoint 对应的恢复入口和训练状态契约，再用新 `RUN_ID` 验证恢复；不要只凭 `.pt` 文件存在就假设优化器、课程等级及 rollout 已完整恢复。

每台服务器的最终验收记录至少包括：主机名及存储挂载、Docker/containerd root、镜像标签与源码版本、权重路径、物理 GPU 列表、`RUN_ID`、每 rank env 数、首个 PPO update、持续增长的 step、TensorBoard event 文件、W&B offline 目录、checkpoint、GPU/CPU/RAM/显存余量及已知异常。旧实验输出和镜像归档只有在明确目标、完成备份且确认不再使用之后才单独清理。
