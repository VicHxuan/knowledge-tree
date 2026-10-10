---
title: "强化学习工程避坑指南与注意事项手册"
sidebar:
  order: 2
---

## 1. 环境与回合管理

### 1.1 Step 计数

区分：

* `global_step`：训练总推进步数；
* `episode_step[env]`：单环境回合步数；
* `rollout_step`：当前 PPO rollout 内步数。

单环境 timeout 必须由 `episode_step[env]` 判断。

### 1.2 Reset 类型

自然终止：

```text
success
failure
collision
timeout
```

仅 reset 对应环境。

全局 reset 仅用于：

```text
curriculum切换
地图整体切换
显式全局重初始化
```

### 1.3 Synthetic Reset

课程切换等训练逻辑产生的 reset 不属于自然 termination 或 timeout。

要求：

```text
不计success
不计failure
不计timeout
不产生terminal reward
```

必要时该步 reward 清零。

### 1.4 环境连续性

episode 内禁止无语义依据地刷新：

```text
地图
静态障碍
目标
动态障碍状态
```

动态对象应按照连续动力学更新。

---

# 2. 可观测性

## 2.1 终止因素

能够导致：

```text
collision
failure
状态突变
```

的环境因素，应满足以下至少一项：

```text
Actor可观测
可由历史状态推断
不参与终止判定
```

不可观测对象不应直接产生隐藏碰撞终止。

## 2.2 多智能体交互

其他 Agent 的：

```text
观测
碰撞
奖励影响
```

应使用一致配置。

推荐：

```text
不可观测 → 不启用相互碰撞
可观测   → 可启用相互碰撞
```

---

# 3. Observation / State

## 3.1 状态契约

每个输入变量应明确：

```text
shape
dtype
单位
坐标系
归一化方法
更新时间
reset行为
```

## 3.2 时序信息

动态任务应提供必要历史信息，例如：

```text
frame history
state history
pose history
RNN hidden state
```

历史长度应与目标运动时间尺度匹配。

## 3.3 Ego-motion

多帧视觉/深度输入应考虑自身运动。

基本量：

$$
\Delta R_{t\rightarrow now},\qquad
\Delta p_{t\rightarrow now}
$$

常用方式：

```text
rotation compensation
SE(3) compensation
relative-pose embedding
```

## 3.4 历史初始化

reset 后不得使用旧 episode 历史。

推荐：

```text
首帧复制填满frame history
previous action清零
previous trajectory清零
previous_valid=0
RNN hidden state清零
```

## 3.5 Previous Plan 与 Actual State

应区分：

```text
上一时刻计划执行的参考轨迹
当前实际测量状态
```

二者不可混用。

---

# 4. 坐标系与归一化

## 4.1 坐标系

常见坐标系：

```text
world
body
camera
planning
anchor
sensor
```

每个三维量必须明确其所属坐标系和转换顺序。

## 4.2 坐标转换验证

至少进行：

```text
单轴平移
单轴旋转
固定目标
固定障碍
轨迹正向/逆向重建
```

验证位置、速度、加速度和方向一致性。

## 4.3 数值尺度

输入建议归一化到相近数量级。

典型变量：

```text
position
velocity
acceleration
goal distance
trajectory duration
score
```

禁止使用会在远距离条件下过度压缩有效差异的尺度。

---

# 5. Action 设计

## 5.1 固定动作

传统离散动作：

$$
a\in\{0,\ldots,N-1\}
$$

每个 index 具有固定物理语义。

## 5.2 Candidate Selection Action

候选选择问题中：

$$
\mathcal C_t=
\{c_t^1,\ldots,c_t^N\}
$$

策略输出：

$$
a_t=i
$$

实际语义：

$$
a_t^{physical}=c_t^i
$$

离散 index 仅作为候选引用。

## 5.3 Candidate 可观测性

若 action 表示“选择当前候选 \(i\)”，则候选本身必须进入 Actor observation。

推荐：

$$
l_i=f_\theta(o_t,c_i)
$$

而非仅使用：

$$
[l_1,\ldots,l_N]=f_\theta(o_t)
$$

## 5.4 Shared Scorer

候选选择网络推荐：

```text
candidate feature
+ scene feature
→ shared candidate encoder
→ shared scorer
→ one logit per candidate
```

避免每个 candidate 使用独立固定输出神经元。

## 5.5 Permutation Test

对已构造完成的 candidate tensor 随机排列：

$$
Q'=PQ
$$

应满足：

$$
f(PQ)\approx Pf(Q)
$$

用于检查：

```text
candidate index依赖
positional encoding错误
reshape错位
非共享head
mask错位
```

## 5.6 Candidate Representation Test

检查：

$$
D_{feature}(i,j)\rightarrow0
$$

时是否同时满足：

$$
D_{trajectory}(i,j)\rightarrow0
$$

建议统计：

```text
nearest-neighbor trajectory error
P95 / P99 trajectory error
alias rate
```

---

# 6. Reward 设计

## 6.1 Reward 分层

推荐分为：

```text
任务结果
任务推进
动作质量
时间成本
```

典型结构：

$$
r=
r_{terminal}
+r_{progress}
+r_{action}
+r_{time}
$$

## 6.2 Dense Reward

每个 dense reward 应对应明确的学习目标。

避免重复处罚，例如：

```text
current clearance
swept clearance
static risk
dynamic risk
```

对同一危险重复计权。

## 6.3 Episode 累计量

Reward 系数必须按 episode sum 检查。

若：

$$
r_{dense}\times T
$$

长期大于 terminal reward，则 terminal 可能失去主导作用。

同时记录：

```text
step mean
episode sum
```

## 6.4 Progress Reward

推荐：

$$
r_{progress}
=
clip(
k\Delta d,
r_{min},
r_{max}
)
$$

避免状态跳变或异常值产生过大回报。

## 6.5 Timeout

timeout 应与 success 区分。

推荐：

$$
r_{timeout}<0
$$

即使接近目标但未满足 success，也应保持负终止回报。

---

# 7. 动态风险

## 7.1 时空一致性

动态碰撞判断应比较：

$$
p_{agent}(t+\tau)
$$

与：

$$
p_{obstacle}(t+\tau)
$$

不能仅将未来轨迹与当前障碍位置比较。

## 7.2 常用量

推荐使用：

```text
TTC
time-to-closest-approach
minimum future separation
future collision envelope
```

## 7.3 计算优化

大规模并行环境中避免直接计算：

$$
N_{env}
\times
N_{candidate}
\times
N_{time}
\times
N_{obstacle}
$$

推荐：

```text
broad phase
radius filtering
top-K obstacles
sparse trajectory sampling
piecewise-linear approximation
analytic TTC
batched GPU calculation
```

## 7.4 Static / Dynamic 分离

建议维护：

```text
StaticGeometry
DynamicObstacleBatch
```

动态对象不得同时进入 static risk 和 dynamic risk。

---

# 8. Candidate Regret

对于每条候选定义：

$$
C_i
=
w_dR_{dynamic,i}
+
w_sR_{static,i}
-
w_gG_i
$$

最优候选：

$$
C_{best}=\min_i C_i
$$

选择候选 \(a\) 的 regret：

$$
R_{regret}
=
\frac{C_a-C_{best}}
{\max(C_{max}-C_{min},\epsilon)}
$$

适用于：

```text
trajectory selection
route selection
proposal selection
discrete candidate ranking
```

---

# 9. Collision 与安全尺度

## 9.1 Collision Authority

训练中应指定唯一 termination authority，例如：

```text
depth
physics contact
geometry
```

其他碰撞信号仅用于诊断。

## 9.2 安全距离

区分：

```text
centerline clearance
body clearance
safety clearance
```

一般关系：

$$
d_{body}
=
d_{centerline}
-r_{body}
$$

$$
d_{safe}
=
d_{centerline}
-r_{body}
-margin
$$

## 9.3 控制与规划分离

碰撞诊断至少包含：

```text
reference trajectory
actual trajectory
tracking error
reference clearance
actual clearance
collision signal
physics contact
```

---

# 10. Actor / Critic

## 10.1 Shared Critic

结构：

```text
shared encoder
├── policy head
└── value head
```

特点：

```text
低显存
低计算
Actor/Critic共享梯度
```

## 10.2 Independent Critic

结构：

```text
Actor encoder  → policy
Critic encoder → value
```

特点：

```text
表示解耦
显存与计算成本更高
```

## 10.3 梯度契约

必须明确：

```text
Critic loss是否回传Actor encoder
是否使用detach
哪些层共享
哪些层独立
```

## 10.4 Privileged Critic

若使用 simulator truth，应明确数据边界：

| 信息              | Actor | Critic | Reward |
| --------------- | ----: | -----: | -----: |
| 可部署观测           |     ✓ |      ✓ |     可选 |
| simulator truth |     ✗ |     可选 |      ✓ |
| oracle action   |     ✗ |    通常✗ |  诊断/奖励 |

privileged information 不得进入部署时 Actor 路径。

---

# 11. 网络训练稳定性

## 11.1 Dropout

PPO policy 网络推荐：

```text
dropout = 0
```

避免 rollout 与 update 阶段同一 observation 得到不同 logits。

## 11.2 BatchNorm

若使用预训练 CNN：

```text
固定 running mean / variance
```

避免 PPO 重算 log-prob 时发生分布漂移。

## 11.3 Frozen Layer

冻结配置应明确到：

```text
CNN
Transformer
Query encoder
Policy head
Critic
BN affine
BN running statistics
```

禁止使用模糊的“冻结整个模型”描述。

---

# 12. PPO Batch 配置

## 12.1 基本量

$$
N_{transition}^{local}
=
N_{env}^{local}
\times
T_{rollout}
$$

若每 rank 分为 \(M\) 个 minibatch：

$$
N_{minibatch}^{local}
\approx
\frac{
N_{transition}^{local}
}{M}
$$

全局有效 minibatch 近似：

$$
N_{minibatch}^{global}
=
N_{minibatch}^{local}
\times WORLD\_SIZE
$$

## 12.2 参数语义

区分：

```text
num_envs
rollout length
minibatch size
number of minibatches
epochs
world size
```

多卡训练必须区分 local 与 global 语义。

---

# 13. 分布式训练

## 13.1 环境布局

典型结构：

```text
1 rank
= 1 GPU
= 1 simulator process
= 1 independent scene
```

不同 rank 不共享同一个物理场景。

## 13.2 同步内容

通常仅同步：

```text
gradients
parameters
curriculum state
global statistics
```

## 13.3 随机种子

推荐：

```text
seed_rank = base_seed + rank
```

## 13.4 NCCL Timeout

出现 NCCL timeout 时优先查找：

```text
最早异常rank
NaN / Inf
CUDA error
reward error
observation error
```

通信超时通常是后续现象。

---

# 14. 数值安全

以下数据建议统一 finite check：

```text
observation
state
candidate features
reward components
total reward
logits
value
loss
```

发现 NaN / Inf 时应立即报告：

```text
rank
env id
step
对应张量
对应reward分项
```

---

# 15. 统计与日志

## 15.1 Rate 定义

任何 rate 必须明确：

```text
numerator
denominator
time window
aggregation scope
```

例如 success rate 不应只有名称。

## 15.2 Rollout / Episode / Latest Outcome

建议分开记录：

```text
rollout metrics
episode metrics
latest-outcome metrics
curriculum metrics
```

## 15.3 TensorBoard

诊断离散统计时：

```text
Smoothing = 0
```

平滑显示值不得直接作为原始计数解释。

## 15.4 Reward 日志

每个主要 reward 项记录：

```text
step mean
episode sum
```

---

# 16. Curriculum

## 16.1 晋级条件

除性能阈值外，应设置最小数据覆盖条件。

例如：

```text
coverage >= threshold
success_rate >= threshold
```

## 16.2 晋级后统计

课程切换后应：

```text
清空上一阶段latest outcome
重新统计coverage
```

## 16.3 难度变化

每一级尽量只调整有限因素：

```text
obstacle count
dynamic speed
interaction
map density
```

避免多个维度同时发生大幅变化。

---

# 17. Checkpoint Contract

Checkpoint 应保存：

```text
model architecture
observation contract
action contract
reward version
sensor config
normalization
preprocessing
training parameters
source checkpoint
network freeze config
critic mode
```

文件扩展名不能作为兼容性依据。

---

# 18. 感知预处理

以下内容属于模型输入契约：

```text
resolution
depth range
normalization
resize
pooling
quantization
erosion
hole filling
invalid-value handling
```

训练、测试和部署应保持一致。

不同降采样方式应作为独立实验配置。

---

# 19. 实验流程

推荐顺序：

```text
单元测试
→ 少量env
→ 单GPU完整rollout
→ 完整PPO update
→ 小规模多GPU
→ 正式训练
```

代码修改后不得直接启动大规模长时间训练。

---

# 20. 推荐单元测试

至少包含：

```text
Observation shape test
finite-value test
reset-history test
coordinate transform test
candidate representation test
candidate permutation test
reward range test
collision-source test
checkpoint load/save test
PPO recompute consistency test
```

---

# 21. 训练评价指标

不得仅依赖：

```text
reward mean
policy loss
value loss
```

至少同时监控：

```text
success rate
failure rate
timeout rate
termination reason
episode length
episode reward components
entropy
KL
clip fraction
explained variance
collision rate
task-specific metrics
```

---

# 22. 训练性能指标

同时记录：

```text
environment FPS
rollout time
update time
sensor time
reward compute time
GPU utilization
GPU memory
```

模型性能比较应同时考虑样本效率与 wall-clock 成本。

---

# 23. 工程检查清单

## Environment

* [ ] global step 与 episode step 分离；
* [ ] 普通终止使用 partial reset；
* [ ] synthetic reset 独立处理；
* [ ] episode 内环境连续；
* [ ] collision authority 唯一；
* [ ] failure source 可观测。

## State

* [ ] shape / dtype / unit 明确；
* [ ] 坐标系明确；
* [ ] normalization 明确；
* [ ] history reset 正确；
* [ ] 动态任务包含时序信息；
* [ ] ego-motion 处理明确。

## Action

* [ ] action 物理语义明确；
* [ ] candidate action 对 Actor 可见；
* [ ] candidate representation 已验证；
* [ ] permutation test 通过；
* [ ] train / play action contract 一致。

## Reward

* [ ] 各 reward 项目标明确；
* [ ] 无明显重复处罚；
* [ ] dynamic risk 使用时空信息；
* [ ] dense reward episode sum 合理；
* [ ] timeout 与 success 区分；
* [ ] synthetic reset 不污染 reward；
* [ ] 所有 reward 有 finite check。

## Actor / Critic

* [ ] Actor 仅使用可部署信息；
* [ ] Critic 输入契约明确；
* [ ] shared / independent 模式明确；
* [ ] gradient sharing 明确；
* [ ] frozen layer 明确；
* [ ] BN / dropout 行为明确。

## PPO

* [ ] local/global env 数明确；
* [ ] rollout size 明确；
* [ ] minibatch 定义明确；
* [ ] KL / entropy / clip fraction 有记录；
* [ ] 多卡初始化一致。

## Distributed

* [ ] WORLD_SIZE / RANK / LOCAL_RANK 正确；
* [ ] env 数可被 world size 整除；
* [ ] rank seed 独立；
* [ ] rank 0 负责公共输出；
* [ ] NCCL 故障优先检查本地异常。

## Logging

* [ ] rate 分母明确；
* [ ] rollout / episode / curriculum 指标分离；
* [ ] step mean / episode sum 同时记录；
* [ ] TensorBoard smoothing 与 raw value 区分。

## Deployment

* [ ] checkpoint contract 完整；
* [ ] runtime dependency 完整；
* [ ] preprocessing 与训练一致；
* [ ] CUDA warm-up 通过；
* [ ] 至少完成一次完整 rollout 和 update；
* [ ] run directory 唯一。

---

# 24. 核心规则

1. 环境语义、State、Action、Reward 必须形成闭合契约。
2. 终止因素应对 Actor 可观测。
3. 动态任务应提供时序信息并处理 ego-motion。
4. Candidate Action 必须显式提供 candidate representation。
5. Reward 应主要评价 Action 质量。
6. 动态风险应采用时空一致的预测或 TTC。
7. Static 与 Dynamic risk 应避免重复计权。
8. Dense reward 应按 episode 累计量设计。
9. Collision、body size 和 safety margin 应分别定义。
10. Actor、Critic 和 Reward 的信息边界必须明确。
11. Shared / Independent Critic 和梯度路径必须显式定义。
12. 冻结策略必须精确到具体网络层和运行统计。
13. PPO 多卡参数必须区分 local 与 global 语义。
14. Rate 指标必须明确统计分母和时间窗口。
15. 大规模训练前必须完成小规模完整 PPO 验证。
16. Checkpoint 必须携带完整训练与模型 Contract。
17. 分布式通信异常应优先追踪最早本地错误。
18. 训练效果必须使用任务指标、episode 指标和优化指标联合判断。
