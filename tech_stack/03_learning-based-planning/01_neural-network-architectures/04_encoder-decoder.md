# Encoder–Decoder：基础架构、时序状态重建与辅助监督学习

## 1. 概述（Overview）

### 1. 概述

Encoder–Decoder（编码器–解码器）是一种通用的神经网络架构，其基本思想是：通过 Encoder 将输入数据转换为潜在表示（Latent Representation），再由 Decoder 将潜在表示映射到指定的输出空间。

经典应用包括数据重建、特征压缩、去噪、序列转换等。在强化学习中，还可以利用 Decoder 执行辅助状态重建任务，通过额外的监督信号改善 Encoder 的特征表示能力。

本文重点讨论：

- Encoder–Decoder 的基础结构与训练原理。
- 从输入重建到条件状态重建的扩展。
- 利用历史观测实现时序状态编码。
- 在强化学习中引入 Auxiliary Loss 的机制。
- Encoder 与 Decoder 的前向计算和反向梯度传播。
- 相关方法在机器人强化学习领域的发展。

需要区分的是，Encoder–Decoder 是一种**架构组织方式**，而 MLP、CNN、RNN、GRU、LSTM 和 Transformer 是可用于实现 Encoder 或 Decoder 的不同网络结构。

---

### 2. 学习动机

此前对 Encoder–Decoder 的理解主要局限于两种应用：

1. 作为完整的监督学习或重建网络，由 Encoder 提取特征，再由 Decoder 生成结果。
2. 仅保留 Encoder，将其作为独立的特征提取网络使用。

由此产生了一个认识上的局限：当只使用 Encoder 而不使用 Decoder 时，其作用似乎与普通的 MLP、CNN 等特征提取网络没有本质区别。

进一步学习后发现，这种认识忽略了 **Decoder 在训练阶段对 Encoder 表示学习的监督作用**。

Encoder 输出的 Latent 可以同时服务于多个网络分支。例如，在强化学习中：

- Actor 使用 Latent 进行动作决策。
- Decoder 使用同一个 Latent 重建真实或特权状态。
- Decoder 的重建误差产生辅助梯度，反向更新 Encoder。
- Encoder 因而不仅受到策略奖励的优化约束，还受到物理状态重建的监督约束。

这里的关键是：

**Latent 不仅是特征提取结果，也是连接主任务与辅助监督任务的共享表示。**

特别是在机器人强化学习中，仿真环境能够提供许多实际部署时不可直接观测的真实状态，因此可以利用这些状态训练 Decoder，并通过反向传播促使 Encoder 从有限、带噪声的观测中提取更有意义的信息。

需要强调，Decoder 并不是 Encoder 的必要组成部分。是否添加 Decoder，取决于是否存在有价值的重建目标，以及辅助监督能否改善主任务。辅助梯度也可能与主任务梯度发生冲突，因此需要通过实验验证。

---

## 2. 基础与定义（Foundations）

### 3. 基础架构

#### 3.1 经典 AutoEncoder

最经典的 Encoder–Decoder 形式为：

$$
x\xrightarrow{E_\phi}z\xrightarrow{D_\psi}\hat{x}
$$

其中：

| 符号 | 含义 |
|---|---|
| $x$ | 原始输入 |
| $E_\phi$ | Encoder，参数为 $\phi$ |
| $z$ | 潜在表示（Latent） |
| $D_\psi$ | Decoder，参数为 $\psi$ |
| $\hat{x}$ | 重建结果 |

前向计算过程为：

$$
z=E_\phi(x)
$$

$$
\hat{x}=D_\psi(z)
$$

采用均方重建误差时，损失函数可以表示为：

$$
\mathcal L_{\mathrm{recon}}
=
\|x-\hat{x}\|_2^2
$$

Encoder 将原始输入映射到潜在空间，Decoder 再从潜在空间重建原始输入。

从监督学习的角度理解，它仍然符合：

$$
\text{Input}\rightarrow\text{Network}\rightarrow\text{Prediction}\rightarrow\text{Loss}
$$

但与常见监督学习不同，经典 AutoEncoder 的监督目标就是输入本身：

$$
y=x
$$

因此，这是一种通过自重建目标学习数据表示的方法。

从功能上，可以将 Encoder 理解为特征提取部分，将 Decoder 理解为根据特征完成重建任务的输出部分。

不过需要注意：这种功能划分并不意味着一般监督网络都必须拆成 Encoder 和 Decoder。普通的特征提取主干加预测头，也可以完成类似的计算。

#### 3.2 梯度反向传播

训练过程中，重建损失通过 Decoder、Latent 逐层传播到 Encoder。

梯度路径为：

```text
                 Forward
    x ──> Encoder ──> z ──> Decoder ──> x_hat
                                              │
                                              ▼
                                       Reconstruction
                                           Loss
                                              │
                 Backward                     ▼
    Encoder <──── Latent <──── Decoder <──── Gradient
```

对于 Encoder 参数 $\phi$，根据链式法则：

$$
\frac{\partial\mathcal L_{\mathrm{recon}}}{\partial\phi}
=
\frac{\partial\mathcal L_{\mathrm{recon}}}{\partial\hat{x}}
\frac{\partial\hat{x}}{\partial z}
\frac{\partial z}{\partial\phi}
$$

因此，尽管重建误差是在 Decoder 的输出端计算的，Encoder 仍然能够接收到有效的梯度。

**Decoder Loss 反向训练 Encoder 并不是新机制，而是经典 Encoder–Decoder 架构本身的基础训练方式。**

后续辅助监督方法的扩展，主要体现在重建目标、共享网络结构和训练策略上。

---

## 3. 核心原理（Core Principles）

### 4. 条件重建与状态估计

#### 4.1 从输入重建到目标状态重建

经典 AutoEncoder 的目标是：

$$
D_\psi(E_\phi(x))\approx x
$$

但实际任务中，输入往往只是对真实系统的不完整观测。

例如，在机器人应用中，我们希望获取某个动态障碍物的真实运动状态，但实际只能通过深度相机等传感器获取间接、带噪声的观测。

因此，可以将 Encoder–Decoder 扩展为：

$$
D_\psi(E_\phi(H_t))\approx s_t^{GT}
$$

其中：

- $H_t$：机器人历史观测。
- $E_\phi$：历史信息 Encoder。
- $z_t$：历史观测的潜在表示。
- $D_\psi$：状态重建 Decoder。
- $s_t^{GT}$：由仿真器提供的真实或特权状态。

具体为：

$$
z_t=E_\phi(H_t)
$$

$$
\hat{s}_t=D_\psi(z_t)
$$

$$
\mathcal L_{\mathrm{recon}}
=
\|\hat{s}_t-s_t^{GT}\|_2^2
$$

这里与经典 AutoEncoder 最重要的区别是：

**Decoder 的监督目标不再必须等于 Encoder 的输入。**

输入和输出可以属于不同的物理空间，甚至具有完全不同的维度。

#### 4.2 应用于机器人状态估计

例如：

| Encoder 输入 | Decoder 监督目标 |
|---|---|
| 带噪声的球位置历史 | 球的真实位置、速度 |
| 历史深度图 | 动态障碍物的运动状态 |
| IMU、关节状态历史 | 机体真实线速度 |
| 历史动作与状态变化 | 动力学相关参数 |

这种架构的价值在于：

仿真训练时可以利用 Ground Truth 构造监督信号，而实际部署时通过 Encoder 从可用观测中学习近似的隐藏状态表示。

需要注意，能够提供 Ground Truth 并不意味着这些状态一定能从历史观测中恢复。重建目标的选择应考虑可观测性、历史窗口长度及任务相关性。

---

### 5. 时序处理与 History Encoder

#### 5.1 为什么需要历史观测？

机器人在某个时刻的单帧观测通常不足以描述完整的环境状态。

例如，只通过单帧深度图，通常难以直接获取视野中动态障碍物的准确运动速度。

一个自然的思路是：

**连续积累多帧观测，再利用它们随时间变化的信息推断隐藏状态。**

设当前观测为 $o_t$，历史窗口长度为 $K$：

$$
H_t=
[o_{t-K},o_{t-K+1},\ldots,o_{t-1}]
$$

通过时序 Encoder：

$$
z_t=E_\phi(H_t)
$$

使 Latent 包含历史运动信息。

这里的历史输入可以是原始传感器数据，也可以是已经经过特征提取的状态或视觉特征。

#### 5.2 常见时序 Encoder

| 方法 | 基本原理 | 主要特点 |
|---|---|---|
| History MLP | 固定时间窗口展开后输入 MLP | 实现简单，计算并行度高 |
| RNN | 递归更新隐藏状态 | 可保持跨时间步的记忆 |
| GRU/LSTM | 带门控机制的递归状态更新 | 更适合长期时序依赖 |
| Transformer | 通过 Attention 建模不同时间 Token 的关系 | 表达能力强，计算代价可能更高 |

需要强调，这些方法是不同的时序建模结构，并不是按照 Encoder–Decoder → RNN → GRU → Transformer 这样的顺序派生出来的。

它们均可以被用于实现 Encoder，某些情况下也可以用于实现 Decoder。

具体网络结构见：

`/home/zyd/hjr_ws/knowledge-tree/tech_stack/03_learning-based-planning/01_neural-network-architectures/03_rnn-lstm.md`

本文不展开介绍各类时序网络的内部结构，而重点关注时序 Encoder 如何与 Decoder 共同训练。

#### 5.3 完整时序流程

以固定历史窗口为例。

在时刻 $t$：

$$
H_t=[o_{t-K},\ldots,o_{t-1}]
$$

执行：

$$
z_t=E_\phi(H_t)
$$

Actor 接收当前观测与历史 Latent：

$$
a_t=\pi_\theta(o_t,z_t)
$$

执行动作后，进入下一时刻。

历史窗口更新：

$$
H_{t+1}=[o_{t-K+1},\ldots,o_t]
$$

完整闭环为：

```text
                 时刻 t
                    │
                    ▼
            获取当前观测 o_t
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       │
读取历史 Buffer H_t             │
        │                       │
        ▼                       │
      Encoder                   │
        │                       │
        ▼                       │
     Latent z_t                 │
        │                       │
        └───────────┬───────────┘
                    ▼
                   Actor
                    │
                    ▼
                  Action
                    │
                    ▼
                Environment
                    │
                    ▼
           更新历史 Buffer
                    │
                    ▼
                 时刻 t+1
```

对于 History MLP，每个决策时刻通常重新计算历史窗口的特征。

对于 GRU/LSTM，可以通过递归隐藏状态维护历史信息，但需要额外处理 Episode Reset、Hidden State 初始化和反向传播截断等问题。

还需要区分：

- **控制时间步：** Actor 生成新动作的频率。
- **观测时间步：** 传感器产生新数据的频率。
- **历史时间跨度：** Encoder 能访问的历史信息覆盖多长时间。

这些时间尺度不一定相同，尤其是在多频率传感器融合任务中。

---

### 6. 强化学习中的辅助监督

#### 6.1 核心思想

这是本文最重要的部分，也是此次学习的主要动机。

在强化学习中，可以让 Encoder 同时服务于 Actor 和 Decoder。

结构为：

```text
                 History H_t
                      │
                      ▼
                    Encoder
                      │
                      ▼
                   Latent z_t
                      │
           ┌──────────┴──────────┐
           │                     │
           ▼                     ▼
         Actor                 Decoder
           ▲                     │
           │                     ▼
    Current Obs o_t         Predicted State
           │                     │
           ▼                     ▼
         Action             Reconstruction
           │                   Loss
           ▼                     │
       PPO Loss                  │
           │                     │
           └──────────┬──────────┘
                      │
                      ▼
              Update Encoder
```

数学表达为：

$$
z_t=E_\phi(H_t)
$$

Actor 输出：

$$
a_t\sim\pi_\theta(\cdot\mid o_t,z_t)
$$

Decoder 输出：

$$
\hat{s}_t=D_\psi(z_t)
$$

总训练目标可以抽象成：

$$
\mathcal L
=
\mathcal L_{\mathrm{policy}}
+
\lambda_{\mathrm{aux}}
\mathcal L_{\mathrm{aux}}
$$

其中：

- $\mathcal L_{\mathrm{policy}}$：策略优化目标，例如 PPO Loss。
- $\mathcal L_{\mathrm{aux}}$：状态重建等辅助任务损失。
- $\lambda_{\mathrm{aux}}$：辅助损失权重。

这里暂时省略 PPO 中的 Value Loss、Entropy Bonus 等其他损失项，以便集中分析 Encoder 的训练机制。

#### 6.2 Latent 作为共享表示

从整个架构来看，Latent 实际上是一个共享的信息通道：

$$
H_t\xrightarrow{Encoder}z_t
$$

不同任务基于同一个 Latent 执行不同操作：

$$
\begin{aligned}
z_t &\rightarrow \text{Actor}\rightarrow a_t\\
z_t &\rightarrow \text{Decoder}\rightarrow\hat{s}_t
\end{aligned}
$$

它的核心意义是：

**不需要让 Decoder 的预测结果显式输入 Actor，也可以利用 Decoder 的监督信号改善 Actor 所依赖的 Encoder 表示。**

这与显式状态估计存在区别。

显式状态估计：

$$
H_t\xrightarrow{Estimator}\hat{s}_t
$$

$$
[o_t,\hat{s}_t]\xrightarrow{Actor}a_t
$$

隐式 Latent 辅助监督：

$$
H_t\xrightarrow{Encoder}z_t
$$

$$
\begin{cases}
[o_t,z_t]\xrightarrow{Actor}a_t\\
z_t\xrightarrow{Decoder}\hat{s}_t
\end{cases}
$$

后者中，Decoder 更像一个训练时的监督头，而不是 Actor 必须经过的前向计算模块。

#### 6.3 梯度传播路径

由于 Actor 和 Decoder 共享 Encoder，可以分别定义两个梯度：

$$
g_{\mathrm{policy}}
=
\nabla_\phi\mathcal L_{\mathrm{policy}}
$$

$$
g_{\mathrm{aux}}
=
\nabla_\phi\mathcal L_{\mathrm{aux}}
$$

联合训练时：

$$
\nabla_\phi\mathcal L
=
g_{\mathrm{policy}}
+
\lambda_{\mathrm{aux}}g_{\mathrm{aux}}
$$

因此，Encoder 接收两种不同来源的监督：

1. 策略梯度：鼓励 Latent 有助于获得更高任务回报。
2. 辅助梯度：鼓励 Latent 保留能够恢复目标状态的信息。

从表示学习的角度看，辅助重建任务相当于为 Latent 加入了额外的学习约束。

不过 Decoder 不会自动保证 Latent 中每个维度都具有明确的物理含义。Latent 仍然可能是混合、非线性的抽象表示。

#### 6.4 梯度冲突

两个训练目标不一定完全一致。

当：

$$
g_{\mathrm{policy}}^\top g_{\mathrm{aux}}<0
$$

说明两个梯度方向存在冲突。

可以通过梯度余弦相似度衡量：

$$
\cos(g_{\mathrm{policy}},g_{\mathrm{aux}})
=
\frac{
g_{\mathrm{policy}}^\top g_{\mathrm{aux}}
}{
\|g_{\mathrm{policy}}\|
\|g_{\mathrm{aux}}\|
}
$$

因此，增加 Auxiliary Loss 并不一定提升主任务性能。

值得比较的训练方式包括：

| 训练方式 | PPO 更新 Encoder | Auxiliary 更新 Encoder |
|---|---|---|
| 仅 PPO | 是 | 否 |
| 联合训练 | 是 | 是 |
| PPO 对 Encoder 截断梯度 | 否 | 是 |
| Auxiliary 对 Encoder 截断梯度 | 是 | 否 |
| 分阶段训练 | 依具体阶段决定 | 依具体阶段决定 |

这几个方案可以通过 `detach()`、冻结参数、分阶段优化等方法实现。

辅助监督方法真正值得研究的内容，不只是增加一个 Decoder，而是**如何设计监督目标、损失权重及梯度更新机制，使共享表示真正服务于主任务**。

#### 6.5 训练与部署的区别

对于只作为辅助监督使用的 Decoder：

训练阶段：

$$
\text{Encoder}+\text{Actor}+\text{Decoder}
$$

部署阶段：

$$
\text{Encoder}+\text{Actor}
$$

Decoder 不再需要运行。

但如果 Actor 直接依赖 Decoder 输出的显式状态，那么 Decoder 就属于部署必需模块，不能直接移除。

因此，“Decoder 仅训练时使用”是一种具体架构选择，而不是所有 Encoder–Decoder 方法的统一要求。

---

## 4. 实现与应用（Implementation & Applications）

### 7. PyTorch 基础实现

为了理解梯度路径，可以先实现一个最小化示例。

其中：

- Encoder 使用 MLP 编码历史观测。
- Actor 使用当前观测与 Latent 预测动作。
- Decoder 根据 Latent 预测真实状态。
- 两类 Loss 均可更新 Encoder。

这里使用监督动作损失模拟主任务损失，便于理解梯度机制；正式 PPO 实现应替换为 PPO 的策略损失。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoderPolicy(nn.Module):
    def __init__(self, obs_dim, history_len, latent_dim, state_dim, action_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim * history_len, 128),
            nn.ELU(),
            nn.Linear(128, latent_dim),
        )

        self.actor = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, state_dim),
        )

    def forward(self, history, obs):
        # history: [B, T, D]
        # obs:     [B, D]

        h = history.flatten(start_dim=1)
        z = self.encoder(h)

        actor_input = torch.cat([obs, z], dim=-1)
        action = self.actor(actor_input)

        state_hat = self.decoder(z)

        return action, state_hat


# 构造示例数据
B, T, D = 32, 10, 9

model = EncoderDecoderPolicy(
    obs_dim=D,
    history_len=T,
    latent_dim=64,
    state_dim=6,
    action_dim=4,
)

history = torch.randn(B, T, D)
obs = torch.randn(B, D)

state_gt = torch.randn(B, 6)
action_target = torch.randn(B, 4)

action, state_hat = model(history, obs)

# 主任务损失：此处使用监督回归演示梯度
loss_policy = F.mse_loss(action, action_target)

# 辅助状态重建损失
loss_aux = F.mse_loss(state_hat, state_gt)

# 联合损失
loss = loss_policy + 1.0 * loss_aux

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这段代码可以进一步用于验证两种 Loss 对 Encoder 的梯度。

例如，使用 `torch.autograd.grad()` 分别计算：

```python
params = list(model.encoder.parameters())

g_policy = torch.autograd.grad(
    loss_policy,
    params,
    retain_graph=True,
)

g_aux = torch.autograd.grad(
    loss_aux,
    params,
    retain_graph=True,
)
```

需要注意，上述两次 `autograd.grad()` 应在原计算图释放之前调用；如果要在同一训练步中观察梯度，通常应放在 `loss.backward()` 之前。

**建议完成的基础实验：**

- 仅使用 `loss_policy` 更新网络。
- 仅使用 `loss_aux` 更新网络。
- 联合使用两个 Loss。
- 调整辅助损失权重。
- 使用 `detach()` 控制不同分支是否更新 Encoder。
- 观察不同情况下的梯度范数及训练损失。

这里首先验证梯度路径，而不是要求随机生成的数据获得有意义的策略性能。后续应使用有明确输入与目标关系的时序数据，再比较不同训练方式的预测效果。

---

## 5. 分析与总结（Analysis & Summary）

### 9. 技术演进总结

从上述论文中，可以归纳几类不同的训练思路。

| 代表方法 | 主要训练方式 | 监督信息 | 核心作用 |
|---|---|---|---|
| AutoEncoder | 重建学习 | 原始输入 | 学习压缩表示 |
| UNREAL | 主任务 + 辅助任务 | 像素变化、奖励等 | 丰富共享表示 |
| RMA | 分阶段训练 | 特权 Latent | 历史适应与蒸馏 |
| Concurrent Training | 策略与状态估计协同训练 | 真实物理状态 | 显式状态估计 |
| DWL | 时序编码与重建学习 | 真实状态 | 去噪历史表示 |
| Soccer | PPO + 辅助重建 | 真实球与动力学状态 | 感知鲁棒性与隐状态推断 |

从研究思路上，可以看到三个逐渐深入的问题：

**第一：Encoder 应该学习什么？**

从最初保留原始输入信息，到学习与具体任务相关的特征。

**第二：监督信号从哪里来？**

除了主任务的 Reward，还可以利用仿真器的 Ground Truth、专家数据或其他辅助任务构造额外监督。

**第三：这些监督信号如何共同优化 Encoder？**

不同 Loss 可以共同更新共享参数，但也可能产生梯度冲突。因此需要研究 Loss 权重、梯度截断、分阶段训练以及多任务优化等技术。

---

### 10. 总结与思考

通过本文的学习，可以得到以下认识。

#### 10.1 Encoder–Decoder 的本质

Encoder–Decoder 并不仅仅是一种输入压缩与重建网络。

更一般地，它是一种通过 Latent 连接输入表示与目标任务的架构组织方式。

Decoder 的目标可以是原始输入、物理状态、未来观测或其他任务相关信息。

#### 10.2 Decoder 的训练作用

当 Decoder 用于辅助监督时，其主要价值可能不是部署阶段的输出，而是通过额外的 Loss 约束 Encoder 的内部表示。

尤其在强化学习中：

$$
\text{Policy Loss}
+
\text{Auxiliary Loss}
\rightarrow
\text{Shared Encoder}
$$

使 Encoder 同时学习任务相关信息与辅助目标信息。

#### 10.3 训练与部署不必完全一致

训练阶段可以利用仿真器提供的特权状态，而部署阶段只保留能够使用实际观测的网络。

因此，训练阶段存在的 Decoder 并不一定需要部署。

#### 10.4 辅助监督不是越多越好

一个辅助任务能否改善主任务，取决于：

- 重建目标是否与任务相关。
- 目标状态是否能够从输入推断。
- Encoder 是否具有足够的表示容量。
- 辅助 Loss 与主任务梯度是否兼容。
- 辅助监督的权重是否合理。

因此，辅助监督的设计应当通过消融实验验证，而不能只根据 Reconstruction Loss 的下降判断其有效性。

#### 10.5 后续研究方向

下一步值得深入学习的是：

**Auxiliary-Supervised Representation Learning for Reinforcement Learning**

重点关注：

- 历史 Encoder 的状态表示能力。
- 特权状态重建与 Latent 学习。
- PPO 与 Auxiliary Loss 的梯度关系。
- Joint Training、Stop-Gradient 与 Two-Stage Training 的区别。
- 辅助监督对动态环境泛化能力的影响。

最终目标并不是让 Decoder 重建尽可能多的真实状态，而是利用合适的监督信号，使 Encoder 学到对下游决策真正有用的表示。

---

## 6. 关联与参考（Related Topics & References）

### 8. 相关文献与技术发展

这里主要关注辅助任务、特权信息、状态估计及 Encoder–Decoder 在强化学习中的应用。

这些研究可以组成一条学习主线，但不应直接视为严格的技术继承关系。

#### 8.1 经典 AutoEncoder

**Hinton & Salakhutdinov, 2006**

*Reducing the Dimensionality of Data with Neural Networks*

**核心思想：**

通过 Encoder 将高维数据压缩到低维 Latent，再通过 Decoder 重建输入，利用重建损失训练整个网络。

重点理解：

- 输入与重建目标的关系。
- Latent 作为信息压缩表示。
- Reconstruction Loss 对 Encoder 的梯度传播。
- 为什么重建任务可以用于表示学习。

这一阶段主要建立 Encoder–Decoder 的基础概念。

#### 8.2 UNREAL（2016）

**Reinforcement Learning with Unsupervised Auxiliary Tasks**

Jaderberg et al., 2016

论文：[arXiv:1611.05397](https://arxiv.org/abs/1611.05397)

**核心思想：通过辅助任务改善强化学习的共享特征表示。**

UNREAL 不属于 Soccer 式的物理状态重建 Encoder–Decoder，但它展示了如何通过额外学习任务训练共享网络。

主要辅助任务：

- Pixel Control：学习动作对视觉区域变化的影响。
- Reward Prediction：预测奖励。
- Value Function Replay：利用过去经验进行额外的价值学习。

这类任务可以在环境奖励稀疏时提供额外的学习信号。

**学习重点：**

为什么一个辅助任务即使不直接参与动作生成，也可能通过改善共享特征表示来提升主策略？

**实践：**

实现一个共享 Encoder，同时连接 Actor 和 Reward Prediction Head，观察不同 Loss 是否都能更新 Encoder。

#### 8.3 RMA（2021）

**RMA: Rapid Motor Adaptation for Legged Robots**

Kumar et al., 2021

论文：[arXiv:2107.04034](https://arxiv.org/abs/2107.04034)

**核心思想：使用特权信息学习 Latent，再通过历史观测预测 Latent，实现快速环境适应。**

RMA 采用分阶段训练。

第一阶段：

$$
z_t=\mu(e_t)
$$

$$
a_t=\pi(o_t,z_t)
$$

其中 $e_t$ 包含训练时可获得的环境特权参数，例如摩擦、质量等。

环境编码器与基础策略通过强化学习共同训练。

第二阶段：

$$
\hat z_t=\phi(H_t)
$$

适应模块根据历史观测预测第一阶段得到的 Latent，基础策略在该阶段保持固定。

部署时使用适应模块输出的 $\hat z_t$。

**学习重点：**

- 特权信息与真实可观测信息的区别。
- Teacher Latent 与 Student Latent。
- 分阶段训练与联合训练的区别。
- 历史适应模块与策略网络的不同更新频率。

RMA 的适应模块约 10 Hz，基础策略约 100 Hz，是一个具有参考价值的异步频率设计。

**实践：**

实现 Teacher Encoder + Student History Encoder，用 MSE 监督 Student 拟合 Teacher Latent。

#### 8.4 Concurrent Training（2022）

**Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion**

Ji et al., IEEE RA-L 2022

论文：[arXiv:2202.05481](https://arxiv.org/abs/2202.05481)

**核心思想：将状态估计与控制策略放在同一训练框架中。**

状态估计模块根据观测输出明确的物理状态，例如机体线速度、足部高度和接触概率。

策略利用这些估计信息进行动作决策。

学习时需要区分两种设计：

显式状态估计：

$$
H_t\xrightarrow{Estimator}\hat{s}_t
$$

$$
[o_t,\hat{s}_t]\xrightarrow{Actor}a_t
$$

隐式 Latent：

$$
H_t\xrightarrow{Encoder}z_t
$$

$$
[o_t,z_t]\xrightarrow{Actor}a_t
$$

前者更易于解释，但依赖预先选择的状态变量。

后者可以学习更灵活的表示，但其物理意义未必明确。

**学习重点：**

- 状态估计器与控制策略的接口。
- 显式状态估计和隐式特征表示的差异。
- 状态估计误差对控制性能的影响。

**实践：**

比较使用显式估计速度和使用隐式 Latent 的 Actor。

#### 8.5 DWL（2024）

**Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning**

Gu et al., RSS 2024

论文：[RSS Proceedings](https://roboticsproceedings.org/rss20/p058.html)

**核心思想：利用时序 Encoder 与状态重建监督，学习对控制有用的去噪世界状态表示。**

DWL 使用 GRU 处理带噪声的历史观测，并通过 Decoder 重建相关真实状态。

其重点在于通过状态重建监督改善历史信息的表征，而不仅是学习某个动作输出。

相较于 RMA，DWL 更强调历史时序表示与状态重建任务之间的关联。

**学习重点：**

- GRU Hidden State 如何更新。
- 历史观测的噪声如何处理。
- Decoder 重建哪些真实状态。
- Reconstruction Loss 如何影响 Encoder。
- Actor 如何利用 Latent。
- 训练和部署阶段保留哪些网络模块。

建议重点阅读论文 Figure 3 及相关方法章节，厘清网络连接和梯度路径。

**实践：**

使用 GRU 替换 History MLP，对比两种 Encoder 在历史状态重建任务中的表现。

#### 8.6 Soccer（2026）

**Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots**

Wang et al., 2026；arXiv 初稿发表于 2025 年

论文：[arXiv:2511.03996](https://arxiv.org/abs/2511.03996)

项目：[Humanoid Kick](https://humanoid-kick.github.io/)

**核心思想：在视觉感知不可靠的条件下，利用历史编码与辅助状态重建，实现感知与运动控制的协同学习。**

该论文采用：

- 50 帧历史观测，约 1 秒。
- MLP History Encoder。
- 64 维 Latent。
- Actor 接收当前观测与历史 Latent。
- Decoder 重建真实球位置、运动状态及部分动力学信息。
- PPO + Multi-Critic + AMP 完成控制策略训练。

训练阶段：

$$
H_t\xrightarrow{Encoder}z_t
$$

$$
\begin{aligned}
[o_t,z_t]&\xrightarrow{Actor}a_t\\
z_t&\xrightarrow{Decoder}\hat{s}_t
\end{aligned}
$$

Decoder 的重建目标由仿真器提供，辅助监督用于改善 Encoder 的表示。

部署时保留 Encoder 和 Actor，去除 Decoder。

与 DWL 相比，Soccer 更加突出真实视觉感知中的噪声、延迟、低频更新和检测失败问题。

**重点阅读：**

| 位置 | 内容 |
|---|---|
| Figure 8，第 14 页 | 训练与部署整体架构 |
| Section 4.2，第 15 页 | 时序输入、Encoder–Decoder 与 PPO |
| Appendix Table 1，第 22 页 | 网络层数和超参数 |
| Appendix Table 3，第 23 页 | Actor、Critic、Decoder 使用的状态 |

**实践：**

比较以下方案：

1. 仅 PPO，不增加 Decoder。
2. PPO + Decoder 联合训练。
3. 先完成 PPO 训练，再冻结 Encoder，仅训练 Decoder。

论文观察到，第三种方式的事后状态估计无法达到联合训练时的去噪效果。

其启发在于：

**Decoder 的价值不只是学习读取已有 Latent，而是在联合训练时主动塑造 Encoder 的 Latent。**

---

### 参考文献

1. Hinton, G. E., & Salakhutdinov, R. R. (2006). *Reducing the Dimensionality of Data with Neural Networks*. Science.
2. Jaderberg, M., et al. (2016). *Reinforcement Learning with Unsupervised Auxiliary Tasks*. [arXiv](https://arxiv.org/abs/1611.05397)
3. Kumar, A., et al. (2021). *RMA: Rapid Motor Adaptation for Legged Robots*. [arXiv](https://arxiv.org/abs/2107.04034)
4. Ji, Y., et al. (2022). *Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion*. [arXiv](https://arxiv.org/abs/2202.05481)
5. Gu, X., et al. (2024). *Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning*. [RSS](https://roboticsproceedings.org/rss20/p058.html)
6. Wang, Y., et al. (2026). *Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots*. [arXiv](https://arxiv.org/abs/2511.03996)
