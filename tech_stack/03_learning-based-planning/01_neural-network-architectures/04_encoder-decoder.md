# 编码器与解码器

## 1. 概述

编码器将观测映射为潜在表示，解码器将表示映射为预测目标。重建或状态预测损失通过解码器回传到编码器；与策略等主任务共享编码器时，多个损失共同塑造表示。

## 2. 基础与定义

### 2.1 网络结构

给定输入 $x$，编码器与解码器分别为

$$
z=f_\theta(x),\qquad \hat y=g_\phi(z).
$$

自编码器取目标 $y=x$；状态重建可取训练期间可获得的真实状态 $s$。潜在变量 $z$ 的维数和目标共同决定需要保留的信息。

### 2.2 历史观测

部分可观测控制中，输入可由历史窗口 $H_t$ 或循环状态提供。策略使用当前观测和潜在表示：

$$
z_t=f_\theta(H_t),\qquad a_t\sim\pi_\psi(\cdot\mid o_t,z_t).
$$

部署时的编码输入必须能从实际传感器与历史记录获得。

## 3. 核心原理

### 3.1 重建损失

连续目标常用

$$
L_{\rm aux}=\frac12\mathbb E\|g_\phi(f_\theta(H))-s\|^2.
$$

目标可按尺度归一化或使用加权误差；概率预测可使用负对数似然，匹配相应噪声模型。

### 3.2 共享梯度

主任务损失记为 $L_{\rm main}$，联合目标为 $L=L_{\rm main}+\beta L_{\rm aux}$。共享编码器获得

$$
\nabla_\theta L=\nabla_\theta L_{\rm main}+\beta\nabla_\theta L_{\rm aux}.
$$

解码器由辅助目标训练；主任务头由主任务目标训练。若某条分支对 $z$ 停止梯度，该分支对编码器的贡献为零。
<details class="kt-derivation" id="encoder-gradient">
<summary>重建梯度传播</summary>

单样本平方损失令 $e=\hat y-s$，有

$$
\nabla_\phi\ell=\left(\frac{\partial g_\phi}{\partial\phi}\right)^\top e,
\qquad
\nabla_\theta\ell=
\left(\frac{\partial f_\theta}{\partial\theta}\right)^\top
\left(\frac{\partial g_\phi}{\partial z}\right)^\top e.
$$

冻结编码器后训练解码器，只改变表示的读取方式；联合训练则同时改变表示本身。

</details>

### 3.3 梯度相互作用

令共享参数上的两类梯度为 $g_m,g_a$。沿联合负梯度更新一步，对主任务的一阶变化为

$$
\Delta L_{\rm main}\approx-\eta\left(\|g_m\|^2+\beta g_m^\top g_a\right).
$$

梯度夹角、范数与权重共同决定辅助任务对当前主任务更新的影响。辅助目标应围绕控制相关信息选择，并通过消融实验检验。

### 3.4 训练与部署

仿真真值可以作为辅助监督目标，实际观测作为编码输入。若动作生成只依赖编码器与策略头，部署可移除解码器。显式状态估计、潜在特征和教师表示蒸馏对应不同的目标与梯度路径。

## 4. 实现与应用

先构造时间对齐的观测历史和目标，前向计算共享表示及各分支损失，再按总损失更新。以下示例以监督动作回归演示共享梯度，策略梯度训练时用对应策略损失替换主任务损失。

<details class="kt-derivation" id="encoder-training">
<summary>共享编码器训练实现</summary>

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
encoder_params = list(model.encoder.parameters())
g_main = torch.autograd.grad(loss_policy, encoder_params, retain_graph=True)
g_aux = torch.autograd.grad(loss_aux, encoder_params, retain_graph=True)
loss.backward()
optimizer.step()
```

</details>

检查各分支梯度、重建误差与主任务指标，并比较联合训练、冻结表示后重建及无辅助目标三种设置。

## 5. 分析与总结

辅助监督的作用由目标信息、共享位置与梯度路径决定。重建精度和控制收益需分别评价，历史窗口长度与噪声模型需保持训练部署一致。

## 6. 关联与参考

### 6.1 相关方法

| 方法 | 表示学习机制 |
| --- | --- |
| [UNREAL](https://arxiv.org/abs/1611.05397) | 辅助任务训练共享特征 |
| [RMA](https://arxiv.org/abs/2107.04034) | 历史适应模块拟合特权环境表示 |
| [Concurrent Training](https://arxiv.org/abs/2202.05481) | 联合训练状态估计与控制策略 |
| [DWL](https://roboticsproceedings.org/rss20/p058.html) | 去噪世界模型支持人形机器人运动 |
| [Vision-Driven Soccer](https://arxiv.org/abs/2511.03996) | 编码解码结构处理视觉噪声与检测失败 |

这些方法可按输入信息、监督目标、共享参数和部署模块比较。

- [循环神经网络与 LSTM](03_rnn-lstm.md)
- [时序与多模态融合](07_temporal-multimodal-fusion.md)
- [多 Critic](../02_rl-algorithms/01_rl-foundations/05_multi-critic.md)
