---
title: "TRPO (Trust Region Policy Optimization) 算法完整解析与代码实现"
sidebar:
  order: 3
---

TRPO（信任区域策略优化）是一种在理论上能够保证策略单调递增的强化学习算法。为了避免策略更新步长过大导致性能崩溃，TRPO 放弃了简单的一阶梯度下降，转而将策略更新建模为一个带 KL 散度硬约束的信任区域优化问题，并通过泰勒展开、共轭梯度法以及回溯线搜索等二阶优化技术进行求解。

---

## 1. 完整的数学公式与算法流程

### 步骤 1：构建替代目标函数 (Surrogate Objective)

在当前策略 $\pi_{\theta_{old}}$ 的基础上，我们希望寻找一个更优的策略 $\pi_\theta$。TRPO 利用重要性采样（Importance Sampling）构建了局部的替代目标函数：
$$ L(\theta) = \mathbb{E}*{s \sim \rho*{\theta_{old}}, a \sim \pi_{\theta_{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A_{\theta_{old}}(s, a) \right] $$
其中 $A_{\theta_{old}}(s, a)$ 是旧策略下的优势函数。

### 步骤 2：引入信任区域约束 (Trust Region Constraint)

为了保证最大化替代目标函数能够真实反映期望回报的提升，必须限制新旧策略分布之间的差异。TRPO 引入了平均 KL 散度的硬约束 $\delta$（例如 0.01）：
$$ \bar{D}*{KL}(\theta*{old}, \theta) = \mathbb{E}*{s \sim \rho*{\theta_{old}}} \left[ D_{KL}(\pi_{\theta_{old}}(\cdot|s) | \pi_\theta(\cdot|s)) \right] \le \delta $$
优化问题转化为：
$$ \max_\theta L(\theta) \quad \text{s.t.} \quad \bar{D}*{KL}(\theta*{old}, \theta) \le \delta $$

### 步骤 3：泰勒展开与二次近似

由于直接求解带有非线性约束的优化问题极其困难，TRPO 在 $\theta_{old}$ 处对目标函数进行一阶泰勒展开，对 KL 散度约束进行二阶泰勒展开：
$$ L(\theta) \approx g^T (\theta - \theta_{old}) $$
$$ \bar{D}*{KL}(\theta*{old}, \theta) \approx \frac{1}{2} (\theta - \theta_{old})^T H (\theta - \theta_{old}) \le \delta $$
其中：

* $g = \nabla_\theta L(\theta)|*{\theta*{old}}$ 是目标函数的策略梯度向量。
* $H = \nabla^2_\theta \bar{D}*{KL}|*{\theta_{old}}$ 是 KL 散度的海森矩阵（Hessian Matrix，即费雪信息矩阵 FIM）。

### 步骤 4：共轭梯度法求搜索方向 (Conjugate Gradient)

令搜索方向 $x = \theta - \theta_{old}$，由拉格朗日乘子法可知，解析解（自然梯度方向）为：
$$ x \propto H^{-1}g $$
在深度神经网络中，参数量极大，直接计算和求逆海森矩阵 $H$ 会导致内存和算力崩溃。TRPO 使用**共轭梯度法 (CG)**，通过多次迭代求解线性方程组 $Hx = g$ 来近似得到 $x$。CG 算法不需要直接生成 $H$，只需要计算海森矩阵与任意向量的乘积 (Hessian-Vector Product, $Hx$) 即可。

### 步骤 5：计算最大更新步长

确定了搜索方向 $x$ 后，需要计算在这个方向上能走的最大步长比例 $\beta$，使其刚好到达信任区域的边界：
$$ \frac{1}{2} (\beta x)^T H (\beta x) = \delta \implies \beta = \sqrt{\frac{2\delta}{x^T H x}} $$
理论上的全步长更新量为：
$$ \Delta \theta = \beta x $$

### 步骤 6：回溯线搜索 (Backtracking Line Search)

由于步骤 3 使用了泰勒局部近似，直接走全步长 $\Delta \theta$ 可能会导致目标函数下降，或者违反真实的 KL 散度约束。
TRPO 引入步长衰减系数 $\alpha \in (0, 1)$，依次尝试更新量 $\alpha^0 \Delta \theta, \alpha^1 \Delta \theta, \alpha^2 \Delta \theta \dots$，直到满足以下两个条件：

1. 替代目标函数确实提升了：$L(\theta_{old} + \alpha^i \Delta \theta) > 0$
2. 真实的 KL 散度满足约束：$\bar{D}*{KL}(\theta*{old}, \theta_{old} + \alpha^i \Delta \theta) \le \delta$
   找到合适的 $\alpha^i$ 后，正式更新 Actor 网络参数。Critic 网络则继续使用常规的一阶优化器（如 Adam）更新。

---

## 2. 核心代码架构实现

以下代码去除了环境交互与数据加载的冗余部分，专注于展示 TRPO 独有的二阶优化过程（共轭梯度法、HVP 计算与回溯线搜索），符合 Python 与 PyTorch 规范。

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

# =======================================
# 1. TRPO 核心数学工具函数
# =======================================

def flat_grad(grads):
    """将梯度元组展平为一维向量"""
    return torch.cat([g.contiguous().view(-1) for g in grads])

def get_flat_params(model):
    """获取模型所有参数的一维展开向量"""
    return torch.cat([p.contiguous().view(-1) for p in model.parameters()])

def set_flat_params(model, flat_params):
    """将一维参数向量重新重塑并赋值给模型"""
    prev_ind = 0
    for p in model.parameters():
        flat_size = int(np.prod(p.size()))
        p.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(p.size()))
        prev_ind += flat_size

def hessian_vector_product(actor, obs, vector, damping=1e-1):
    """
    计算海森矩阵与向量的乘积 Hx (Hessian-Vector Product)
    通过对 (grad(KL) * vector) 再次求导实现，避免显式实例化庞大的海森矩阵
    """
    mean, logstd = actor(obs)
    dist = Normal(mean, logstd.exp())
    
    with torch.no_grad():
        old_mean, old_logstd = actor(obs)
        old_dist = Normal(old_mean, old_logstd.exp())
        
    kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()
    
    # 第一次求导：KL 散度关于网络参数的梯度
    kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    flat_kl_grad = flat_grad(kl_grad)
    
    # 将梯度与目标向量做内积
    kl_grad_vector_product = (flat_kl_grad * vector).sum()
    
    # 第二次求导：内积关于网络参数的梯度，即为 Hx
    hvp = torch.autograd.grad(kl_grad_vector_product, actor.parameters())
    flat_hvp = flat_grad(hvp)
    
    # 增加阻尼项以提高数值稳定性
    return flat_hvp + vector * damping

def conjugate_gradient(actor, obs, b, nsteps=10, residual_tol=1e-10):
    """共轭梯度法 (CG)：迭代求解线性方程组 Hx = b"""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    
    for _ in range(nsteps):
        Hp = hessian_vector_product(actor, obs, p)
        alpha = rdotr / torch.dot(p, Hp)
        x += alpha * p
        r -= alpha * Hp
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x

# =======================================
# 2. 网络架构定义
# =======================================

class Actor(nn.Module):
    """策略网络：输出高斯分布的动作均值与独立对数标准差"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        return self.net(x), self.logstd.expand(x.size(0), -1)

class Critic(nn.Module):
    """价值网络：评估状态价值"""
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# =======================================
# 3. TRPO 核心算法类与更新逻辑
# =======================================

class TRPOAgent:
    def __init__(self, obs_dim, action_dim, device):
        self.device = device
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # TRPO 专属超参数
        self.max_kl = 0.01          # 信任区域硬约束 δ
        self.cg_iters = 10          # 共轭梯度法最大迭代次数
        self.line_search_coef = 0.5 # 回溯线搜索衰减率
        self.line_search_iters = 10 # 回溯线搜索最大尝试次数

    def compute_surrogate_loss(self, obs, actions, old_logprobs, advantages):
        """步骤1：计算重要性采样的替代目标函数"""
        mean, logstd = self.actor(obs)
        dist = Normal(mean, logstd.exp())
        logprobs = dist.log_prob(actions).sum(-1)
        ratio = torch.exp(logprobs - old_logprobs)
        return (ratio * advantages).mean()

    def trpo_update(self, obs, actions, old_logprobs, returns, advantages):
        """执行一次完整的 TRPO 参数更新"""
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # -----------------------------
        # 更新 Critic (标准均方误差与一阶优化)
        # -----------------------------
        for _ in range(10): # Critic 多次 Epoch 更新
            values = self.critic(obs).squeeze()
            critic_loss = F.mse_loss(values, returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # -----------------------------
        # 更新 Actor (二阶优化过程)
        # -----------------------------
        # 步骤 3: 计算当前策略梯度 g
        surrogate_loss = self.compute_surrogate_loss(obs, actions, old_logprobs, advantages)
        grads = torch.autograd.grad(surrogate_loss, self.actor.parameters())
        g = flat_grad(grads)
        
        # 步骤 4: 使用共轭梯度法计算搜索方向 x = H^-1 * g
        x = conjugate_gradient(self.actor, obs, g, nsteps=self.cg_iters)
        
        # 步骤 5: 计算最大步长 beta 与全步长 full_step
        Hx = hessian_vector_product(self.actor, obs, x)
        xHx = torch.dot(x, Hx)
        beta = torch.sqrt(2 * self.max_kl / (xHx + 1e-8))
        full_step = beta * x
        
        # 记录更新前的状态用于线搜索评估
        old_params = get_flat_params(self.actor)
        with torch.no_grad():
            old_mean, old_logstd = self.actor(obs)
            old_dist = Normal(old_mean, old_logstd.exp())
            old_surrogate = surrogate_loss.item()

        # 步骤 6: 回溯线搜索
        success = False
        for i in range(self.line_search_iters):
            alpha = self.line_search_coef ** i
            step = alpha * full_step
            
            # 临时赋予网络新参数
            set_flat_params(self.actor, old_params + step)
            
            # 评估新参数下的表现
            with torch.no_grad():
                new_surrogate = self.compute_surrogate_loss(obs, actions, old_logprobs, advantages).item()
                new_mean, new_logstd = self.actor(obs)
                new_dist = Normal(new_mean, new_logstd.exp())
                kl_div = torch.distributions.kl.kl_divergence(old_dist, new_dist).sum(-1).mean().item()
            
            # 条件检验：收益增加 且 KL 散度不超标
            if new_surrogate > old_surrogate and kl_div <= self.max_kl:
                success = True
                break
                
        # 如果线搜索失败，则放弃本次 Actor 更新，恢复原参数
        if not success:
            set_flat_params(self.actor, old_params)

# =======================================
# 4. 主循环调用抽象演示
# =======================================
def main_training_loop(agent, env_rollout_data):
    # 假设 rollout 数据已包含：状态, 动作, 旧对数概率, 目标回报, GAE 优势值
    obs, actions, logprobs, returns, advantages = env_rollout_data
    
    # 触发 TRPO 核心更新
    agent.trpo_update(obs, actions, logprobs, returns, advantages)
```
