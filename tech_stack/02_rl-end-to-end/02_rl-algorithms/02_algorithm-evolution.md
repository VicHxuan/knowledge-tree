# 强化学习算法演进与梳理

这份总结基于提供的强化学习经典文献，按照时间线与算法演进逻辑进行梳理。全文去除了非必要的强调格式，并在每个算法的流程中统一了数学符号表达（如状态 $s_t$，动作 $a_t$，奖励 $r_t$，策略参数 $\theta$，价值参数 $\phi$，折扣因子 $\gamma$），以保证学术化与严谨性。

### 1. REINFORCE (1992)

《Simple statistical gradient-following algorithms for connectionist reinforcement learning》

算法流程

1. 智能体使用参数化策略 $\pi_\theta$ 与环境交互，采样获得一条完整的轨迹 $\tau = (s_0, a_0, r_0, \dots, s_T, a_T, r_T)$。
2. 对于轨迹中的每一个时间步 $t$，计算从当前时刻到回合结束的累积折扣回报 $R_t = \sum_{k=t}^T \gamma^{k-t} r_k$。
3. 利用对数导数技巧（Log-derivative trick）计算目标函数 $J(\theta)$ 关于策略参数的梯度估计值：
   $\nabla_\theta J(\theta) = \frac{1}{T} \sum_{t=0}^T R_t \nabla_\theta \ln \pi_\theta(a_t|s_t)$
4. 沿着梯度上升的方向更新策略网络参数：$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$。

独特工作
首次将对数导数技巧（亦称分数函数估计器）引入神经网络模型，证明了在不依赖环境动态转移模型的前提下，可通过经验采样的累积回报实现对策略梯度的无偏估计。

### 2. Policy Gradient Theorem (2000)

《Policy Gradient Methods for Reinforcement Learning with Function Approximation》

算法流程

1. 确立策略梯度定理，推导出策略目标函数梯度的一般形式：
   $\nabla_\theta J(\theta) = \mathbb{E}*{s \sim \rho^\pi, a \sim \pi*\theta} [\nabla_\theta \ln \pi_\theta(a|s) Q^\pi(s, a)]$
2. 引入参数化的价值函数逼近器 $Q_\phi(s, a) \approx Q^\pi(s, a)$ 替代未知的真实状态-动作价值。
3. 利用时序差分（TD）误差更新价值网络参数 $\phi$，最小化评估误差。
4. 在满足兼容性条件 $\nabla_\phi Q_\phi(s, a) = \nabla_\theta \ln \pi_\theta(a|s)$ 的前提下，使用近似价值函数更新策略参数：
   $\theta \leftarrow \theta + \alpha \nabla_\theta \ln \pi_\theta(a|s) Q_\phi(s, a)$

独特工作
奠定了现代 Actor-Critic 框架的数学基础。给出了严格的数学证明：当价值函数逼近器满足兼容性条件时，用其代替真实回报进行策略更新不会引入额外的偏差。

### 3. DQN - Deep Q-Network (2015)

《Human-level control through deep reinforcement learning》

算法流程

1. 智能体使用 $\epsilon$-贪婪策略基于当前动作价值网络 $Q_\phi(s, a)$ 与环境交互，将状态转移元组 $(s_t, a_t, r_t, s_{t+1}, d_t)$ 存入经验回放池 $\mathcal{D}$。
2. 在训练阶段，从 $\mathcal{D}$ 中随机均匀采样小批量（Mini-batch）数据。
3. 引入独立的目标网络 $Q_{\phi'}$ 计算时序差分目标：
   $y_t = r_t + \gamma (1 - d_t) \max_{a'} Q_{\phi'}(s_{t+1}, a')$
4. 最小化当前网络预测值与目标值的均方误差以更新参数 $\phi$：
   $L(\phi) = \mathbb{E}*{(s_t, a_t, r_t, s*{t+1}) \sim \mathcal{D}} [(y_t - Q_\phi(s_t, a_t))^2]$
5. 每隔固定的步数，将当前网络参数硬同步至目标网络：$\phi' \leftarrow \phi$。

独特工作
首次成功将深度神经网络与 Q-learning 结合。通过引入经验回放机制打破了序列数据的时序相关性，并利用目标网络截断了时序差分自举（Bootstrapping）过程中的误差传播，显著提升了训练稳定性。

### 4. TRPO - Trust Region Policy Optimization (2015)

《Trust Region Policy Optimization》

算法流程

1. 收集状态转移轨迹，并基于旧策略 $\pi_{\theta_{old}}$ 计算广义优势估计 $A_{\theta_{old}}(s, a)$。
2. 构造基于重要性采样的局部替代目标函数，并引入平均 KL 散度作为硬约束：
   $\max_\theta L(\theta) = \mathbb{E} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A_{\theta_{old}}(s, a) \right]$
   $\text{s.t.} \quad \mathbb{E} \left[ D_{KL}(\pi_{\theta_{old}}(\cdot|s) | \pi_\theta(\cdot|s)) \right] \le \delta$
3. 对目标函数进行一阶泰勒展开，对 KL 约束进行二阶泰勒近似。令 $g$ 为目标梯度，$H$ 为 KL 散度的海森矩阵（费雪信息矩阵）。
4. 由于直接求逆 $H$ 计算开销过大，使用共轭梯度法（Conjugate Gradient）近似求解自然梯度搜索方向 $x \approx H^{-1}g$。
5. 计算满足 KL 约束的最大理论步长 $\beta = \sqrt{2\delta / (x^T H x)}$。
6. 使用回溯线搜索（Backtracking Line Search）在此方向上寻找实际更新步长，确保目标函数提升且满足 KL 约束，最终更新 $\theta$。

独特工作
提出了信任区域优化框架，从理论上推导并保证了策略更新的单调递增性，解决了传统策略梯度算法中步长选择对收敛性极其敏感的难题。

### 5. A3C (2016)

《Asynchronous Methods for Deep Reinforcement Learning》

算法流程

1. 初始化全局策略网络参数 $\theta$ 和全局价值网络参数 $\phi$。开启多个并行的计算线程（Worker），每个 Worker 持有独立的网络副本 $\theta', \phi'$。
2. Worker 在各自独立的环境实例中执行 $n$ 步交互，计算 $n$ 步截断累积回报：
   $R_t = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V_{\phi'}(s_{t+n})$
3. 计算策略梯度（含熵正则化项）和价值函数梯度的累积量：
   $d\theta = \sum \left[ \nabla_{\theta'} \ln \pi_{\theta'}(a_t|s_t) (R_t - V_{\phi'}(s_t)) + \beta \nabla_{\theta'} H(\pi_{\theta'}) \right]$
   $d\phi = \sum \nabla_{\phi'} (R_t - V_{\phi'}(s_t))^2$
4. Worker 异步地将本地梯度 $d\theta, d\phi$ 应用于全局网络参数 $\theta, \phi$ 的更新中。
5. Worker 将全局最新参数同步至本地副本，进入下一轮采样。

独特工作
通过多线程异步并发与环境交互，自然地打破了训练数据的时序相关性，在舍弃经验回放池（降低内存开销）的前提下实现了同策略（On-policy）算法的稳定训练，并显著提升了训练效率。

### 6. PPO - Proximal Policy Optimization (2017)

《Proximal Policy Optimization Algorithms》

算法流程

1. 智能体在环境中收集交互轨迹，计算广义优势估计（GAE）$\hat{A}_t$。
2. 计算新旧策略的概率比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$。
3. 构造截断的替代目标函数，限制策略更新幅度：
   $L^{CLIP}(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$
4. 结合价值函数的均方误差损失 $L^{VF}(\phi)$ 和策略熵奖励 $S[\pi_\theta]$，构成总目标函数。
5. 舍弃复杂的二阶共轭梯度运算，直接使用 Adam 等一阶优化器，在同一批数据上进行多个周期的随机梯度下降更新 $\theta$ 和 $\phi$。

独特工作
提出了截断替代目标函数，用极为简化的截断（Clip）操作代替了 TRPO 中的 KL 散度二阶约束运算。在保持信任区域优化稳定性的同时，实现了计算效率的量级提升。

### 7. Rainbow (2017)

《Rainbow: Combining Improvements in Deep Reinforcement Learning》

算法流程

1. 结合分布式强化学习（Distributional RL），将价值网络输出从标量期望改为回报的概率分布预测 $Z_\phi(s, a)$。
2. 结合 Dueling 架构，将网络特征提取分为状态价值流和优势函数流，随后聚合。
3. 结合 Noisy Nets，将线性层权重加上参数化噪声 $y = (b + Wx) + (b_{noise} \odot \epsilon_b + W_{noise} \odot \epsilon_W x)$，取代启发式的 $\epsilon$-贪婪探索。
4. 在采样阶段，基于优先经验回放（PER）机制，根据当前预测分布与目标分布的 KL 散度设定采样权重。
5. 计算基于多步（Multi-step）和双网络（Double DQN）交叉验证的目标分布。
6. 计算预测分布与目标分布的交叉熵损失以更新参数 $\phi$。

独特工作
提供了严谨的消融实验证明，证实了 DQN 提出以来的六种主流改进（Double, Prioritized, Dueling, Multi-step, Distributional, Noisy）在数学机制与作用面上的正交性与互补性。其融合版本代表了离散动作空间中基于价值方法的性能巅峰。

### 8. SAC - Soft Actor-Critic (2018)

《Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning》

算法流程

1. 确立最大熵强化学习目标，最大化期望回报与策略熵的加权和：
   $J(\pi) = \mathbb{E}*{\tau \sim \pi} \left[ \sum*{t=0}^T \gamma^t (r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))) \right]$
2. 策略网络采用重参数化技巧（Reparameterization Trick）输出动作 $a_t = f_\theta(\epsilon_t; s_t)$ 以保证梯度可微。将交互数据存入回放池。
3. 从回放池采样，计算带有熵正则化项的软时序差分目标：
   $y_t = r_t + \gamma (1 - d_t) \left( \min_{i=1,2} Q_{\phi'*i}(s*{t+1}, a_{t+1}) - \alpha \ln \pi_\theta(a_{t+1}|s_{t+1}) \right)$
4. 最小化均方误差以更新两个独立的 Critic 网络参数 $\phi_1, \phi_2$（缓解高估偏差）。
5. 最小化以下目标以更新 Actor 网络参数 $\theta$（最大化软价值）：
   $J_\pi(\theta) = \mathbb{E}*{s_t, \epsilon_t} \left[ \alpha \ln \pi*\theta(f_\theta(\epsilon_t; s_t)|s_t) - \min_{i=1,2} Q_{\phi_i}(s_t, f_\theta(\epsilon_t; s_t)) \right]$
6. 通过最小化 $J(\alpha) = \mathbb{E} [-\alpha \ln \pi_\theta(a|s) - \alpha \bar{\mathcal{H}}]$ 自动调整温度系数 $\alpha$。

独特工作
提出了离策略（Off-policy）最大熵 Actor-Critic 框架，将重参数化与自动熵调节机制结合。在连续控制任务中，既达到了极高的样本利用效率，又有效避免了确定性策略梯度容易陷入局部最优的问题，展现出对超参数的强鲁棒性。

### 9. RPO - Robust Policy Optimization (2022)

《Robust Policy Optimization in Deep Reinforcement Learning》

算法流程

1. 继承 PPO 算法的整体架构，使用策略网络输出连续分布的均值 $\mu_\theta(s_t)$ 和标准差 $\sigma_\theta(s_t)$。
2. 在动作采样阶段，从预设的均匀分布 $U(-\alpha, \alpha)$ 中采样一个扰动向量 $z_t$。
3. 将策略网络输出的均值进行扰动：$\mu' = \mu_\theta(s_t) + z_t$。
4. 基于扰动后的新高斯分布 $\mathcal{N}(\mu', \sigma_\theta(s_t)^2)$ 进行实际的动作采样 $a_t$ 并与环境交互。
5. 利用收集到的轨迹计算广义优势估计（GAE），并严格按照 PPO 的截断替代目标函数 $L^{CLIP}(\theta)$ 进行参数的随机梯度下降更新。

独特工作
指出了 PPO 算法在连续控制中容易发生方差衰减（Variance Decay）与策略过早收敛的问题。引入了一种与状态无关的均匀扰动机制（Uniform Perturbation），在不增加数学复杂度的情况下，强制维持了探索过程中的策略随机性，提升了算法的渐进性能与鲁棒性。
