# Dyna：基于模型生成经验的强化学习架构

## 1. 概述

> **状态：草稿** — Knowledge Tree V2 规划章节，正文撰写中。

Dyna（Sutton，1990/1991）是 Model-Based RL 的原型架构：智能体在真实环境中交互的同时学习一个环境模型，并让模型**生成虚拟经验**，与真实经验一起更新值函数或策略。「直接强化学习 + 规划」在同一架构内统一，而规划的形式被简化为「用模型采样的转移做 Q-learning 式更新」。现代方法 MBPO 可视为这一思想在深度函数逼近下的直系后裔：模型生成合成经验，交给 model-free 学习器消化。

## 2. 基础与定义

### 2.1 直接与间接强化学习

- **直接 RL（direct RL）**：用真实交互的 $(s,a,r,s')$ 直接更新值函数/策略——即 model-free 的更新路径。
- **间接 RL（indirect RL / planning）**：先用数据学习模型，再由模型产生计算，反哺值函数/策略——Dyna 的核心主张是两者同时进行、互相加强。

### 2.2 模型学习

Dyna 的「模型」最一般地定义为预测环境行为的任意结构：

- **分布模型（distribution model）**：给出 $p(s'\mid s,a)$ 与 $r(s,a)$ 的完整分布——可做穷举/树搜索式规划；
- **样本模型（sample model）**：只能从 $\hat p(s'\mid s,a)$ 中采样出一对 $(\hat r,\hat s')$——Dyna 只需要这一种。

表格情形下模型学习退化为记录频次：$\text{Model}(s,a)\leftarrow (s',r)$。

### 2.3 真实经验与模拟经验

- 真实经验：环境交互产生，用于（a）更新模型，（b）直接 RL 更新；
- 模拟经验：从模型采样 $(s,a)\to(\hat r,\hat s')$，只用于（c）间接 RL（planning）更新。

## 3. 核心原理

### 3.1 Dyna-Q 算法流程

每个真实交互步之后：

1. 观测转移 $(s,a,r,s')$，**更新模型** $\text{Model}(s,a)\leftarrow(s',r)$；
2. 用该真实转移做**直接 Q 更新**：$Q(s,a)\leftarrow Q(s,a)+\alpha\,[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]$；
3. 重复 $n$ 次**规划步**：随机抽取一个历史 $(s,a)$，向模型查询 $(\hat r,\hat s')\leftarrow\text{Model}(s,a)$，用同样的 Q-learning 规则更新 $Q(s,a)$。

$n$ 是「每步规划量」超参数：$n=0$ 退化为纯 Q-learning，$n$ 越大对模型的依赖越重。

### 3.2 采样式规划

相比依赖分布模型的树搜索（穷举所有转移），Dyna 的规划是对状态转移的**随机采样**：不完整的模型也能产生有用的更新方向，计算量由 $n$ 线性控制。这使「planning」从显式求解变成了「额外的伪经验更新」，与现代深度 RL 的 mini-batch 更新天然兼容。

### 3.3 Dyna 与 MBPO

| 维度 | 经典 Dyna-Q | MBPO（2019） |
| :--- | :--- | :--- |
| 模型 | 表格样本模型 | 深度概率系综动力学模型 |
| rollout 长度 | 单步转移 | 短时域多步 rollout（分支） |
| 学习器 | Q-learning | SAC 等 model-free 学习器 |
| 起点分布 | 随机抽取历史状态 | 真实 replay buffer 采样 |

血缘关系是明确的：模型的作用始终是「生成经验给 model-free 学习器」，而不是通过模型反传梯度。

### 3.4 模型误差与数据复用

- 模型记错转移 → 规划更新把错误「钉」进值函数；Dyna-Q+ 通过对久未尝试的状态-动作加入探索奖励，缓解模型对环境变化的过时。
- 数据复用：同一份真实数据既训练模型又做直接更新，规划步还能以任意频率重复利用——样本效率的来源即在此，但复用率越高，模型偏差被放大的风险也越大。

## 4. 实现与应用

### 4.1 Dyna-Q 实现

```python
import random
from collections import defaultdict

def dyna_q(env, episodes=200, alpha=0.1, gamma=0.95, eps=0.1, n_plan=50):
    Q = defaultdict(float)                    # (s, a) -> value
    model = {}                                # (s, a) -> (r, s_next)
    seen = []                                 # 已经历过的 (s, a)

    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = random.choice(env.actions) if random.random() < eps else \
                max(env.actions, key=lambda a: Q[(s, a)])
            s2, r, done, _ = env.step(a)
            if (s, a) not in model:
                seen.append((s, a))
            model[(s, a)] = (r, s2)            # 1. 更新模型

            td = r + gamma * max(Q[(s2, a2)] for a2 in env.actions) - Q[(s, a)]
            Q[(s, a)] += alpha * td            # 2. 直接 RL

            for _ in range(n_plan):            # 3. 规划：模型生成虚拟经验
                s0, a0 = random.choice(seen)
                r0, s1 = model[(s0, a0)]
                td = r0 + gamma * max(Q[(s1, a2)] for a2 in env.actions) - Q[(s0, a0)]
                Q[(s0, a0)] += alpha * td
            s = s2
    return Q
```

（示意实现：`seen` 去重与边界处理从简；应用于确定性网格世界类任务。）

### 4.2 应用场景

- 小规模、模型易学的任务：网格世界、迷宫——教学中演示「规划加速学习」的标准载体；
- 大规模任务中 Dyna 思想的化身：experience replay 本身即「复用历史经验」，加模型采样后即 MBPO。

## 5. 分析与总结

- Dyna 的贡献是**架构思想**而非具体算法：把模型放进学习循环，以伪经验更新的形式做规划。
- 优点：样本效率显著高于纯 model-free（同样本量下值传播更快）；实现简单、对模型形式无要求。
- 局限：表格模型不可扩展；确定性单步模型无法表达随机环境；模型误差无不确定性度量，会被 Q 更新无差别吸收。
- 这些局限的逐项修复——概率模型（PETS）、系综与短 rollout（MBPO）、潜在世界模型与策略反传（Dreamer）——构成 Model-Based RL 分支的后续脉络。

## 6. 关联与参考

### 6.1 相关知识

- **先修知识**：[强化学习核心公式手册](../01_rl-foundations/01_reinforcement-learning-foundations.md)——Q-learning 与 TD 更新
- **本分支地基**：🚧 基于模型的强化学习基础（撰写中）——真实/想象 rollout 与模型不确定性
- **进阶阅读**：🚧 MBPO（撰写中）——Dyna 思想的深度学习规模化
- **相关方法**：🚧 值函数方法与 DQN（撰写中）——Dyna-Q 与 DQN 同属值方法家族

### 6.2 参考资料

1. Sutton, "Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming"（ICML 1990）。正式链接待补充。
2. Sutton, "Dyna, an Integrated Architecture for Learning, Planning, and Reacting"（SIGART Bulletin，1991）。正式链接待补充。
3. Sutton & Barto，《Reinforcement Learning: An Introduction》（第 2 版，2018）第 8 章——Dyna-Q 与 Dyna-Q+ 的标准叙述。正式链接待补充。
4. Janner et al., "When to Trust Your Model: Model-Based Policy Optimization"（NeurIPS 2019）——Dyna 血缘的现代版（MBPO）。正式链接待补充。
