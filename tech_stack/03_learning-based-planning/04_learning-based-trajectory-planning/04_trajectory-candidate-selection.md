# 轨迹候选生成与选择

## 1. 概述

候选选择将连续规划分成两步：生成有限条可执行轨迹，再根据当前场景选择其中一条。生成器规定可选解的范围，评分器或强化学习策略决定实际选择；控制器将选中轨迹变成运动。

## 2. 基础与定义

### 2.1 候选集合

时刻 $t$ 的候选集合为 $\mathcal T_t=\{\tau_{t,1},\ldots,\tau_{t,K}\}$。每条轨迹带有几何、时域、导数与质量特征 $z_{t,i}$，有效标记为 $m_{t,i}$。候选可来自基元采样、传统优化或网络生成。

### 2.2 动作语义

策略动作 $a_t=i$ 表示选择当前集合的第 $i$ 条轨迹。其物理含义由本步 $z_{t,i}$ 确定。观测因此包含环境特征 $o_t$ 与候选集合，策略写为 $\pi_\theta(i\mid o_t,\mathcal T_t)$。

## 3. 核心原理

### 3.1 共享评分与有效掩码

对每条候选使用共享评分器 $l_{t,i}=f_\theta(o_t,z_{t,i})$，在有效集合 $\mathcal V_t=\{i:m_{t,i}=1\}$ 上归一化：

$$
\pi_\theta(i\mid o_t,\mathcal T_t)=
\begin{cases}
\dfrac{e^{l_{t,i}}}{\sum_{j\in\mathcal V_t}e^{l_{t,j}}},&i\in\mathcal V_t,\\
0,&i\notin\mathcal V_t.
\end{cases}
$$

当有效集合为空时，执行已定义的停止或恢复行为并记录事件；该情况不进入普通 softmax 采样。独立评分加对称聚合可以保持候选排列等变，候选间交互也可用满足相同对称性的网络实现。

<details class="kt-derivation" id="candidate-permutation">
<summary>候选排列等变性</summary>

对候选排列 $P$，若共享评分器和上下文聚合满足 $l(o,PZ)=Pl(o,Z)$，且掩码同步排列，则

$$
\operatorname{softmax}(Pl)=P\operatorname{softmax}(l).
$$

测试时随机重排候选与掩码，将输出概率逆排列后比较。最终被选轨迹的物理属性应一致；固定编号嵌入或未同步的缓存会破坏这一关系。

</details>

### 3.2 PPO 损失与梯度

rollout 保存当时的候选特征、掩码、动作和旧 log-prob。更新时在相同候选集合上计算

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid o_t,\mathcal T_t)}{\pi_{\mathrm{old}}(a_t\mid o_t,\mathcal T_t)},\qquad
L_{\mathrm{actor}}=-\mathbb E_t\!\left[\min(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)\right].
$$

固定生成器时，梯度只更新选择网络及其可训练编码器。若联合更新生成器，候选分布、动作含义和回放特征都会变化，需要单独定义联合目标和数据一致性；单纯重算候选可能改变概率比的条件事件。

### 3.3 动态碰撞评价

轨迹与障碍物预测在共同时间起点上计算净间距，独立记录静态和动态风险。所有候选使用同一碰撞几何与控制限制；候选时长不同时，将评价范围、执行窗口和时间代价明确分开。

### 3.4 候选质量与选择质量

设参考代价为 $C(\tau)$，单次选择遗憾为

$$
\mathcal R_t=C(\tau_{t,a_t})-\min_{i\in\mathcal V_t}C(\tau_{t,i}).
$$

它度量给定集合和参考代价下的选择差距。候选集合本身的不足，通过有效候选率、可达性和对更强生成器的比较评价。单步最优代价候选与长期导航最优决策分别分析。

## 4. 实现与应用

### 4.1 采样与执行

1. 从同一感知快照生成候选并构造特征。
2. 完成有限性、动力学和几何检查，形成有效掩码。
3. 采样候选编号，保存当时的概率与候选数据。
4. 执行约定窗口，记录实际执行轨迹和环境转移。
5. 按 PPO 流程更新，并在固定测试集评估生成与选择两部分。

### 4.2 诊断实验

比较随机有效选择、原生成器评分、学习选择器及参考代价选择。保持生成器、控制器、场景和动作周期一致。若安全层改写了动作，分别保存策略选择和实际执行结果，训练概率仍对应采样策略动作。

## 5. 分析与总结

有限候选降低决策搜索范围，性能同时受候选覆盖与选择能力限制。对候选数的扩展应同时检查推理时间、有效候选比例、熵和重复轨迹率。增加相似候选可能改变概率质量分配，却没有增加可执行方向。

## 6. 关联与参考

- [YOPO](./02_yopo.md)。
- [PPO](/knowledge-tree/learning-based-planning/rl-algorithms/model-free-rl/ppo/)。
- [导航评估](/knowledge-tree/simulation-evaluation/evaluation-methodology/navigation-metrics/)。
- [轨迹选择器项目](/knowledge-tree/projects/rl-trajectory-selector/)。
- [Deep Sets](https://arxiv.org/abs/1703.06114)。
