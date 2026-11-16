# 时序与多模态融合

## 1. 概述

时序与多模态融合将不同时间、不同传感器的观测变成共同表示，为预测或控制提供信息。结构设计需要同时确定时间对齐、坐标关系、缺失数据表示和训练梯度路径。

## 2. 基础与定义

### 2.1 观测与编码

第 $m$ 类观测记为 $o_t^{(m)}$，对应时间戳、有效标记 $v_t^{(m)}$ 与采样间隔。独立编码器产生

$$
z_t^{(m)}=f_{\theta_m}(o_t^{(m)}),\qquad
z_t=F_\phi(z_{\le t}^{(1)},\ldots,z_{\le t}^{(M)},v_{\le t}).
$$

几何传感器应先定义坐标变换；学习表示还可加入时间差与模态标识。

### 2.2 任务目标

输出头 $g_\psi(z_t)$ 由任务损失监督。可加入状态重建、跨模态对齐或预测损失，以约束共享表示中的信息。

## 3. 核心原理

### 3.1 融合层级

| 层级 | 结构 | 适用条件 |
| --- | --- | --- |
| 输入融合 | 对齐后拼接原始输入 | 时空网格和尺度容易统一 |
| 特征融合 | 独立编码后拼接、加权或注意力 | 模态结构与采样率不同 |
| 输出融合 | 合并预测分布或估计结果 | 各模块具有独立预测与置信信息 |

### 3.2 时间聚合

固定历史窗口可由 MLP 或时序卷积编码，流式状态可由 RNN 更新，注意力可根据内容选择历史位置。在线输出时刻 $t$ 的可见数据应以实际到达时间为准，并显式编码观测年龄。

### 3.3 缺失与可信度

有效标记参与门控或注意力掩码。加权融合可取

$$
z=\sum_m\alpha_m\tilde z^{(m)},\qquad
\alpha_m=\frac{v_m\exp(s_m)}{\sum_jv_j\exp(s_j)}.
$$

至少有一个有效模态时该式有定义；全缺失时使用明确的缺失表示、历史预测或回退输出。学习权重描述模型分配，概率置信度需要单独校准。

### 3.4 损失与梯度

总损失 $L=L_{\rm task}+\sum_j\beta_jL_{{\rm aux},j}$ 沿融合器进入各编码器：

$$
\nabla_{\theta_m}L=
\left(\frac{\partial z^{(m)}}{\partial\theta_m}\right)^\top
\left(\frac{\partial z}{\partial z^{(m)}}\right)^\top\nabla_zL
+\nabla_{\theta_m}L_{\rm aux,direct}.
$$

最后一项表示直接连接该模态编码器的辅助目标。共享位置、掩码和停止梯度共同决定实际更新路径。
<details class="kt-derivation" id="fusion-weight-gradient">
<summary>加权融合的梯度</summary>

若 $\alpha_m$ 固定，则 $\partial L/\partial\tilde z^{(m)}=\alpha_m\nabla_zL$。若权重由输入学习，还需累加权重分支的梯度。

对有效模态的分数有

$$
\frac{\partial z}{\partial s_m}=\alpha_m(\tilde z^{(m)}-z),\qquad
\frac{\partial L}{\partial s_m}=\alpha_m(\tilde z^{(m)}-z)^\top\nabla_zL.
$$

权重学习因此与当前任务目标有关，而非直接等同于传感器测量方差。

</details>

## 4. 实现与应用

按时间戳构造可见观测，统一几何坐标与数值尺度，编码各模态并应用有效掩码，再进行时序融合和任务预测。训练加入与部署一致的延迟、噪声和缺失模式。

消融实验分别移除单个模态、历史信息或辅助目标；同时检查未来信息泄漏、全缺失输入、时间戳扰动和跨场景泛化。

## 5. 分析与总结

融合收益来自互补信息，限制来自时空失配、模态质量变化与训练部署差异。结构选择应围绕任务需要的信息和实时计算预算展开。

## 6. 关联与参考

- [Baltrušaitis 等：Multimodal Machine Learning](https://arxiv.org/abs/1705.09406)
- [编码器与解码器](04_encoder-decoder.md)
- [循环神经网络与 LSTM](03_rnn-lstm.md)
- [注意力机制与 Transformer](05_attention-transformer.md)
