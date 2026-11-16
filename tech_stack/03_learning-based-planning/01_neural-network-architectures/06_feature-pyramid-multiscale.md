# 特征金字塔与多尺度融合

## 1. 概述

特征金字塔将深层语义与浅层空间细节连接，在多个分辨率上产生可用于预测的特征。各尺度任务损失通过横向连接和自顶向下路径共同训练骨干网络。

## 2. 基础与定义

### 2.1 多尺度特征

骨干网络输出 $C_l\in\mathbb R^{B\times c_l\times H_l\times W_l}$，层级增加时分辨率通常降低。金字塔输出 $P_l$ 采用统一通道数，保留各尺度网格。

### 2.2 预测目标

多尺度预测头产生类别、位置或其他任务输出。训练目标为

$$
L=\sum_l\alpha_lL_l(P_l),
$$

各尺度的样本分配、有效位置与归一化决定损失权重的实际作用。

## 3. 核心原理

### 3.1 横向连接

先用 $1\times1$ 卷积将骨干通道统一为 $d$，记为 $S_l=\operatorname{Conv}_{1\times1}(C_l)$。从最高层开始建立中间融合特征

$$
F_L=S_L,\qquad
F_l=S_l+\operatorname{Resize}(F_{l+1};H_l,W_l).
$$

再以 $3\times3$ 卷积得到输出 $P_l=\operatorname{Conv}_{3\times3}(F_l)$。

### 3.2 梯度传播

加法融合使损失梯度同时进入横向分支与上采样分支。第 $l$ 层的共享参数既可能收到本尺度损失，也可能收到沿自顶向下路径传来的更细尺度损失。
<details class="kt-derivation" id="fpn-gradient">
<summary>融合梯度递推</summary>

令 $U_l$ 为固定尺寸的线性上采样算子，$F_l=S_l+U_lF_{l+1}$。来自 $F_l$ 的梯度 $G_l$ 对 $S_l$ 的贡献为 $G_l$，对 $F_{l+1}$ 的贡献为 $U_l^\top G_l$。多个下游分支的贡献相加，形成跨尺度监督。

</details>

### 3.3 空间对齐

融合前应匹配分辨率、采样网格和通道。奇数尺寸应直接指定目标特征大小，避免仅按倍数缩放造成一像素偏差。密集预测还需保持特征位置与原图坐标的映射一致。

## 4. 实现与应用

<details class="kt-derivation" id="fpn-pytorch">
<summary>PyTorch 特征金字塔</summary>

```python
import torch
from torch import nn
from torch.nn import functional as F

class FeaturePyramid(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.output = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in channels)

    def forward(self, features):
        # features 按高分辨率到低分辨率排列
        if len(features) != len(self.lateral):
            raise ValueError("feature count must match configured levels")
        results = [None] * len(features)
        top = None
        for i in range(len(features) - 1, -1, -1):
            current = self.lateral[i](features[i])
            if top is not None:
                current = current + F.interpolate(top, size=current.shape[-2:], mode="nearest")
            results[i] = self.output[i](current)
            top = current
        return results
```

</details>

验证不同尺度及奇数尺寸的形状、全部横向分支的梯度，以及多尺度损失对骨干的传播。

## 5. 分析与总结

多尺度结构适合目标大小变化明显的任务。高分辨率增加细节和计算成本，深层语义通过融合进入浅层预测；评价时应按对象尺度分别检查性能。

## 6. 关联与参考

- [Lin 等：Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
- [卷积神经网络](02_cnn.md)
- [时序与多模态融合](07_temporal-multimodal-fusion.md)
