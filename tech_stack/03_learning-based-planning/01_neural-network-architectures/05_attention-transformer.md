# 注意力机制与 Transformer

## 1. 概述

注意力根据查询与键的匹配程度聚合值向量。Transformer 将多头注意力、逐位置前馈网络和残差归一化组合为序列模型，任务损失通过注意力权重与特征投影共同更新参数。

## 2. 基础与定义

### 2.1 查询、键与值

设 $Q\in\mathbb R^{n_q\times d_k}$、$K\in\mathbb R^{n_k\times d_k}$、$V\in\mathbb R^{n_k\times d_v}$，缩放点积注意力为

$$
S=\frac{QK^\top}{\sqrt{d_k}}+M,\qquad
A=\operatorname{softmax}_{\rm row}(S),\qquad O=AV.
$$

掩码 $M$ 对可见位置取 0、屏蔽位置取 $-\infty$，每个有效查询至少保留一个键。

### 2.2 训练目标

自回归生成使用 $L=-\sum_t\log p_\theta(y_t\mid y_{<t},x)$。目标序列右移后作为解码输入，因果掩码限制每个位置可访问的前缀。其他任务可在编码表示上连接分类、回归或策略损失。

## 3. 核心原理

### 3.1 多头注意力

每个头独立投影 $Q_h=X_qW_h^Q$、$K_h=X_kW_h^K$、$V_h=X_vW_h^V$，输出拼接后映射：

$$
\operatorname{MHA}(X_q,X_k,X_v)=\operatorname{Concat}(O_1,\ldots,O_H)W^O.
$$

自注意力从同一序列生成三组向量；交叉注意力从目标序列生成查询，从来源序列生成键和值。

### 3.2 Transformer 层

原始后归一化结构为

$$
Y=\operatorname{LN}(X+\operatorname{MHA}(X,X,X)),
\quad Z=\operatorname{LN}(Y+\operatorname{FFN}(Y)).
$$

前馈网络逐位置计算 $\operatorname{FFN}(y)=W_2\sigma(W_1y+b_1)+b_2$。解码器在因果自注意力与前馈层之间加入交叉注意力；位置编码提供顺序信息。

### 3.3 注意力梯度

若输出梯度为 $G_O$，有 $G_V=A^\top G_O$、$G_A=G_OV^\top$。Softmax 的行内耦合进一步形成分数梯度，再回传到查询与键。
<details class="kt-derivation" id="attention-gradient">
<summary>注意力梯度推导</summary>

对一行 $a=\operatorname{softmax}(s)$，雅可比为 $\operatorname{diag}(a)-aa^\top$。因此

$$
G_{S,ij}=A_{ij}\left(G_{A,ij}-\sum_lA_{il}G_{A,il}\right),
$$

$$
G_Q=\frac{G_SK}{\sqrt{d_k}},\qquad
G_K=\frac{G_S^\top Q}{\sqrt{d_k}}.
$$

掩码作为固定可见性条件，屏蔽位置的注意力和对应分数梯度均为零。每个投影矩阵继续通过线性层链式法则获得梯度。

</details>

### 3.4 序列与计算量

标准稠密自注意力的分数矩阵包含 $n^2$ 个元素，计算量约为 $O(n^2d)$。局部窗口、稀疏连接与缓存改变计算和可访问信息，应与任务时序条件一致。
<details class="kt-derivation" id="transformer-shapes">
<summary>编码解码张量维数</summary>

取批次 $B$、来源长度 $n_s$、目标长度 $n_t$、模型维数 512、8 个头，每头维数为 64。

| 运算 | 张量形状 |
| --- | --- |
| 来源嵌入与位置编码 | $B\times n_s\times512$ |
| 编码器各头查询 | $B\times8\times n_s\times64$ |
| 编码器注意力矩阵 | $B\times8\times n_s\times n_s$ |
| 解码器因果注意力矩阵 | $B\times8\times n_t\times n_t$ |
| 交叉注意力矩阵 | $B\times8\times n_t\times n_s$ |
| 拼接多头输出 | $B\times n_t\times512$ |
| 词表投影 | $B\times n_t\times \lvert\mathcal V\rvert$ |

原始模型前馈层将 512 维扩展到 2048 维再映射回 512 维。训练并行处理右移后的目标序列；生成时逐步追加新 token。

</details>

## 4. 实现与应用

输入先完成嵌入和位置编码，再进入注意力层，最后连接任务输出头。以下 PyTorch 代码保留编码解码层堆叠，输入为已有位置编码的浮点特征；布尔掩码中 True 表示可见，可广播到批次、头、查询和键四个维度。

<details class="kt-derivation" id="transformer-pytorch">
<summary>PyTorch 编码解码实现</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if num_heads <= 0 or d_model <= 0 or d_model % num_heads:
            raise ValueError("d_model must be a positive multiple of num_heads")
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor = None, v: torch.Tensor = None, mask: torch.Tensor = None):
        # 兼容自注意力与交叉注意力
        if k is None: k = q
        if v is None: v = k

        batch_size = q.size(0)
        # 线性映射并拆分多头
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            allowed = torch.broadcast_to(mask.bool(), scores.shape)
            if not allowed.any(dim=-1).all():
                raise ValueError("each query must have at least one visible key")
            scores = scores.masked_fill(~allowed, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        # 拼接多头并输出映射
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(out)

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        if is_decoder:
            self.cross_attn = MultiHeadAttention(d_model, num_heads)
            self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x, enc_out=None, src_mask=None, tgt_mask=None):
        # 1. 多头自注意力
        self_mask = tgt_mask if self.is_decoder else src_mask
        x = self.norm1(x + self.self_attn(x, x, x, self_mask))
        # 2. 多头交叉注意力 (仅解码器)
        if self.is_decoder and enc_out is not None:
            x = self.norm2(x + self.cross_attn(q=x, k=enc_out, v=enc_out, mask=src_mask))
        # 3. 前馈网络
        x = self.norm_ffn(x + self.ffn(x))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_layers: int):
        super().__init__()
        self.encoder = nn.ModuleList([TransformerLayer(d_model, num_heads, False) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([TransformerLayer(d_model, num_heads, True) for _ in range(num_layers)])

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = src
        for layer in self.encoder:
            enc_out = layer(enc_out, src_mask=src_mask)

        dec_out = tgt
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        return dec_out
```

</details>

<details class="kt-derivation" id="attention-eigen">
<summary>Eigen 多头注意力</summary>

下例只计算无掩码的注意力前向过程。要求正模型维数能被正头数整除，键和值长度一致，各输入列数等于模型维数。

```cpp
#include <vector>
#include <cmath>
#include <Eigen/Dense>

class MultiHeadAttentionEigen {
private:
    int num_heads, d_k;
    Eigen::MatrixXf W_q, W_k, W_v, W_o;

public:
    MultiHeadAttentionEigen(int d_model, int heads) : num_heads(heads), d_k(d_model / heads) {
        W_q = Eigen::MatrixXf::Random(d_model, d_model);
        W_k = Eigen::MatrixXf::Random(d_model, d_model);
        W_v = Eigen::MatrixXf::Random(d_model, d_model);
        W_o = Eigen::MatrixXf::Random(d_model, d_model);
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& q_in, const Eigen::MatrixXf& k_in, const Eigen::MatrixXf& v_in) {
        int seq_q = q_in.rows();
        Eigen::MatrixXf Q_full = q_in * W_q;
        Eigen::MatrixXf K_full = k_in * W_k;
        Eigen::MatrixXf V_full = v_in * W_v;

        Eigen::MatrixXf out_full = Eigen::MatrixXf::Zero(seq_q, num_heads * d_k);

        // 遍历所有注意力头进行并行计算合并
        for (int h = 0; h < num_heads; ++h) {
            Eigen::MatrixXf Q = Q_full.block(0, h * d_k, seq_q, d_k);
            Eigen::MatrixXf K = K_full.block(0, h * d_k, k_in.rows(), d_k);
            Eigen::MatrixXf V = V_full.block(0, h * d_k, v_in.rows(), d_k);

            Eigen::MatrixXf scores = (Q * K.transpose()) / std::sqrt(d_k);

            // Softmax
            for (int i = 0; i < scores.rows(); ++i) {
                scores.row(i) = (scores.row(i).array() - scores.row(i).maxCoeff()).exp();
                scores.row(i) /= scores.row(i).sum();
            }

            out_full.block(0, h * d_k, seq_q, d_k) = scores * V;
        }

        return out_full * W_o;
    }
};
```

</details>

验证自注意力与交叉注意力形状、填充掩码和因果性，并检查被屏蔽位置变化是否影响有效输出。

## 5. 分析与总结

注意力提供内容相关的信息聚合，位置编码与掩码决定顺序和可见性。在线决策需要使用当时可获得的历史数据，训练与部署应保持相同的信息条件。

## 6. 关联与参考

- [Vaswani 等：Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch：MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [编码器与解码器](04_encoder-decoder.md)
