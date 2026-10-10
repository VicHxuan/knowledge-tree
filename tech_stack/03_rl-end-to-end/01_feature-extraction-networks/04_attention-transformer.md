# 1. 注意力机制与Transformer

## 1.1 数学定义构造

给定输入序列的查询矩阵 $Q \in \mathbb{R}^{L_q \times d_k}$、键矩阵 $K \in \mathbb{R}^{L_k \times d_k}$ 和值矩阵 $V \in \mathbb{R}^{L_k \times d_v}$，缩放点积注意力的数学定义为：

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

多头注意力机制通过多组不同的线性投影将 $Q, K, V$ 映射到多个子空间并行计算，随后拼接输出：

$$MultiHead(Q, K, V) = \text{Concat}(head_1, \dots, head_h)W^O$$
$$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

其中 $W_i^Q, W_i^K, W_i^V, W^O$ 均为可学习的权重矩阵。

依据 $Q, K, V$ 数据来源的不同，注意力分为两种核心形态：

1. **自注意力**：$Q, K, V$ 均来自同一个序列本身，用于提取序列内部的全局依赖关系。
2. **交叉注意力**：$Q$ 来自目标序列（如解码器特征），$K, V$ 来自源序列（如编码器输出），用于实现跨模态或跨序列的信息对齐与特征融合。

Transformer 架构由编码器与解码器堆叠而成，除注意力层外，还包含基于全连接的前馈神经网络与残差层归一化：

$$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

## 1.2 层级架构性质

1. **全局感受野**
   无论是距离多远的序列节点，在注意力矩阵 $QK^T$ 的内积计算中均能直接完成特征交互。任意两个位置的信息传递路径长度恒为 $O(1)$，彻底攻克了长序列信息衰减瓶颈。

2. **高度并行化**
   不同于循环神经网络必须依序等待上一时间步的隐状态，Transformer 的矩阵运算完全摆脱了时序依赖，使得整个序列能够在 GPU 上实现完全并行的特征计算。

3. **位置排列不变性**
   注意力机制本身对输入序列的顺序不敏感。为打破这种排列不变性，必须在底层输入中显式注入位置编码，使模型能够感知词元在序列中的绝对或相对空间位置。

## 1.3 代码实现

以下代码实现了统一的多头注意力模块（兼顾自注意力与交叉注意力），并以此构建完整的 Transformer 编码器-解码器架构。

### 1.3.1 PyTorch实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor = None, v: torch.Tensor = None, mask: torch.Tensor = None):
        # 兼容自注意力与交叉注意力
        if k is None: k = q
        if v is None: v = q
        
        batch_size = q.size(0)
        # 线性映射并拆分多头
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
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
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
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
            enc_out = layer(enc_out, mask=src_mask)
            
        dec_out = tgt
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)
            
        return dec_out
```

### 1.3.2 C++ 标准库实现

此处提炼核心矩阵乘法与注意力逻辑，采用二维数组模拟张量操作，展现底层计算本质。

```cpp
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

using Matrix = std::vector<std::vector<double>>;

class TransformerMath {
public:
    static Matrix matmul(const Matrix& A, const Matrix& B) {
        Matrix C(A.size(), std::vector<double>(B[0].size(), 0.0));
        for(size_t i=0; i<A.size(); ++i)
            for(size_t k=0; k<B.size(); ++k)
                for(size_t j=0; j<B[0].size(); ++j)
                    C[i][j] += A[i][k] * B[k][j];
        return C;
    }

    static Matrix transpose(const Matrix& A) {
        Matrix C(A[0].size(), std::vector<double>(A.size(), 0.0));
        for(size_t i=0; i<A.size(); ++i)
            for(size_t j=0; j<A[0].size(); ++j) C[j][i] = A[i][j];
        return C;
    }

    static void softmax(Matrix& A) {
        for(auto& row : A) {
            double max_val = *std::max_element(row.begin(), row.end());
            double sum = 0.0;
            for(double& val : row) { val = std::exp(val - max_val); sum += val; }
            for(double& val : row) val /= sum;
        }
    }
};

class SimpleAttention {
private:
    Matrix W_q, W_k, W_v; // 权重初始化省略以保持凝练

public:
    // k_in和v_in可与q_in相同(自注意力)或不同(交叉注意力)
    Matrix forward(const Matrix& q_in, const Matrix& k_in, const Matrix& v_in) {
        Matrix Q = TransformerMath::matmul(q_in, W_q);
        Matrix K = TransformerMath::matmul(k_in, W_k);
        Matrix V = TransformerMath::matmul(v_in, W_v);

        Matrix scores = TransformerMath::matmul(Q, TransformerMath::transpose(K));
        double scale = std::sqrt(K[0].size());
        for(auto& row : scores) for(double& val : row) val /= scale;

        TransformerMath::softmax(scores);
        return TransformerMath::matmul(scores, V);
    }
};

class SimpleTransformerBlock {
private:
    SimpleAttention self_attn, cross_attn;
    bool is_decoder;

public:
    SimpleTransformerBlock(bool decoder) : is_decoder(decoder) {}

    Matrix forward(Matrix src, Matrix tgt = {}) {
        // 自注意力 (编码器处理src，解码器处理tgt)
        Matrix& x = is_decoder ? tgt : src;
        Matrix self_out = self_attn.forward(x, x, x); 
        // 简化省略残差与LayerNorm
        
        if (is_decoder) {
            // 交叉注意力: Q来自tgt，K/V来自编码器特征src
            Matrix cross_out = cross_attn.forward(self_out, src, src);
            return cross_out; // 省略FFN
        }
        return self_out; // 省略FFN
    }
};
```

### 1.3.3 C++ Eigen库实现

依托 Eigen 高性能矩阵运算，通过逻辑循环切分矩阵特征维度，实现严格的多头注意力机制。

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

class TransformerEigen {
private:
    MultiHeadAttentionEigen enc_self_attn;
    MultiHeadAttentionEigen dec_self_attn;
    MultiHeadAttentionEigen cross_attn;
    
public:
    TransformerEigen(int d_model, int heads) 
        : enc_self_attn(d_model, heads), dec_self_attn(d_model, heads), cross_attn(d_model, heads) {}

    Eigen::MatrixXf forward(const Eigen::MatrixXf& src, const Eigen::MatrixXf& tgt) {
        // 编码器层: 自注意力
        Eigen::MatrixXf enc_out = src + enc_self_attn.forward(src, src, src);
        
        // 解码器层: 自注意力
        Eigen::MatrixXf dec_out = tgt + dec_self_attn.forward(tgt, tgt, tgt);
        
        // 解码器层: 交叉注意力 (Q来自tgt，K/V来自enc_out)
        dec_out = dec_out + cross_attn.forward(dec_out, enc_out, enc_out);
        
        return dec_out;
    }
};
```

---

# 2. Transformer 架构：从输入到最终输出

本节按照原始论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中的编码器—解码器 Transformer，说明一次完整前向计算如何从源序列输入得到目标序列输出。论文面向机器翻译等序列到序列任务，因此包含完整的 Encoder 和 Decoder；只需要特征提取时，通常仅保留 Encoder。

## 2.1 原论文的整体配置

原论文中的 Transformer Base 配置如下：

| 组件 | 原论文配置 |
| --- | ---: |
| Encoder 层数 | $N=6$ |
| Decoder 层数 | $N=6$ |
| 模型维度 | $d_{\text{model}}=512$ |
| 注意力头数 | $h=8$ |
| 每个头的 Query/Key 维度 | $d_k=64$ |
| 每个头的 Value 维度 | $d_v=64$ |
| FFN 隐藏维度 | $d_{\text{ff}}=2048$ |
| 激活函数 | ReLU |
| Dropout | $0.1$ |
| 归一化位置 | 残差相加后 LayerNorm，即 Post-LN |

完整的数据流为：

```text
源序列 tokens
    ↓ Token Embedding × √d_model
    ↓ 加入位置编码
    ↓ 6 层 Transformer Encoder
Encoder memory
    ↓ 作为每个 Decoder 层 Cross-Attention 的 K、V

右移后的目标序列 tokens
    ↓ Token Embedding × √d_model
    ↓ 加入位置编码
    ↓ Masked Self-Attention
    ↓ Cross-Attention：Q 来自 Decoder，K/V 来自 Encoder
    ↓ Position-wise FFN
    ↓ 堆叠 6 个 Decoder 层
Decoder hidden states
    ↓ 词表线性映射
    ↓ Softmax
每个位置的下一 token 概率
    ↓ 训练时并行计算；推理时自回归生成
最终目标序列
```

## 2.2 步骤一：源序列离散化与输入表示

设源序列经过分词后得到

$$
X=(x_1,x_2,\ldots,x_n),
$$

其中 $n$ 是源序列长度。每个 token $x_i$ 先通过可学习的 Embedding 矩阵映射到 $d_{\text{model}}$ 维：

$$
e_i
=
\sqrt{d_{\text{model}}}\,E_{\text{src}}[x_i],
\qquad
e_i\in\mathbb R^{d_{\text{model}}}.
$$

原论文将词向量乘以 $\sqrt{d_{\text{model}}}$，以使 Embedding 与位置编码处于合适的数值尺度。

自注意力本身不能区分 token 顺序，因此需要加入位置编码。原论文使用固定正弦—余弦位置编码：

$$
PE_{(pos,2i)}
=
\sin\!\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right),
$$

$$
PE_{(pos,2i+1)}
=
\cos\!\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right).
$$

最终 Encoder 输入为

$$
H^{(0)}
=
\operatorname{Dropout}
\left(
\sqrt{d_{\text{model}}}\,E_{\text{src}}[X]
+PE_{\text{src}}
\right),
$$

其形状为

$$
H^{(0)}\in\mathbb R^{n\times d_{\text{model}}}.
$$

批量训练时还会构造源序列 Padding Mask，使注意力不能读取补齐位置。

## 2.3 步骤二：Encoder 提取源序列上下文

Encoder 由 6 个结构相同但参数不共享的层堆叠而成。第 $l$ 层包含“多头自注意力”和“逐位置前馈网络”两个子层。

### 2.3.1 多头自注意力

对第 $l$ 层输入 $H^{(l-1)}$，每个注意力头分别做线性映射：

$$
Q_i=H^{(l-1)}W_i^Q,
\qquad
K_i=H^{(l-1)}W_i^K,
\qquad
V_i=H^{(l-1)}W_i^V.
$$

因为 $Q_i,K_i,V_i$ 都来自同一个序列，所以这是自注意力。第 $i$ 个头的注意力分数、权重和输出依次为

$$
S_i
=
\frac{Q_iK_i^T}{\sqrt{d_k}}+M_{\text{src}},
$$

$$
A_i=\operatorname{softmax}(S_i),
$$

$$
\operatorname{head}_i=A_iV_i.
$$

其中 $M_{\text{src}}$ 在 Padding 位置填入负无穷，使 Softmax 后对应权重为 0。8 个头并行计算后拼接并投影：

$$
\operatorname{MHA}(H^{(l-1)})
=
\operatorname{Concat}
\left(
\operatorname{head}_1,\ldots,\operatorname{head}_8
\right)W^O.
$$

多头机制允许不同子空间分别学习语义、位置和依赖关系。注意力输出经过 Dropout、残差连接和 LayerNorm：

$$
\widetilde H^{(l)}
=
\operatorname{LayerNorm}
\left(
H^{(l-1)}
+\operatorname{Dropout}
\left(
\operatorname{MHA}(H^{(l-1)})
\right)
\right).
$$

这里是原论文使用的 Post-LN；现代大模型中也常见先归一化再进入子层的 Pre-LN 变体。

### 2.3.2 Position-wise FFN

每个位置独立通过同一组两层全连接网络：

$$
\operatorname{FFN}(x)
=
\operatorname{ReLU}(xW_1+b_1)W_2+b_2,
$$

其中维度变化为

$$
512\rightarrow2048\rightarrow512.
$$

FFN 不在不同 token 之间交换信息，而是对每个 token 的通道特征进行非线性变换。随后再次执行残差连接和 LayerNorm：

$$
H^{(l)}
=
\operatorname{LayerNorm}
\left(
\widetilde H^{(l)}
+\operatorname{Dropout}
\left(
\operatorname{FFN}(\widetilde H^{(l)})
\right)
\right).
$$

经过 6 层 Encoder 后得到

$$
Z=H^{(6)}\in\mathbb R^{n\times512}.
$$

$Z$ 称为 Encoder memory：它保留源序列每个位置的上下文表示，并在后续所有 Decoder 层中作为交叉注意力的 Key 和 Value。

## 2.4 步骤三：构造 Decoder 输入

设目标序列为

$$
Y=(y_1,y_2,\ldots,y_m).
$$

训练时不能让位置 $t$ 直接看到真实的 $y_t$ 或未来 token，因此将目标序列右移一位，在最前面加入起始符：

$$
Y_{\text{in}}
=
(\langle\text{BOS}\rangle,y_1,y_2,\ldots,y_{m-1}).
$$

Decoder 输入同样由缩放后的 Embedding 与位置编码相加得到：

$$
G^{(0)}
=
\operatorname{Dropout}
\left(
\sqrt{d_{\text{model}}}\,E_{\text{tgt}}[Y_{\text{in}}]
+PE_{\text{tgt}}
\right).
$$

同时构造因果掩码

$$
M_{\text{causal}}(t,j)
=
\begin{cases}
0, & j\leq t,\\
-\infty, & j>t,
\end{cases}
$$

使第 $t$ 个位置只能注意自身及更早位置。这样训练时可以并行计算所有位置，同时保持自回归约束。

## 2.5 步骤四：Decoder 融合已生成内容与源序列

Decoder 同样堆叠 6 层。与 Encoder 相比，每层多出一个 Encoder-Decoder Cross-Attention 子层，因此一层 Decoder 包含三个子层。

### 2.5.1 Masked Multi-Head Self-Attention

对第 $l$ 层输入 $G^{(l-1)}$ 计算带因果掩码的自注意力：

$$
U^{(l)}
=
\operatorname{LayerNorm}
\left(
G^{(l-1)}
+\operatorname{Dropout}
\left(
\operatorname{MaskedMHA}
\left(G^{(l-1)},G^{(l-1)},G^{(l-1)}\right)
\right)
\right).
$$

这一步建模已经提供或已经生成的目标 token 之间的依赖关系。

### 2.5.2 Encoder-Decoder Cross-Attention

交叉注意力的数据来源不同：

$$
Q_i=U^{(l)}W_i^Q,
\qquad
K_i=ZW_i^K,
\qquad
V_i=ZW_i^V.
$$

Query 来自 Decoder，表示“当前目标位置需要查找什么”；Key 和 Value 来自 Encoder memory，表示“源序列能够提供什么信息”。因此

$$
C^{(l)}
=
\operatorname{LayerNorm}
\left(
U^{(l)}
+\operatorname{Dropout}
\left(
\operatorname{MHA}
\left(U^{(l)},Z,Z\right)
\right)
\right).
$$

该步骤让每个目标位置根据当前生成上下文，从整个源序列中选择相关信息。

### 2.5.3 Decoder FFN

交叉注意力输出再经过逐位置 FFN、残差连接和 LayerNorm：

$$
G^{(l)}
=
\operatorname{LayerNorm}
\left(
C^{(l)}
+\operatorname{Dropout}
\left(
\operatorname{FFN}(C^{(l)})
\right)
\right).
$$

经过 6 层后得到最终 Decoder 表示

$$
G=G^{(6)}\in\mathbb R^{m\times512}.
$$

## 2.6 步骤五：Linear、Softmax 与最终输出

Decoder 每个位置的隐藏向量 $g_t$ 通过词表线性层得到 Logits：

$$
z_t=g_tW_{\text{vocab}}+b_{\text{vocab}},
\qquad
z_t\in\mathbb R^{|\mathcal V|},
$$

再通过 Softmax 得到下一个 token 的条件概率：

$$
p_\theta(y_t\mid y_{<t},X)
=
\operatorname{softmax}(z_t).
$$

原论文在源 Embedding、目标 Embedding 和输出 Softmax 前的线性映射之间共享权重矩阵，以减少参数量并统一输入输出语义空间。

训练时采用 Teacher Forcing，所有目标位置可以并行计算，最小化负对数似然或交叉熵：

$$
\mathcal L_{\text{CE}}
=
-\sum_{t=1}^{m}
\log p_\theta(y_t\mid y_{<t},X).
$$

推理时没有真实目标序列，过程必须自回归执行：

1. 将 $\langle\text{BOS}\rangle$ 输入 Decoder。
2. 根据 Softmax 概率选择第一个 token $\hat y_1$。
3. 将 $\hat y_1$ 拼回 Decoder 输入，再预测 $\hat y_2$。
4. 重复上述过程，直到生成 $\langle\text{EOS}\rangle$ 或达到最大长度。

最终序列为

$$
\hat Y
=
(\hat y_1,\hat y_2,\ldots,\langle\text{EOS}\rangle).
$$

可以使用贪心搜索选择每一步概率最大的 token，也可以使用 Beam Search 保留多条候选序列。原论文的机器翻译实验使用 Beam Search 得到最终译文。

## 2.7 完整流程中的信息来源

| 计算位置 | Query 来源 | Key/Value 来源 | 是否需要掩码 | 主要作用 |
| --- | --- | --- | --- | --- |
| Encoder Self-Attention | 源序列 | 源序列 | Padding Mask | 建模源序列内部关系 |
| Decoder Masked Self-Attention | 右移后的目标序列 | 右移后的目标序列 | Padding Mask + Causal Mask | 建模已经生成的目标上下文 |
| Decoder Cross-Attention | Decoder 当前特征 | Encoder memory | 源序列 Padding Mask | 将目标位置与源序列信息对齐 |
| Linear + Softmax | Decoder 最终特征 | 词表参数 | 无 | 输出下一 token 的概率分布 |

因此，完整 Transformer 并不是单独一个 Attention 层，而是“输入表示、位置编码、Encoder 自注意力、Decoder 掩码自注意力、Encoder-Decoder 交叉注意力、FFN、残差归一化以及词表输出层”的组合。Encoder 负责理解输入，Decoder 负责结合输入语义和已生成内容，逐步产生最终输出。
