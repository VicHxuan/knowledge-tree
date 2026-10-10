# PPO
## 1.全文逻辑梳理

这份文件会完整的讲解PPO的算法

不区分整体和局部重点流程，按照完整流程全部理清一遍

在强化学习中设置好每个env的state输入、action输出，actor和critic的网络架构

有 num_envs 个并行环境。每次 PPO iteration 中，每个环境采集 rollout 个 step，因此一轮共得到

$$ N_{\text{data}}=\text{num\_envs}\times\text{rollout} $$

条 transition。

每条 transition 通常包含

$$ (s_t,a_t,r_t,s_{t+1},d_t,\log\pi_{\text{old}}(a_t|s_t),V_{\text{old}}(s_t)) $$

等信息，并进一步计算 return 和 advantage。

更新阶段，将整批 rollout 数据划分为 num_minibatches 个 minibatch，每个 minibatch 含 minibatch_size 条数据。通常配置满足

$$ \text{num\_envs}\times\text{rollout} = \text{num\_minibatches}\times\text{minibatch\_size} $$

然后对整批 rollout 数据重复训练 epoch 遍。每一个 epoch 中，通常先重新 shuffle 打乱全部 rollout 数据，再划分 minibatch，并依次对每个 minibatch 执行一次梯度更新。

所有 epoch 完成后，得到更新后的 policy，然后使用新 policy 开始下一轮 rollout。


## 调参经验

1.rollout、discount_gamma、lamda
rollout表示每次更新前，每个env会收集多长的数据，即一段步长
discount_gamma是折扣率，影响boot-strap时候计算后续step的价值的权重
lamda是PPO里面GAE估计使用的参数

这些参数总体上体现了：我希望当前的step会受到多远的未来的评价，这里的“多远的未来”就受到这3个参数的影响

## tips
1.PPO中训练失败但loss曲线趋近于0分析：
当前发现这种情况限定于skrl算法库的PPO
注意到skrl的loss计算采用平均loss
