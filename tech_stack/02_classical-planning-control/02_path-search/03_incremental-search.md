# 增量搜索与路径修复

## 1. 概述

增量搜索复用上次搜索结果，在边代价变化后只修复受影响区域。LPA* 以局部一致性组织传播，D* Lite 将这一思想用于移动起点的重规划。

## 2. 基础与定义

### 2.1 代价状态

LPA* 保存已传播值 $g(v)$ 和一步前瞻值：

$$
rhs(s)=0,\qquad rhs(v)=\min_{u\in\operatorname{Pred}(v)}[g(u)+c(u,v)],\quad v\ne s.
$$

$g=rhs$ 称为局部一致；不一致节点进入优先队列。

## 3. 核心原理

### 3.1 优先级与修复

LPA* 按字典序键

$$
k(v)=\left[\min(g(v),rhs(v))+h(v,goal),\ \min(g(v),rhs(v))\right]
$$

处理节点。$g>rhs$ 时接受更优前瞻值；$g<rhs$ 时撤销过低旧值并更新受影响后继。

<details class="kt-derivation" id="lpa-repair">
<summary>LPA* 更新流程</summary>

```text
初始化：所有 g 与 rhs 为无穷，起点 rhs 为 0
更新节点：重新计算 rhs，撤销旧队列项，不一致时按新键入队
修复：弹出最小有效键
  若 g > rhs：令 g = rhs，更新后继
  否则：令 g = 无穷，更新自身及后继
停止：目标局部一致，且最小队列键不优于目标键
边变化：修改边权，更新受影响终点，再执行修复
```

实现可用版本号或不可变键快照识别陈旧队列项。键比较采用确定的字典序，避免浮点容差破坏堆所需的严格弱序。

</details>

### 3.2 D* Lite

D* Lite 从目标反向维护代价，前瞻值使用后继节点。起点移动后，以启发式和累计偏移 $k_m$ 调整键，复用既有代价信息。

### 3.3 D* 的关系

原始 D* 通过代价升高与降低传播修复路径；LPA* 和 D* Lite 使用 $g/rhs$ 形式表达不一致。它们共享增量修复目标，状态和队列规则分别定义。

## 4. 实现与应用

### 4.1 增量验证

依次测试边代价增加、降低、断路与恢复，以及起点移动。每次修改后与从头运行 Dijkstra 的结果对照；同时记录重用节点数和实际扩展量。

## 5. 分析与总结

增量收益取决于变化范围和旧解的可复用程度。边权更新、依赖传播与队列失效处理必须共同保持一致。

## 6. 关联与参考

[Koenig 与 Likhachev：D* Lite](https://idm-lab.org/bib/abstracts/papers/aaai02b.pdf)、[Lifelong Planning A*](https://idm-lab.org/bib/abstracts/papers/aij04.pdf)。

关联：[图搜索](./01_heuristic-search.md)、[重规划](../03_trajectory-optimization/10_replanning-and-trajectory-continuity.md)。
