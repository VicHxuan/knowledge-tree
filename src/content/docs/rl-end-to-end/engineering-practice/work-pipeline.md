---
title: "小脑 Work Pipeline"
sidebar:
  order: 4
---

# 小脑-work-pipeline
> 在小脑方面，即训练model输出trajectory或者way point，已经完成了很多类似的工作内容，可以整理一份工作流程

## 1.任务设计

这部分主要是在抽象的高层含义上，设计自己的任务的goal(对应termination)，能使用的传感器信息(对应state)，输出的信息(对应action)，性能指标(对应reward)。这部分是粗略设计，主要关注方向，可实施性

通常来说，在“小脑”的范畴下，任务的目标是从A点安全快速的到达B点，要求输出可执行的运动指令，例如轨迹或者单步期望动作，提供给下游追踪-控制-执行环

## 2.仿真训练

当前的主流方案都是learning-based，至于ego，gcopter的方法更适合作为baseline，或者高精度高稳定性的场景使用。这些非learning的方法也不涉及仿真训练，不在此讨论

这里讨论的learning-based的训练通常包括2类：deep-learning(dl)和reinforcement-learning(rl)。二者也可以搭配使用，通常dl为快速训练粗略的模型，rl为后训练优化

### 2.1 dl

dl的需要手动收集数据-训练，因此最好有稳定的数据来源，例如从轻量级仿真器快速提取，例如yopo
获取数据后，按照 (输入-输出对) 计算loss，loss通常自行设计
这里按照处理方式不同，有一些具体划分
behavior cloning (expert learning) ：有明确的模仿对象，要求输出尽可能的拟合
learning (guidance learning) ： yopo的创新点，即在计算loss的时候，额外引入不属于输入量的信息，这点和强化学习的reward很像，本质上使用利用环境信息求loss，而不是无人机的信息

dl本身需要手动设定loss，特别的，这里的loss可以引入一些非输入的信息(这样可能就称为guidance learning)这点和强化学习的reward相似

### 2.2 rl

rl的好处在于数据无需手动收集，但是带来的弊端就是需要手动搭建收集数据的环境，以及reward相比于loss梯度的低指导效率带来的低数据效率
rl的分支主要是方法不同，例如on-policy的PPO或者off-policy的SAC，在这里不赘述，常规方案选取PPO进行并行训练