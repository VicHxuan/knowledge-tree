---
title: "ROS1 面试知识点整理"
sidebar:
  order: 2
---

# 1. 本人的 ROS1 经历

我的 ROS1 经历主要集中在**机器人算法部署、节点通信、状态机任务组织和系统调试**。

核心能力可以概括为：

```text
使用 ROS1 将感知、定位、规划、控制等算法模块组织成完整机器人系统。
```

主要做过的内容：

* 创建 ROS package 和 ROS node
* 编写 publisher / subscriber
* 设计 topic 输入输出接口
* 使用 launch 文件统一启动多个节点
* 使用 yaml 文件管理参数
* 使用状态机组织机器人任务流程
* 使用 ROS 工具进行系统调试
* 将算法模块接入仿真或实机系统

典型系统结构：

```text
传感器输入
    ↓
感知 / SLAM / 状态估计
    ↓
规划节点
    ↓
控制接口节点
    ↓
MAVROS / PX4 / 底层控制器
```

我对自己的 ROS1 定位：

```text
熟悉 ROS1 工程开发流程，能够完成机器人算法模块集成、节点通信设计、状态机任务管理、仿真或实机部署调试。
```

不刻意强调：

* ROS 底层通信源码
* ROS 内核机制
* 从零开发机器人中间件

更适合强调：

* 系统集成能力
* topic 数据流设计能力
* 状态机任务组织能力
* ROS 调试能力
* 仿真 / 实机部署经验

# 2. 坐标系、tf、tf2、hz 和时间同步

坐标系是机器人系统中不同模块理解空间数据的基础。

常见坐标系：

```text
map / world
    全局坐标系，用于地图、全局定位、全局规划。

odom
    局部连续坐标系，短时间平滑，但可能随时间漂移。

base_link
    机器人本体坐标系，固定在机器人机体上。

lidar / camera / imu
    传感器坐标系，表示传感器自身测量数据所在坐标系。
```

典型 tf tree：

```text
map / world
    ↓
odom
    ↓
base_link
    ↓
lidar / camera / imu
```

tf 的作用：

```text
维护不同坐标系之间的位置和姿态变换关系。
```

例如：

```text
LiDAR 点云在 lidar 坐标系下
    ↓
通过 tf 转换到 base_link
    ↓
再转换到 odom / map / world
```

tf 常见内容：

* 父坐标系
* 子坐标系
* 平移
* 旋转
* 时间戳

tf 和 tf2 的区别：

```text
tf 是早期版本；
tf2 是后续更完善的版本；
新代码一般更推荐 tf2。
```

tf2 相比 tf 的优势：

* 时间缓存机制更规范
* 类型支持更好
* API 更清晰
* 线程安全性更好
* 更适合新 ROS 工程

坐标系常见问题：

* `frame_id` 错误
* `child_frame_id` 错误
* tf tree 断裂
* 坐标轴方向不一致
* ENU / NED 坐标系混用
* 点云、轨迹、目标点不在同一坐标系
* 使用了错误时间戳下的 tf 变换

`hz` 表示 topic 发布频率。

常用命令：

```bash
rostopic hz /topic_name
```

典型频率关系：

```text
/odom              50 Hz
/lidar_points      10 Hz
/planner_cmd       20 Hz
/control_cmd       50 Hz
```

频率理解：

* 不同 topic 不一定需要同频
* 高频 topic 适合控制、里程计、IMU
* 低频 topic 适合点云、目标点、地图
* 关键是下游模块要知道自己使用的数据时间

时间同步解决的问题：

```text
多个 topic 的数据是否属于同一时刻，或者是否足够接近同一时刻。
```

常见同步方式：

```text
1. 使用最新消息
   简单，但时间一致性较弱。

2. 根据 header.stamp 查询 tf
   按消息真实时间做坐标变换。

3. message_filters 精确同步
   要求多个 topic 时间戳严格匹配。

4. message_filters 近似同步
   允许一定时间误差，工程中更常用。

5. rosbag 离线分析
   录制 topic 和 tf，检查频率、延迟、时间戳。
```

面试表达重点：

```text
坐标系问题不仅是 frame 名称问题，还包括时间戳、tf tree、topic 频率和模块间数据语义是否一致。
```

# 3. callback 和 spin

callback 是 ROS 中的回调函数。

作用：

```text
当订阅的 topic 收到消息后，执行对应处理函数。
```

典型写法：

```cpp
ros::Subscriber odom_sub = nh.subscribe("/odom", 10, odomCallback);
ros::Subscriber cloud_sub = nh.subscribe("/cloud", 10, cloudCallback);
ros::Subscriber goal_sub = nh.subscribe("/goal", 10, goalCallback);
```

callback 适合做：

* 接收消息
* 更新最新状态
* 缓存传感器数据
* 设置任务标志位
* 处理简单逻辑

callback 不适合做：

* 长时间阻塞
* 大量计算
* 复杂状态机全部逻辑
* 等待其他 topic
* while 死循环

`ros::spin()` 的作用：

```text
持续处理 callback，程序会阻塞在 spin 中。
```

适合场景：

```text
节点主要逻辑都由 callback 触发。
```

示例：

```cpp
ros::spin();
```

`ros::spinOnce()` 的作用：

```text
只处理一次当前回调队列，然后继续执行后续代码。
```

适合场景：

```text
节点需要按照固定频率运行主循环。
```

典型结构：

```cpp
ros::Rate rate(50);

while (ros::ok())
{
    ros::spinOnce();

    updateStateMachine();
    publishCommand();

    rate.sleep();
}
```

这种结构的含义：

```text
spinOnce()
    更新订阅数据

updateStateMachine()
    执行任务状态机

publishCommand()
    发布控制或规划输出

rate.sleep()
    控制循环频率
```

`queue_size` 的含义：

```text
消息处理不过来时，订阅队列最多缓存多少条消息。
```

queue_size 设置理解：

* 控制、里程计等实时数据：不宜过大
* 目标点、任务指令等低频数据：可以适当增大
* callback 太慢时，消息可能堆积或丢弃
* 实时系统更关注最新消息，而不是旧消息

单线程 callback 问题：

```text
默认情况下，如果一个 callback 执行很慢，其他 callback 会被阻塞。
```

多线程处理方式：

```cpp
ros::AsyncSpinner spinner(2);
spinner.start();
ros::waitForShutdown();
```

AsyncSpinner 适合：

* 多个高频 topic
* 某些 callback 计算量较大
* 不希望一个 callback 阻塞其他 callback

AsyncSpinner 需要注意：

* 多线程读写共享变量可能产生冲突
* 需要 mutex、数据拷贝或双缓冲
* 状态机变量要避免被多个 callback 同时修改

面试表达重点：

```text
callback 负责接收和更新数据；
spin / spinOnce 决定 callback 何时执行；
复杂任务逻辑最好放在主循环状态机中，而不是全部塞进 callback。
```

# 4. 项目的 ROS 介绍模板

介绍 ROS 项目时，可以按下面顺序组织：

```text
项目背景
    ↓
系统架构
    ↓
topic 数据流
    ↓
状态机设计
    ↓
调试方法
    ↓
遇到的问题
    ↓
个人负责内容
```

项目背景：

```text
这个项目主要目标是完成无人机 / 机器人自主导航、避障、轨迹跟踪或仿真实机部署。
系统需要接收传感器信息和机器人状态，根据目标点和环境信息生成运动指令，并通过底层控制接口执行。
```

ROS 系统架构：

```text
系统被拆成多个 ROS 节点：

传感器节点
定位 / SLAM 节点
感知处理节点
规划节点
控制接口节点
任务状态机节点
可视化 / 日志节点
```

Topic 数据流：

```text
输入：
    odom
    imu
    point cloud
    goal
    map
    robot state

输出：
    trajectory
    velocity command
    position command
    attitude setpoint
    debug marker
```

典型数据流：

```text
/odom + /cloud + /goal
        ↓
planner node
        ↓
/position_cmd 或 /trajectory
        ↓
controller node
        ↓
/mavros/setpoint_xxx
        ↓
PX4 / robot controller
```

状态机设计：

```text
INIT
    初始化参数和模块

WAIT_ODOM
    等待定位数据

TAKEOFF
    起飞或进入任务准备状态

WAIT_GOAL
    等待目标点

PLANNING
    执行路径或轨迹规划

TRACKING
    跟踪当前轨迹

REPLAN
    触发重规划

LAND
    降落

FAILSAFE
    异常保护
```

状态切换条件：

* 是否收到 odom
* 是否收到目标点
* 当前轨迹是否有效
* 是否到达目标点
* 是否检测到障碍物
* 规划是否失败
* 控制是否异常
* 定位是否丢失

调试方法：

```bash
rosnode list
rostopic list
rostopic echo /topic
rostopic hz /topic
rqt_graph
rviz
rosbag record
rosbag play
rosrun tf view_frames
rosrun tf tf_echo frame_a frame_b
```

调试顺序：

```text
1. 检查节点是否启动
2. 检查 topic 是否存在
3. 检查 topic 是否连接
4. 检查消息内容是否正确
5. 检查发布频率是否合理
6. 检查 frame_id 和 tf tree
7. 检查时间戳是否同步
8. 检查底层控制接口是否正常
```

常见问题：

```text
问题 1：规划正常，但机器人不动
排查：
    planner 是否发布指令
    controller 是否订阅到指令
    控制模式是否正确
    底层接口是否接收 setpoint

问题 2：轨迹方向异常
排查：
    坐标系是否一致
    ENU / NED 是否混用
    frame_id 是否正确
    tf tree 是否正确

问题 3：点云和机器人位置对不上
排查：
    LiDAR 外参是否正确
    点云时间戳是否正确
    tf 是否延迟或缺失

问题 4：控制响应滞后
排查：
    topic hz 是否过低
    callback 是否阻塞
    queue_size 是否过大
    是否处理了旧消息
```

个人负责内容：

```text
我主要负责 ROS 节点开发、topic 接口设计、状态机任务组织、launch 和 yaml 配置、算法模块接入，以及使用 ROS 工具完成系统联调和问题定位。
```

面试口述模板：

```text
我之前在无人机项目中使用 ROS1 做算法部署和系统集成。

系统中，定位模块发布 odometry，感知模块发布点云或障碍物信息，规划模块订阅当前状态、目标点和环境信息后生成轨迹或控制指令，控制接口节点再将规划结果转换为 MAVROS 或 PX4 可以接受的 setpoint。

任务流程上，我使用状态机管理初始化、等待定位、起飞、等待目标点、规划、轨迹跟踪、重规划、降落和异常保护等状态。

调试时，我会使用 rostopic 检查消息内容和频率，使用 rqt_graph 检查节点连接关系，使用 rviz 检查点云、轨迹、目标点和坐标系，使用 rosbag 录制关键数据进行离线分析。

我主要负责节点开发、topic 接口设计、状态机组织、launch / yaml 配置和系统联调。
```
