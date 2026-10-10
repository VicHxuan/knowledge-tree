---
title: "采样法搜索（RRT 系列）"
sidebar:
  order: 2
---

# 1. RRT (快速扩展随机树, Rapidly-exploring Random Tree)

## 1.1 基本介绍

RRT 算法由 Steven M. LaValle 在 1998 年提出，是一种用于处理高维连续空间和带有复杂运动学约束的单查询路径规划算法。与传统的在网格上“逐格寻找”不同，RRT 采用“撒点”的思想。它以起点作为根节点，每次在空间中随机生成一个采样点，然后在现有树中找到距离该点最近的节点，并向该采样点方向“生长”固定步长，生成一个新节点。如此反复，树的枝干会快速向未知的广阔空间探索（具有天然的 Voronoi 偏向性），直到某根枝干触碰到终点区域。

## 1.2 伪代码流程图

```text
Initialize Tree T with StartNode

FOR i = 1 to MAX_ITERATIONS:
    // 1. 在空间中随机采样
    q_rand = SampleFreeSpace()
    
    // 2. 找到树中距离 q_rand 最近的节点
    q_near = NearestNeighbor(T, q_rand)
    
    // 3. 沿着 q_near 到 q_rand 的方向截取固定步长，得到新节点
    q_new = Steer(q_near, q_rand, StepSize)
    
    // 4. 碰撞检测：检查 q_near 到 q_new 的线段是否穿越障碍物
    IF CollisionFree(q_near, q_new):
        T.addNode(q_new)
        T.addEdge(q_near, q_new)
        
        // 5. 检查是否到达终点附近
        IF Distance(q_new, Goal) < Threshold:
            RETURN ExtractPath(T, q_new)

RETURN Failure
```

## 1.3 算法性质与证明

1. **概率完备性 (Probabilistic Completeness)**
   RRT 不具备绝对完备性（即不能保证在有限时间内绝对能找到解），但具有概率完备性。
   **证明**：假设解空间中存在一条连通的、具有一定宽度的无碰撞“管道”。随着采样次数 $N \to \infty$，均匀分布的随机采样点落入该管道并连接成树的概率趋近于 $1$。即只要有解，只要给的时间无限多，RRT 必定能找到解。

2. **探索速度快 (Voronoi Bias)**
   RRT 的随机采样机制导致树倾向于向大块的空白区域（未探索区域）生长。这是因为较大的空白区域在空间的 Voronoi 划分中占据了更大的体积，随机撒点落入这些区域并拉动树枝生长的概率更高。

3. **非最优性**
   由于节点是随机连接的，RRT 找到的路径通常非常曲折、棱角分明，完全不具备最优性，甚至比真实最短路径长很多。通常需要后处理（如贪心平滑 B-Spline）来优化路径。

## 1.4 C++代码实现

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

struct Node {
    double x, y;
    Node* parent;
    Node(double x, double y, Node* p = nullptr) : x(x), y(y), parent(p) {}
};

class RRTPlanner {
private:
    double step_size;
    double max_x, max_y;
    vector<Node*> tree;

    double distance(Node* a, Node* b) {
        return hypot(a->x - b->x, a->y - b->y);
    }

    // 简化版：假设空间全空，无障碍物
    bool checkCollision(Node* from, Node* to) { return true; }

public:
    RRTPlanner(double step, double mx, double my) : step_size(step), max_x(mx), max_y(my) {}

    ~RRTPlanner() { for(auto node : tree) delete node; }

    vector<pair<double, double>> search(double startX, double startY, double goalX, double goalY) {
        mt19937 rng(1234); // 随机数生成器
        uniform_real_distribution<double> distX(0, max_x);
        uniform_real_distribution<double> distY(0, max_y);

        tree.push_back(new Node(startX, startY));
        Node* goalNode = new Node(goalX, goalY);

        for (int i = 0; i < 5000; ++i) {
            // 1. 采样
            Node randNode(distX(rng), distY(rng));

            // 2. 找最近邻
            Node* nearest = tree[0];
            double min_dist = distance(nearest, &randNode);
            for (auto node : tree) {
                double d = distance(node, &randNode);
                if (d < min_dist) { min_dist = d; nearest = node; }
            }

            // 3. 截取步长 (Steer)
            double theta = atan2(randNode.y - nearest->y, randNode.x - nearest->x);
            Node* newNode = new Node(nearest->x + step_size * cos(theta), 
                                     nearest->y + step_size * sin(theta), nearest);

            // 4. 添加到树中
            if (checkCollision(nearest, newNode)) {
                tree.push_back(newNode);
                
                // 5. 检查是否抵达终点
                if (distance(newNode, goalNode) < step_size) {
                    goalNode->parent = newNode;
                    tree.push_back(goalNode);
                    
                    vector<pair<double, double>> path;
                    Node* curr = goalNode;
                    while (curr != nullptr) {
                        path.push_back({curr->x, curr->y});
                        curr = curr->parent;
                    }
                    return path;
                }
            } else { delete newNode; }
        }
        delete goalNode;
        return {}; // 失败
    }
};
```

---

# 2. RRT* (Optimal RRT)

## 2.1 基本介绍

RRT* 是 RRT 的进阶版本，由 Sertac Karaman 和 Emilio Frazzoli 在 2011 年提出。为了解决 RRT 路径曲折、非最优的致命缺点，RRT* 引入了代价（Cost）概念以及两大核心机制：**重新选择父节点（Choose Parent）**和**重连（Rewiring）**。每次生成新节点时，RRT* 不会直接连接到最近的节点，而是在以新节点为圆心的一定半径范围内寻找邻居，并选择一条能够让新节点到达起点代价最小的节点作为父节点。同时，它还会检查新节点的加入是否能作为这些邻居的更优父节点，如果是，则把邻居的树枝“剪断”并重新接到新节点上。

## 2.2 伪代码流程图

```text
Initialize Tree T with StartNode

FOR i = 1 to MAX_ITERATIONS:
    q_rand = SampleFreeSpace()
    q_near = NearestNeighbor(T, q_rand)
    q_new = Steer(q_near, q_rand, StepSize)
    
    IF CollisionFree(q_near, q_new):
        // 动态计算搜索半径 (随节点总数增加而缩小)
        radius = min(MaxRadius, gamma * (log(N) / N)^(1/d))
        Q_near = FindNeighborsInRadius(T, q_new, radius)
        
        // 第一阶段：为 q_new 挑选最佳父节点 (Choose Best Parent)
        q_min = q_near
        c_min = q_near.cost + Distance(q_near, q_new)
        FOR each q in Q_near:
            IF CollisionFree(q, q_new) AND (q.cost + Distance(q, q_new) < c_min):
                q_min = q
                c_min = q.cost + Distance(q, q_new)
                
        q_new.parent = q_min
        q_new.cost = c_min
        T.addNode(q_new)
        
        // 第二阶段：树枝重连 (Rewiring)
        FOR each q in Q_near:
            IF CollisionFree(q_new, q) AND (q_new.cost + Distance(q_new, q) < q.cost):
                q.parent = q_new
                q.cost = q_new.cost + Distance(q_new, q)
                // (严格实现中还需要向下递归更新 q 的所有子节点的代价)
```

## 2.3 算法性质与证明

1. **渐进最优性 (Asymptotic Optimality)**
   这是 RRT* 最伟大的理论贡献。随着采样次数 $N \to \infty$，RRT* 找到的路径的代价值几乎必然（概率为1）收敛到全局绝对最优解。
   **证明原理**：基于随机几何几何学，在 $d$ 维空间中，重连搜索半径被设置为 $r \propto (\frac{\log N}{N})^{1/d}$。这意味着虽然半径在逐渐缩小，但随着节点密度趋于无穷大，邻域内期望包含的节点数保持为正向增长的对数级别。这保证了在任何局部区域，节点之间的所有有效拓扑连接都被尝试过。局部连接的不断优化叠加，最终使得整个树从起点到任何区域的代价积分极小化。

2. **更高的单次迭代计算负担**
   由于引入了邻居搜索、碰撞重检测以及级联的代价更新机制，RRT* 的单次循环开销比 RRT 大得多。但由于渐近最优性，其整体产生的价值远远高于 RRT。

## 2.4 C++代码实现

```cpp
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

struct RRTStarNode {
    double x, y;
    double cost;
    RRTStarNode* parent;
    RRTStarNode(double x, double y, double cost = 0.0, RRTStarNode* p = nullptr) 
        : x(x), y(y), cost(cost), parent(p) {}
};

class RRTStarPlanner {
private:
    double step_size;
    double search_radius;
    vector<RRTStarNode*> tree;

    double dist(RRTStarNode* a, RRTStarNode* b) {
        return hypot(a->x - b->x, a->y - b->y);
    }
    bool checkCollision(RRTStarNode* a, RRTStarNode* b) { return true; }

public:
    RRTStarPlanner(double step, double radius) : step_size(step), search_radius(radius) {}

    // RRT* 的核心单步插入逻辑
    void insertNode(RRTStarNode* q_new, RRTStarNode* q_near) {
        vector<RRTStarNode*> Q_near;
        // 1. 寻找半径内的邻居
        for (auto node : tree) {
            if (dist(node, q_new) <= search_radius) Q_near.push_back(node);
        }

        // 2. Choose Best Parent
        RRTStarNode* q_min = q_near;
        double c_min = q_near->cost + dist(q_near, q_new);
        for (auto q : Q_near) {
            double c_tentative = q->cost + dist(q, q_new);
            if (c_tentative < c_min && checkCollision(q, q_new)) {
                c_min = c_tentative;
                q_min = q;
            }
        }
        q_new->parent = q_min;
        q_new->cost = c_min;
        tree.push_back(q_new);

        // 3. Rewiring (重连)
        for (auto q : Q_near) {
            double c_tentative = q_new->cost + dist(q_new, q);
            if (c_tentative < q->cost && checkCollision(q_new, q)) {
                q->parent = q_new;
                q->cost = c_tentative;
                // 注：此处省略了更新 q 的所有子树代价的递归函数 (Cascade Cost Update)
            }
        }
    }
};
```

---

# 3. PRM (概率路线图, Probabilistic Roadmap)

## 3.1 基本介绍

PRM 由 Lydia Kavraki 等人在 1996 年提出。与 RRT 的单次查询树结构不同，PRM 旨在构建一个通用的**无向图（路线图）**，非常适合环境地图已知且静态、且需要频繁查询不同起点到不同终点的多查询任务（Multi-query）。PRM 分为两个阶段：

1. **学习阶段 (Learning Phase)**：在环境的自由空间中随机撒布大量点（节点），并将每个点与其附近的无碰撞节点进行连线（边），从而构建出一张覆盖整个空间的拓扑网。
2. **查询阶段 (Query Phase)**：给定起点和终点，首先将它们就近接入到已经建好的路线图中，然后在整个图上运行标准的图搜索算法（如 Dijkstra 或 A*）即可快速获得最短路径。

## 3.2 伪代码流程图

```text
// 阶段 1：学习阶段 (构建路线图)
Function ConstructRoadmap(N_samples, k_neighbors):
    Graph = empty
    WHILE Graph.nodes.size() < N_samples:
        q_rand = SampleFreeSpace()
        Graph.addNode(q_rand)
        
    FOR each node u in Graph:
        neighbors = Find_K_Nearest_Neighbors(Graph, u, k_neighbors)
        FOR each v in neighbors:
            IF CollisionFree(u, v):
                Graph.addEdge(u, v, weight=Distance(u,v))

// 阶段 2：查询阶段
Function QueryPath(Start, Goal, Graph):
    // 尝试将起点和终点连入网络
    Start_connected = ConnectToGraph(Start, Graph)
    Goal_connected = ConnectToGraph(Goal, Graph)
    
    IF not Start_connected OR not Goal_connected:
        RETURN Failure
        
    // 在路线图上运行图搜索
    Path = A_Star_Search(Graph, Start, Goal)
    RETURN Path
```

## 3.3 算法性质与证明

1. **分离解耦设计带来的多查询高效性**
   PRM 的图构建代价极其昂贵（需要 $O(N^2)$ 或借助 KD-Tree $O(N \log N)$ 的碰撞检测和距离计算）。但在静态工厂、预设地图的游戏场景中，这部分计算可以完全离线完成（Offline）。一旦无向图构建完毕，在线（Online）寻找任意点对的路径退化成了在稀疏图上运行 A* 算法，速度达到毫秒级。

2. **概率完备性与窄通道问题 (Narrow Passage Problem)**
   PRM 同样具有概率完备性，只要样本足够多，网络就能覆盖解空间。但 PRM 在纯随机采样时面临严重的“窄通道问题”：在两个巨大空旷房间之间只有一扇极其狭窄的门，由于门的面积占总空间的比例极小，随机采样点极难落在门框内，导致路线图在门处断裂断连。针对此问题，学术界衍生出了 Gaussian PRM（在障碍物边缘增加采样密度）等优化版本。

3. **渐近最优性 (PRM*)**
   与 RRT* 类似，如果我们规定 PRM 连接邻居的半径（或 K 个最近邻的 K 值）不是固定的，而是随着节点数量的增加按特定公式变化（即 PRM* 算法），它也能保证找到的路径无限逼近真实全局最短路径。

## 3.4 C++代码实现

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <queue>
#include <random>

using namespace std;

struct PRMNode {
    int id;
    double x, y;
    PRMNode(int i, double x, double y) : id(i), x(x), y(y) {}
};

class PRMPlanner {
private:
    vector<PRMNode*> nodes;
    // 邻接表：adj[u] 存储 pair(v_id, edge_cost)
    vector<vector<pair<int, double>>> adj;

    double dist(PRMNode* a, PRMNode* b) { return hypot(a->x - b->x, a->y - b->y); }
    bool checkCollision(PRMNode* a, PRMNode* b) { return true; } // 假设无碰撞

public:
    PRMPlanner() {}
    ~PRMPlanner() { for (auto n : nodes) delete n; }

    // 阶段1：构建路线图 (Offline)
    void buildRoadmap(int num_samples, double connection_radius, double max_x, double max_y) {
        mt19937 rng(1234);
        uniform_real_distribution<double> distX(0, max_x);
        uniform_real_distribution<double> distY(0, max_y);

        // 随机采样
        for (int i = 0; i < num_samples; ++i) {
            nodes.push_back(new PRMNode(i, distX(rng), distY(rng)));
        }
        
        adj.resize(num_samples);

        // 邻域连接
        for (int i = 0; i < num_samples; ++i) {
            for (int j = i + 1; j < num_samples; ++j) {
                double d = dist(nodes[i], nodes[j]);
                if (d < connection_radius && checkCollision(nodes[i], nodes[j])) {
                    adj[i].push_back({j, d});
                    adj[j].push_back({i, d}); // 无向图
                }
            }
        }
    }

    // 阶段2：查询 (Online) - 使用简化的 Dijkstra 作为演示
    double query(int start_id, int goal_id) {
        vector<double> min_cost(nodes.size(), 1e9);
        priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;

        min_cost[start_id] = 0;
        pq.push({0.0, start_id});

        while (!pq.empty()) {
            double cost = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (u == goal_id) return cost; // 找到目标

            if (cost > min_cost[u]) continue;

            for (auto& edge : adj[u]) {
                int v = edge.first;
                double weight = edge.second;
                if (cost + weight < min_cost[v]) {
                    min_cost[v] = cost + weight;
                    pq.push({min_cost[v], v});
                }
            }
        }
        return -1.0; // 图不连通，寻找失败
    }
};
```
