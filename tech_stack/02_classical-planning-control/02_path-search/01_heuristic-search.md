# 图搜索与启发式搜索

## 1. 概述

图搜索通过逐步扩展候选节点寻找低代价路径。Dijkstra 依据已知路径代价排序，A* 加入剩余代价下界，IDA* 和 JPS 分别改变存储与扩展方式。

## 2. 基础与定义

### 2.1 图与路径代价

图 $G=(V,E)$ 的边代价 $c(u,v)\ge0$。$g(v)$ 为当前起点到 $v$ 的路径代价，$h(v)$ 估计到目标的剩余代价，A* 按 $f(v)=g(v)+h(v)$ 排序。

### 2.2 启发式条件

可采纳性为 $h(v)\le h^*(v)$，一致性为 $h(u)\le c(u,v)+h(v)$，目标处 $h=0$。启发式的距离单位与边代价保持一致。

## 3. 核心原理

### 3.1 Dijkstra 与 A*

从队列中弹出最小键节点，松弛相邻边：

$$
g(v)\leftarrow\min(g(v),g(u)+c(u,v)).
$$

Dijkstra 对应 $h=0$。A* 使用一致启发式时，可在节点首次有效出队后关闭它；仅可采纳而不一致时，应允许更优路径触发重开。

<details class="kt-derivation" id="search-optimality">
<summary>A* 的最优性条件</summary>

若某条更优解尚未完成，其路径上至少存在一个边界节点 $n$。可采纳性给出 $f(n)\le C^*$。因此在按最小 $f$ 出队且正确维护更优路径时，代价大于 $C^*$ 的目标无法先于所有这些节点被接受。一致性进一步保证沿边的 $f$ 非递减，支持不重开已关闭节点。

</details>

### 3.2 IDA*

IDA* 对 $f$ 设置阈值，在阈值内深度优先搜索；失败后把阈值提高到本轮最小超限值。存储主要与搜索深度有关，重复展开会增加计算量。

### 3.3 跳点搜索

JPS 在适用的均匀代价格网上剪除对称路径，仅保留改变后续搜索决策的跳点。自然邻居、强制邻居和跳跃规则依赖移动与对角穿越约定；最优性在对应格网模型下成立。

## 4. 实现与应用

### 4.1 搜索验证

固定格网连通方式、对角代价与拐角穿越规则。比较 A* 与 Dijkstra 的最终代价，覆盖不可达、起终点相同、障碍边界和启发式不一致的情况。

<details class="kt-derivation" id="search-dijkstra-cpp">
<summary>Dijkstra 的 C++ 实现</summary>

示例输入为有效节点编号与非负边权，返回最短距离。

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using namespace std;

struct Node {
    int id;
    double g_cost;
    bool operator>(const Node& other) const {
        return g_cost > other.g_cost; // 最小堆
    }
};

class DijkstraPlanner {
private:
    int num_nodes;
    // 邻接表表示图: graph[u] 存储 (v, weight)
    vector<vector<pair<int, double>>> graph;

public:
    DijkstraPlanner(int n) : num_nodes(n), graph(n) {}

    void addEdge(int u, int v, double weight) {
        graph[u].push_back({v, weight});
        graph[v].push_back({u, weight}); // 无向图
    }

    double search(int start, int goal) {
        priority_queue<Node, vector<Node>, greater<Node>> pq;
        vector<double> dist(num_nodes, numeric_limits<double>::infinity());
        vector<int> parent(num_nodes, -1);

        dist[start] = 0;
        pq.push({start, 0.0});

        while (!pq.empty()) {
            Node current = pq.top();
            pq.pop();

            if (current.id == goal) return current.g_cost; // 找到目标

            // 遇到陈旧的冗余节点
            if (current.g_cost > dist[current.id]) continue;

            for (auto& edge : graph[current.id]) {
                int neighbor = edge.first;
                double weight = edge.second;

                double new_cost = dist[current.id] + weight;
                if (new_cost < dist[neighbor]) {
                    dist[neighbor] = new_cost;
                    parent[neighbor] = current.id;
                    pq.push({neighbor, new_cost});
                }
            }
        }
        return -1.0; // 无法到达
    }
};
```

</details>

<details class="kt-derivation" id="search-ida-cpp">
<summary>IDA* 的 C++ 实现</summary>

示例采用非空矩形四邻域格网、单位边权和曼哈顿启发式，起终点均须有效且自由。

```cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

const double INF = numeric_limits<double>::infinity();

class IDAStarPlanner {
private:
    vector<vector<int>> grid;
    int goalX, goalY;

    double heuristic(int x, int y) {
        return abs(x - goalX) + abs(y - goalY); // 曼哈顿距离
    }

    // DFS 搜索函数
    double search(int x, int y, double g, double bound, vector<pair<int, int>>& path) {
        double f = g + heuristic(x, y);
        if (f > bound) return f; // 超出阈值，剪枝
        if (x == goalX && y == goalY) return -1.0; // -1表示找到

        double min_bound = INF;
        int dx[4] = {0, 1, 0, -1};
        int dy[4] = {1, 0, -1, 0};

        for (int i = 0; i < 4; ++i) {
            int nx = x + dx[i];
            int ny = y + dy[i];

            if (nx >= 0 && nx < grid.size() && ny >= 0 && ny < grid[0].size() && grid[nx][ny] == 0) {
                // 防止环路
                bool in_path = false;
                for (auto p : path) {
                    if (p.first == nx && p.second == ny) in_path = true;
                }
                if (in_path) continue;

                path.push_back({nx, ny});
                double t = search(nx, ny, g + 1.0, bound, path);
                if (t == -1.0) return -1.0; // 成功回溯
                if (t < min_bound) min_bound = t;
                path.pop_back();
            }
        }
        return min_bound;
    }

public:
    IDAStarPlanner(const vector<vector<int>>& map) : grid(map) {}

    bool search(int startX, int startY, int gX, int gY) {
        goalX = gX; goalY = gY;
        double bound = heuristic(startX, startY);
        vector<pair<int, int>> path = {{startX, startY}};

        while (true) {
            double t = search(startX, startY, 0.0, bound, path);
            if (t == -1.0) return true; // 找到了
            if (t == INF) return false; // 无法到达
            bound = t; // 放大阈值
        }
    }
};
```

</details>

## 5. 分析与总结

| 方法 | 主要资源特征 | 适用结构 |
| --- | --- | --- |
| Dijkstra | 无启发式，扩展较广 | 非负边权图 |
| A* | 以启发式减少扩展 | 有可靠下界的图 |
| IDA* | 存储少，可能重复扩展 | 深度与存储受限搜索 |
| JPS | 跳过对称中间节点 | 特定均匀代价格网 |

搜索返回图模型中的路径，连续空间碰撞和动力学可行性由建图与后续验证负责。

## 6. 关联与参考

[LaValle：Planning Algorithms](https://lavalle.pl/planning/)、[Harabor 与 Grastien：Jump Point Search](https://harabor.net/data/papers/harabor-grastien-aaai11.pdf)。

关联：[增量搜索](./03_incremental-search.md)、[动力学约束搜索](./04_kinodynamic-search.md)。
