# 启发式搜索（Dijkstra / A* / IDA* / JPS）

## 1. 概述（Overview）

> 待补充。

## 2. 基础与定义（Foundations）

> 待补充。

## 3. 核心原理（Core Principles）

### 1. Dijkstra 算法

#### 1.1 基本介绍

Dijkstra 算法由 Edsger W. Dijkstra 在 1956 年提出，是最经典的单源最短路径算法。它主要用于解决带权有向图或无向图中的最短路径问题。从本质上讲，它可以看作是启发函数 $h(n) \equiv 0$ 时的特殊 A* 算法。算法采用贪心策略，维护一个按起始点到当前点实际累计代价 $g(n)$ 排序的优先队列，像水波纹一样向四周均匀扩展，直到碰触到目标节点。

#### 1.2 伪代码流程图

```text
Initialize distance array dist[] to Infinity for all nodes
Initialize dist[Start] = 0
PriorityQueue PQ = { (Start, 0) }  // 存储 (Node, g_cost)

WHILE PQ is not empty:
    current = PQ.pop_min_cost_node()

    IF current == Goal:
        RETURN path  // 找到终点，提前终止

    IF current is marked as visited:
        CONTINUE
    Mark current as visited

    FOR each neighbor of current:
        IF neighbor is obstacle or visited:
            CONTINUE
        
        new_cost = dist[current] + weight(current, neighbor)
        IF new_cost < dist[neighbor]:
            dist[neighbor] = new_cost
            parent[neighbor] = current
            PQ.push((neighbor, new_cost))

RETURN Failure
```

#### 1.3 算法性质与证明

1. **最优性 (Optimality)**
   在图中所有边的权值均非负（$w \ge 0$）的前提下，Dijkstra 算法保证能找到最短路径。
   **证明**：假设算法首次将目标节点 $G$ 从优先队列中弹出时，其路径代价为 $g(G)$，且存在一条真实的更短路径经过未访问节点 $U$ 到达 $G$，其真实最短代价为 $C^* < g(G)$。
   因为 $U$ 是通向更优解的中间节点，此时 $g(U) \le C^*$。既然边权非负，那么 $g(U) \le C^* < g(G)$。
   按照优先队列（最小堆）的弹出规则，$U$ 必然会比 $G$ 先出队。这与假设“$G$ 被弹出但存在更优未弹出节点 $U$”矛盾。因此，首次到达 $G$ 的代价即为全局最短代价。

2. **完备性 (Completeness)**
   只要图的节点数有限且起点与终点连通，算法必能找到一条路径。

3. **复杂度限制**
   由于没有任何方向性的启发信息，Dijkstra 会盲目地向所有方向展开搜索（形成圆形波前），导致在广阔的空旷地图中评估大量无关节点，时间复杂度为 $O((V+E)\log V)$。

#### 1.4 C++代码实现

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

---

### 2. A* (A-Star) 算法

#### 2.1 基本介绍

A* 算法是当前最著名的启发式图搜索算法。它在 Dijkstra 算法的基础上，引入了一个启发式评估函数 $h(n)$，用于预估当前节点到目标节点的代价。A* 算法的评估公式为 $f(n) = g(n) + h(n)$，其中 $g(n)$ 是从起点到当前节点的实际代价值。通过按 $f(n)$ 值从小到大弹出节点，A* 算法能够使得搜索波前像水流一样“具有方向感”地朝终点蔓延，从而大幅减少无效的节点扩展。

#### 2.2 伪代码流程图

```text
OpenList = empty PriorityQueue  // 按 f(n) 排序
ClosedList = empty Set

Start.g = 0
Start.h = heuristic(Start, Goal)
Start.f = Start.g + Start.h
OpenList.push(Start)

WHILE OpenList is not empty:
    current = OpenList.pop_lowest_f_node()

    IF current == Goal:
        RETURN reconstruct_path(current)

    ClosedList.add(current)

    FOR each neighbor in get_neighbors(current):
        IF neighbor in ClosedList OR neighbor is obstacle:
            CONTINUE

        tentative_g = current.g + distance(current, neighbor)

        IF neighbor not in OpenList OR tentative_g < neighbor.g:
            neighbor.parent = current
            neighbor.g = tentative_g
            neighbor.h = heuristic(neighbor, Goal)
            neighbor.f = neighbor.g + neighbor.h
            
            IF neighbor not in OpenList:
                OpenList.push(neighbor)

RETURN Failure
```

#### 2.3 算法性质与证明

1. **启发函数的可采纳性 (Admissibility) 决定最优性**
   当且仅当启发函数 $h(n)$ 是可采纳的（即对于任意节点 $n$，$h(n) \le h^*(n)$，$h^*$ 为真实最小剩余代价，这意味着评估总是“乐观”的），A* 能保证找到最短路径。
   **证明**：假设 A* 返回了次优终点解 $G_{sub}$，即 $f(G_{sub}) = g(G_{sub}) > C^*$（$C^*$为真实最优代价）。在搜索树的某处必定有一个属于真实最优路径的节点 $n$ 未被扩展。由于 $h$ 可采纳：
   $$f(n) = g(n) + h(n) \le g(n) + h^*(n) = C^*$$
   因此有 $f(n) \le C^* < f(G_{sub})$。由于 $n$ 的 $f$ 值比次优解小，A* 的优先队列必定会先弹出 $n$ 而不是 $G_{sub}$。这意味着次优解不可能先于最优路径上的节点被确立，证毕。

2. **一致性 (Consistency / Monotonicity)**
   如果启发函数满足三角不等式：$h(x) \le cost(x, y) + h(y)$。那么 A* 首次扩展某个节点时找到的路径即为它的最短路径（不会出现被重新放回 OpenList 的情况）。

#### 2.4 C++代码实现

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>

using namespace std;

struct AStarNode {
    int x, y;
    double g, f;
    AStarNode* parent;
    AStarNode(int x, int y, double g, double h, AStarNode* p = nullptr) 
        : x(x), y(y), g(g), f(g + h), parent(p) {}
};

struct CompareF {
    bool operator()(const AStarNode* a, const AStarNode* b) const {
        return a->f > b->f;
    }
};

class AStarPlanner {
private:
    vector<vector<int>> grid; // 0: free, 1: obstacle

    double heuristic(int x1, int y1, int x2, int y2) {
        // 使用欧几里得距离
        return hypot(x1 - x2, y1 - y2);
    }

public:
    AStarPlanner(const vector<vector<int>>& map) : grid(map) {}

    bool search(int startX, int startY, int goalX, int goalY) {
        priority_queue<AStarNode*, vector<AStarNode*>, CompareF> openList;
        vector<vector<bool>> closedList(grid.size(), vector<bool>(grid[0].size(), false));

        openList.push(new AStarNode(startX, startY, 0.0, heuristic(startX, startY, goalX, goalY)));

        int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
        int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};

        while (!openList.empty()) {
            AStarNode* current = openList.top();
            openList.pop();

            if (current->x == goalX && current->y == goalY) return true; // 寻路成功

            if (closedList[current->x][current->y]) continue;
            closedList[current->x][current->y] = true;

            for (int i = 0; i < 8; ++i) {
                int nx = current->x + dx[i];
                int ny = current->y + dy[i];

                if (nx >= 0 && nx < grid.size() && ny >= 0 && ny < grid[0].size() && grid[nx][ny] == 0) {
                    if (closedList[nx][ny]) continue;
                    
                    double cost_step = (i < 4) ? 1.0 : 1.414; // 直行与斜向代价
                    double g_new = current->g + cost_step;
                    double h_new = heuristic(nx, ny, goalX, goalY);
                    
                    openList.push(new AStarNode(nx, ny, g_new, h_new, current));
                }
            }
        }
        return false;
    }
};
```

---

### 3. IDA* (Iterative Deepening A*)

#### 3.1 基本介绍

标准的 A* 算法必须在内存中维护 OpenList 和 ClosedList，这使得它的空间复杂度随着搜索深度的增加呈指数级增长（$O(b^d)$）。IDA*（迭代加深 A*）将深度优先搜索（DFS）的低内存开销特性与 A* 的启发式剪枝结合起来。它以 A* 的 $f(n)$ 值作为 DFS 搜索的阈值。如果当前路径的 $f(n)$ 超过了阈值，就进行剪枝回溯；如果一轮搜索没有找到目标，则将阈值放宽为本轮被剪枝节点中最小的那个 $f(n)$，然后开始下一轮全新的迭代。

#### 3.2 伪代码流程图

```text
Function IDA_Star(Start, Goal):
    bound = heuristic(Start, Goal)
    path = [Start]
    
    LOOP:
        t = Search(path, 0, bound, Goal)
        IF t == FOUND:
            RETURN path
        IF t == Infinity:
            RETURN Failure
        bound = t  // 更新阈值为下一层最小值

Function Search(path, g, bound, Goal):
    current = path.last()
    f = g + heuristic(current, Goal)
    
    IF f > bound:
        RETURN f  // 超过界限，剪枝并返回当前 f 值供下次迭代参考
    IF current == Goal:
        RETURN FOUND
        
    min_cost = Infinity
    FOR each neighbor of current:
        IF neighbor not in path AND neighbor is free:
            path.push(neighbor)
            t = Search(path, g + cost(current, neighbor), bound, Goal)
            IF t == FOUND:
                RETURN FOUND
            IF t < min_cost:
                min_cost = t
            path.pop()
            
    RETURN min_cost
```

#### 3.3 算法性质与证明

1. **极低的空间复杂度**
   IDA* 不需要优先队列和访问状态表，它只需要在递归调用栈中维护当前正在探索的这一条路径。
   **证明**：在任意时刻，搜索树只有一条长为 $d$ 的枝干存在于内存中。若每个节点的分支因子为 $b$，存储开销严格控制在 $O(bd)$ 内。这使得它非常适合内存受限的嵌入式系统或状态空间极其庞大的问题（如魔方、15数码问题）。

2. **最优性保留**
   与 A* 一样，只要 $h(n)$ 是可采纳的，IDA* 就能找到最优解。因为每一轮它只会探索 $f(n) \le bound$ 的节点，且 `bound` 是逐渐递增的，遇到终点时的 `bound` 必然是最小可能代价。

3. **时间复杂度牺牲**
   每次增加阈值，IDA* 都要从头开始搜索，导致大量浅层节点被重复计算多次。然而在极高维树中，底层节点数量呈指数爆炸，顶层节点的重复计算占用的时间比例实际上可以忽略不计。

#### 3.4 C++代码实现

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

---

### 4. JPS (Jump Point Search)

#### 4.1 基本介绍

A* 算法在均匀网格（Grid Map）地图中由于大量平行的等价路径（Symmetric Paths），会将大量多余的节点放入 OpenList 从而造成性能浪费。JPS（跳点搜索）是专门针对均匀无权网格优化的 A* 算法。它引入了“强迫邻居”（Forced Neighbor）的概念。在搜索时，节点并非仅仅向周围一步一步扩展，而是沿着水平/垂直方向直线发射，直到撞到障碍物（或发现强迫邻居）才停下，把这个关键转折点（跳点 Jump Point）加入 OpenList。JPS 极大降低了堆操作的次数，速度可比 A* 快一到两个数量级。

#### 4.2 伪代码流程图

```text
Function JPS_Search(Start, Goal):
    OpenList.push(Start)
    
    WHILE OpenList is not empty:
        current = OpenList.pop_lowest_f()
        IF current == Goal: RETURN path
        
        // 与 A* 的区别：只扩展 "跳点"
        FOR each direction in get_pruned_directions(current.parent, current):
            jump_node = jump(current, direction, Goal)
            IF jump_node != NULL:
                jump_node.g = current.g + distance(current, jump_node)
                jump_node.f = jump_node.g + heuristic(jump_node, Goal)
                OpenList.push(jump_node)

Function jump(node, dir, Goal):
    next_node = node + dir
    IF next_node is obstacle: RETURN NULL
    IF next_node == Goal: RETURN next_node
    
    // 如果存在强迫邻居，则当前节点成为跳点
    IF has_forced_neighbor(next_node, dir):
        RETURN next_node
        
    // 对角线搜索需要递归发起直行搜索
    IF is_diagonal(dir):
        IF jump(next_node, horizontal(dir), Goal) != NULL OR 
           jump(next_node, vertical(dir), Goal) != NULL:
            RETURN next_node
            
    RETURN jump(next_node, dir, Goal)
```

#### 4.3 算法性质与证明

1. **路径对称性破除 (Symmetry Breaking)**
   在空旷网格中，从点 A 到点 B 有无数条走阶梯状的对角线路径。JPS 根据到达当前节点的方向，利用剪枝规则直接摒弃那些不需要绕过障碍物也能到达的邻居。
   **证明**：如果在直行路径的旁侧没有障碍物，那么从起点到达这些旁侧节点，总是存在一条不经过当前节点的等价（甚至更短）路径。只有当路径旁侧出现障碍物时，某些节点由于被障碍物挡住，除了经过当前节点绕行别无他法，这些节点被称为“强迫邻居”。强迫邻居的出现破坏了空旷空间的对称性，因此程序必须停下来在此抛锚（设为跳点）。

2. **跳点图等效性 (Equivalence to Grid Graph)**
   JPS 并未改变 A* 的启发式公式，它只是压缩了图的结构，将网格图抽象成了一个仅包含 Start、Goal 和所有 Jump Point 的宏观拓扑图。
   由于它只跳过了不需要转弯或没有选择分歧的直线节点，保留了所有关键的拓扑分岔口，因此它严格保持了 A* 算法的最优性和完备性。

#### 4.4 C++代码实现

JPS 的核心在于**“方向剪枝（Pruning）”**和**“跳跃（Jumping）”**规则。由于剪枝和寻找强迫邻居的条件判断分支极多，常规的示例代码往往会省略这些细节。下面这份代码严格实现了原版 JPS 中的 8 方向剪枝和对角线强迫邻居检查。

##### JPS 算法完整 C++ 代码实现

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>

using namespace std;

// 节点结构体定义
struct JPSNode {
    int x, y;
    double g, f;
    JPSNode* parent;

    JPSNode(int x, int y, double g = 0.0, double f = 0.0, JPSNode* p = nullptr)
        : x(x), y(y), g(g), f(f), parent(p) {}
};

// 优先队列的比较函数（最小堆）
struct CompareF {
    bool operator()(const JPSNode* a, const JPSNode* b) const {
        // f值相同时，优先扩展g值更大的节点（更接近终点，往往能减少搜索节点）
        if (a->f == b->f) return a->g < b->g;
        return a->f > b->f;
    }
};

class JPSPlanner {
private:
    vector<vector<int>> grid;
    int rows, cols;
    int goalX, goalY;

    // 内存管理：保存所有申请的节点以便最后释放
    vector<JPSNode*> allocated_nodes;

    JPSNode* createNode(int x, int y, double g, double f, JPSNode* parent) {
        JPSNode* node = new JPSNode(x, y, g, f, parent);
        allocated_nodes.push_back(node);
        return node;
    }

    // 检查坐标是否合法且无障碍物
    bool isValid(int x, int y) {
        return x >= 0 && x < rows && y >= 0 && y < cols && grid[x][y] == 0;
    }

    // 计算启发式代价：8方向移动推荐使用 Octile Distance (对角线距离)
    double heuristic(int x, int y) {
        double dx = abs(x - goalX);
        double dy = abs(y - goalY);
        return (dx + dy) + (sqrt(2.0) - 2.0) * min(dx, dy);
    }

    // 根据 JPS 规则，剪枝获取当前节点应当探索的方向
    vector<pair<int, int>> getPrunedDirections(JPSNode* node) {
        vector<pair<int, int>> dirs;
        int x = node->x, y = node->y;

        if (node->parent == nullptr) {
            // 起点：没有前驱方向，向周围8个方向全部探索
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;
                    if (isValid(x + dx, y + dy)) dirs.push_back({dx, dy});
                }
            }
            return dirs;
        }

        // 计算从父节点走到当前节点的方向
        int dx = clamp(x - node->parent->x, -1, 1);
        int dy = clamp(y - node->parent->y, -1, 1);

        // 对角线移动剪枝
        if (dx != 0 && dy != 0) {
            if (isValid(x + dx, y)) dirs.push_back({dx, 0}); // 直行分量1
            if (isValid(x, y + dy)) dirs.push_back({0, dy}); // 直行分量2
            if (isValid(x + dx, y + dy)) dirs.push_back({dx, dy}); // 继续对角线

            // 强迫邻居检查 (对角线方向的侧边被挡住时，允许切向侧后方)
            if (!isValid(x - dx, y) && isValid(x, y + dy)) {
                if (isValid(x - dx, y + dy)) dirs.push_back({-dx, dy});
            }
            if (!isValid(x, y - dy) && isValid(x + dx, y)) {
                if (isValid(x + dx, y - dy)) dirs.push_back({dx, -dy});
            }
        } 
        // 直行移动剪枝
        else {
            if (dx != 0) { // 水平移动
                if (isValid(x + dx, y)) dirs.push_back({dx, 0});
                // 强迫邻居检查 (上下两侧被挡住时，必须检查侧方)
                if (!isValid(x, y + 1) && isValid(x + dx, y + 1)) dirs.push_back({dx, 1});
                if (!isValid(x, y - 1) && isValid(x + dx, y - 1)) dirs.push_back({dx, -1});
            } else { // 垂直移动
                if (isValid(x, y + dy)) dirs.push_back({0, dy});
                // 强迫邻居检查 (左右两侧被挡住时，必须检查侧方)
                if (!isValid(x + 1, y) && isValid(x + 1, y + dy)) dirs.push_back({1, dy});
                if (!isValid(x - 1, y) && isValid(x - 1, y + dy)) dirs.push_back({-1, dy});
            }
        }
        return dirs;
    }

    // 核心跳跃函数：沿着给定的 dx, dy 飞越，直到找到跳点、障碍物或目标
    pair<int, int> jump(int x, int y, int dx, int dy) {
        int nx = x + dx;
        int ny = y + dy;

        if (!isValid(nx, ny)) return {-1, -1}; // 碰到障碍或边界，跳跃失败
        if (nx == goalX && ny == goalY) return {nx, ny}; // 撞线终点！

        // 检查当前点是否因为有“强迫邻居”而被迫成为“跳点”
        if (dx != 0 && dy != 0) { // 对角线移动时
            // 对角线强迫邻居判定
            if ((isValid(nx - dx, ny + dy) && !isValid(nx - dx, ny)) ||
                (isValid(nx + dx, ny - dy) && !isValid(nx, ny - dy))) {
                return {nx, ny};
            }
            // 【JPS核心机制】：对角线移动必须递归衍生出正交(直行)搜索
            // 如果直行方向发现了跳点，那么当前的对角线点也必须停下成为跳点！
            if (jump(nx, ny, dx, 0).first != -1 || jump(nx, ny, 0, dy).first != -1) {
                return {nx, ny};
            }
        } else { // 直行移动时
            if (dx != 0) { // 水平移动
                if ((isValid(nx, ny + 1) && !isValid(x, ny + 1)) ||
                    (isValid(nx, ny - 1) && !isValid(x, ny - 1))) {
                    return {nx, ny};
                }
            } else { // 垂直移动
                if ((isValid(nx + 1, ny) && !isValid(nx + 1, y)) ||
                    (isValid(nx - 1, ny) && !isValid(nx - 1, y))) {
                    return {nx, ny};
                }
            }
        }

        // 没找到任何感兴趣的点，继续沿着原方向递归飞跃
        return jump(nx, ny, dx, dy);
    }

    // 重构路径，将拐点连接起来
    vector<pair<int, int>> reconstructPath(JPSNode* goalNode) {
        vector<pair<int, int>> path;
        JPSNode* curr = goalNode;
        while (curr != nullptr) {
            path.push_back({curr->x, curr->y});
            curr = curr->parent;
        }
        reverse(path.begin(), path.end());
        return path; // 注意：JPS返回的是“跳点（拐点）”组成的折线路径
    }

public:
    JPSPlanner(const vector<vector<int>>& map) : grid(map) {
        rows = grid.size();
        cols = grid[0].size();
    }

    ~JPSPlanner() {
        for (auto node : allocated_nodes) {
            delete node;
        }
    }

    // JPS 主循环，外部调用接口
    vector<pair<int, int>> search(int startX, int startY, int gX, int gY) {
        goalX = gX;
        goalY = gY;

        if (!isValid(startX, startY) || !isValid(goalX, goalY)) {
            return {}; // 起点或终点不合法
        }

        priority_queue<JPSNode*, vector<JPSNode*>, CompareF> openList;
        // 记录到达某点的最小 g 代价 (取代传统 ClosedList)
        vector<vector<double>> g_score(rows, vector<double>(cols, numeric_limits<double>::infinity()));

        JPSNode* startNode = createNode(startX, startY, 0.0, heuristic(startX, startY), nullptr);
        openList.push(startNode);
        g_score[startX][startY] = 0.0;

        while (!openList.empty()) {
            JPSNode* current = openList.top();
            openList.pop();

            // 寻路成功
            if (current->x == goalX && current->y == goalY) {
                return reconstructPath(current);
            }

            // 遇到过期的陈旧节点，跳过（Lazy Deletion）
            if (current->g > g_score[current->x][current->y]) continue;

            // 1. 获取根据JPS规则剪枝后的前进方向
            vector<pair<int, int>> dirs = getPrunedDirections(current);

            // 2. 朝着这些方向发起跳跃
            for (auto dir : dirs) {
                pair<int, int> jumpPoint = jump(current->x, current->y, dir.first, dir.second);
                
                if (jumpPoint.first != -1) {
                    int jx = jumpPoint.first;
                    int jy = jumpPoint.second;

                    // 计算跳跃这段距离的代价 (欧几里得距离)
                    double jump_dist = hypot(jx - current->x, jy - current->y);
                    double new_g = current->g + jump_dist;

                    // 如果找到了一条更短的路径到达该跳点
                    if (new_g < g_score[jx][jy]) {
                        g_score[jx][jy] = new_g;
                        double new_f = new_g + heuristic(jx, jy);
                        JPSNode* jumpNode = createNode(jx, jy, new_g, new_f, current);
                        openList.push(jumpNode);
                    }
                }
            }
        }
        return {}; // 队列为空，未找到路径
    }
};

// ======================= 测试用例 =======================
int main() {
    // 0代表空地，1代表障碍物
    vector<vector<int>> map = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 1, 1, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };

    JPSPlanner planner(map);
    
    // 从 (0,0) 到 (4,7)
    vector<pair<int, int>> path = planner.search(0, 0, 4, 7);

    if (path.empty()) {
        cout << "Path not found!" << endl;
    } else {
        cout << "JPS Waypoints (Jump Points):" << endl;
        for (auto p : path) {
            cout << "(" << p.first << ", " << p.second << ") -> ";
        }
        cout << "Goal!" << endl;
    }

    return 0;
}
```

## 4. 实现与应用（Implementation & Applications）

> 待补充。

## 5. 分析与总结（Analysis & Summary）

> 待补充。

## 6. 关联与参考（Related Topics & References）

> 待补充。
