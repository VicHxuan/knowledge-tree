# 1. D* (Dynamic A*)

## 1.1 基本介绍

D*（Dynamic A*）由 Anthony Stentz 在 1994 年提出，是专为未知或动态环境（如火星探测器、扫地机器人）设计的增量式路径规划算法。当机器人在沿着预定路线前进时，传感器的视野是有限的；如果突然发现前方有未知的障碍物阻挡，使用 A* 重新规划整条路径会导致极大的计算延迟。
D* 的核心思想是**反向搜索（Reverse Search）**和**波前传播**。它最初以终点为根节点向起点构建一棵最短路径树。当探测到路径上的代价增加（如发现障碍物）时，受影响的节点会产生“升高波（Raise Wave）”向外传播，剥离失效的路径；随后产生“降低波（Lower Wave）”引入新的最优路径，从而只在局部修补受影响的树枝，极大提高了重规划效率。

## 1.2 伪代码流程图

D* 为每个节点维护两个值：$h$（当前估计的最短代价值）和 $k$（该节点在 OpenList 中历史最小的 $h$ 值）。
如果 $k < h$，说明代价增加了，正在传播升高波；如果 $k = h$，说明是最优的，正在传播降低波。

```text
Function Modify-Cost(X, Y, new_cost):
    cost(X, Y) = new_cost
    IF X is CLOSED:
        Insert X into OpenList with h(X)
    RETURN Process-State()

Function Process-State():
    X = OpenList.pop_lowest_k()
    
    // 1. 升高波 (Raise Wave)：代价增加，通知受影响的邻居
    IF k_old < h(X):
        FOR each neighbor Y of X:
            IF Y points to X AND h(Y) != h(X) + cost(X, Y):
                h(Y) = h(X) + cost(X, Y)
                Insert Y into OpenList
                
    // 2. 降低波 (Lower Wave)：找到最优解，提供给邻居
    ELSE IF k_old == h(X):
        FOR each neighbor Y of X:
            IF h(Y) > h(X) + cost(X, Y):
                h(Y) = h(X) + cost(X, Y)
                Y.parent = X
                Insert Y into OpenList
                
    // 3. 状态纠正：处理波浪边缘的次优情况
    ELSE:
        FOR each neighbor Y of X:
            IF Y can be optimized by X:
                h(Y) = h(X) + cost(X,Y)
                Y.parent = X
                Insert Y into OpenList
```

## 1.3 算法性质与证明

1. **反向搜索的天然适应性**
   **证明**：在动态避障场景中，机器人的位置（起点 $S$）在不断移动，而目标位置（终点 $G$）通常是固定的。如果以 $S$ 为根节点建树，机器人每走一步，根节点就变了，整棵树必须推倒重来。D* 以 $G$ 为根节点建树，辐射向全图。机器人向 $G$ 移动只是在沿着既有的树枝往根部走。只要新发现的障碍物没有切断主干，机器人只需稍微查询旁边的树枝即可绕过障碍。

2. **局部更新的完备性与最优性**
   D* 被证明在有限连通图内是完备且最优的。
   通过 $k$ 值的引入，算法能够明确区分“受损的节点”（$k < h$）和“健康的节点”（$k = h$）。升高波像水流遇到石头一样将受损区域的代价值推高，直到遇到周边代价值较低的“健康节点”。健康节点顺势产生降低波，像水流绕过石头一样填补受损区域。因为只有波及到的节点会进入 OpenList，未受影响的庞大地图背侧完全不参与计算。

## 1.4 C++代码实现

*注：原版 D* 状态极其复杂，此处提炼其最核心的 `ProcessState` 波前传播逻辑框架。*

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using namespace std;

const double INF = numeric_limits<double>::infinity();

struct DNode {
    int id;
    double h; // 当前到目标的代价
    double k; // 历史最小h值
    DNode* parent;
    bool inOpen;

    DNode(int id) : id(id), h(INF), k(INF), parent(nullptr), inOpen(false) {}
};

struct CompareDNode {
    bool operator()(const DNode* a, const DNode* b) const {
        return a->k > b->k; // 按 k 值最小堆排序
    }
};

class DStarPlanner {
private:
    vector<DNode*> nodes;
    priority_queue<DNode*, vector<DNode*>, CompareDNode> openList;
    vector<vector<pair<int, double>>> adj; // 邻接表

    void insert(DNode* u, double new_h) {
        if (u->inOpen) {
            u->k = min(u->k, new_h);
        } else {
            u->k = new_h;
            u->inOpen = true;
        }
        u->h = new_h;
        openList.push(u);
    }

public:
    DStarPlanner(int num_nodes) : adj(num_nodes) {
        for (int i = 0; i < num_nodes; ++i) nodes.push_back(new DNode(i));
    }
    
    ~DStarPlanner() { for (auto n : nodes) delete n; }

    // 核心传播函数
    void processState() {
        if (openList.empty()) return;
        DNode* X = openList.top();
        openList.pop();
        X->inOpen = false;
        double k_old = X->k;

        // 1. 升高波：节点代价变大了，传染给依赖它的邻居
        if (k_old < X->h) {
            for (auto& edge : adj[X->id]) {
                DNode* Y = nodes[edge.first];
                double cost = edge.second;
                if (Y->parent == X && Y->h != X->h + cost) {
                    insert(Y, X->h + cost);
                }
            }
        } 
        // 2. 降低波：节点是最优的，尝试优化邻居
        else if (k_old == X->h) {
            for (auto& edge : adj[X->id]) {
                DNode* Y = nodes[edge.first];
                double cost = edge.second;
                if (Y->h > X->h + cost) {
                    Y->parent = X;
                    insert(Y, X->h + cost);
                }
            }
        }
        // 3. 纠正波（略）：处理 k_old > X->h 等边缘情况
    }

    // 假设机器人走到 X，发现到 Y 的代价突变
    void modifyCost(int x_id, int y_id, double new_cost) {
        DNode* X = nodes[x_id];
        DNode* Y = nodes[y_id];
        // 更新图结构中的代价...
        
        // 如果 X 依赖于 Y，说明路径断了
        if (X->parent == Y) {
            insert(X, INF); // 把 X 置为失效，引发升高波
        }
        // 循环直到波浪平息
        // while(openList.top()->k < ... ) processState();
    }
};
```

---

# 2. LPA* (Lifelong Planning A*)

## 2.1 基本介绍

LPA*（Lifelong Planning A*）由 Sven Koenig 和 Maxim Likhachev 于 2004 年提出。它融合了 A* 算法的启发式搜索特性和动态图的最短路径树更新机制。LPA* 主要解决的问题是：**起点和终点固定，但地图上的某些边/节点的通行代价会随着时间不断变化**（例如城市交通路网中的拥堵状态变化）。
LPA* 为每个节点维护两个核心状态：

1. $g(n)$：从起点到节点 $n$ 的实际已计算代价。
2. $rhs(n)$（Right-Hand Side）：基于其所有前驱节点计算出的一步超前预估代价。
   当环境未改变时，$g(n) = rhs(n)$（称为局部一致）。当某处路况改变时，会导致邻居节点的 $rhs$ 变化，产生局部不一致（$g \neq rhs$）。LPA* 通过一个优先队列，精确且仅重新计算这些不一致的节点。

## 2.2 伪代码流程图

```text
Function CalculateKey(s):
    RETURN [min(g(s), rhs(s)) + h(s, Goal), min(g(s), rhs(s))]

Function UpdateVertex(u):
    IF u != Start:
        rhs(u) = min_{s in predecessors(u)} (g(s) + cost(s, u))
        
    IF u is in OpenList: Remove u from OpenList
    
    // 如果不一致，压入优先队列等待修复
    IF g(u) != rhs(u):
        Insert u into OpenList with CalculateKey(u)

Function ComputeShortestPath():
    WHILE (OpenList.topKey() < CalculateKey(Goal)) OR (rhs(Goal) != g(Goal)):
        u = OpenList.pop()
        
        // 情况 1: 局部过一致 (找到了更短的路)
        IF g(u) > rhs(u):
            g(u) = rhs(u)
            FOR each successor s of u: UpdateVertex(s)
            
        // 情况 2: 局部欠一致 (原来的路变差了/断了)
        ELSE:
            g(u) = Infinity
            UpdateVertex(u)
            FOR each successor s of u: UpdateVertex(s)

Function Main():
    Initialize g and rhs to Infinity for all nodes
    rhs(Start) = 0
    Insert Start into OpenList with CalculateKey(Start)
    ComputeShortestPath()
    
    LOOP FOREVER:
        Wait for edge costs to change
        FOR each changed edge (u, v):
            Update cost(u, v)
            UpdateVertex(v) // v 受到了影响
        ComputeShortestPath()
```

## 2.3 算法性质与证明

1. **局部一致性 (Local Consistency) 驱动**
   LPA* 的数学严谨性建立在贝尔曼等式上：
   $$rhs(u) = \min_{s \in pred(u)} (g(s) + c(s,u))$$
   **证明**：如果对图中的所有节点都有 $g(u) = rhs(u)$，则 $g$ 值必然是到达各点的全局最短路径代价。
   当代价 $c(s,u)$ 改变时，只有受影响的节点 $u$ 失去了等式平衡。LPA* 的 `UpdateVertex` 将其推入队列。在 `ComputeShortestPath` 中，算法强制让 $g(u)$ 追平 $rhs(u)$，使得节点恢复平衡，同时这种值的改变又会像多米诺骨牌一样影响后续节点。由于引入了包含启发式函数 $h$ 的 Key，恢复平衡的操作被严格约束在朝向目标的椭圆区域内，避免了全图计算。

2. **状态分类的精确性**

   * **过一致 (Overconsistent, $g > rhs$)**：通常发生在发现了捷径。算法直接赋值 $g = rhs$，一次搞定。
   * **欠一致 (Underconsistent, $g < rhs$)**：通常发生在遇到了障碍物。算法必须先将 $g$ 抹除（设为 $\infty$），然后退回到周边重新评估，这需要两步走，确保逻辑上不会陷入死锁环路。

## 2.4 C++代码实现

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include <algorithm>

using namespace std;

const double INF = numeric_limits<double>::infinity();

struct LPAKey {
    double k1, k2;
    bool operator>(const LPAKey& other) const {
        if (abs(k1 - other.k1) > 1e-5) return k1 > other.k1;
        return k2 > other.k2;
    }
    bool operator<(const LPAKey& other) const {
        if (abs(k1 - other.k1) > 1e-5) return k1 < other.k1;
        return k2 < other.k2;
    }
};

struct LPANode {
    int id;
    int x, y;
    double g, rhs;
    bool inOpen;

    LPANode(int id, int x, int y) : id(id), x(x), y(y), g(INF), rhs(INF), inOpen(false) {}
};

class LPAPlanner {
private:
    vector<LPANode*> nodes;
    int goalId;

    double heuristic(LPANode* a, LPANode* b) {
        return hypot(a->x - b->x, a->y - b->y);
    }

    LPAKey calcKey(LPANode* s) {
        double min_val = min(s->g, s->rhs);
        return {min_val + heuristic(s, nodes[goalId]), min_val};
    }

    struct QueueElement {
        LPANode* node;
        LPAKey key;
        bool operator>(const QueueElement& other) const { return key > other.key; }
    };

    priority_queue<QueueElement, vector<QueueElement>, greater<QueueElement>> openList;
    vector<vector<pair<int, double>>> pred; // 前驱节点列表
    vector<vector<pair<int, double>>> succ; // 后继节点列表

public:
    LPAPlanner(int num_nodes, int goal) : goalId(goal), pred(num_nodes), succ(num_nodes) {
        // 节点初始化代码略...
    }

    void updateVertex(LPANode* u, int startId) {
        if (u->id != startId) {
            u->rhs = INF;
            for (auto& p : pred[u->id]) {
                LPANode* s = nodes[p.first];
                u->rhs = min(u->rhs, s->g + p.second);
            }
        }
        
        // 如果在优先队列里，C++标准库无法直接移除，通常采用Lazy Deletion(懒惰删除)或自定义带更新的堆
        // 此处伪逻辑：假定我们已经将其从队列清理

        if (u->g != u->rhs) {
            openList.push({u, calcKey(u)});
            u->inOpen = true;
        }
    }

    void computeShortestPath(int startId) {
        while (!openList.empty()) {
            LPANode* u = openList.top().node;
            LPAKey k_old = openList.top().key;
            LPAKey k_new = calcKey(u);
            
            // 延迟删除：如果 key 过期了，用新 key 重新插入
            if (k_old < k_new) {
                openList.pop();
                openList.push({u, k_new});
                continue;
            }

            LPANode* goalNode = nodes[goalId];
            if (!(k_old < calcKey(goalNode)) && goalNode->rhs == goalNode->g) break;

            openList.pop();
            u->inOpen = false;

            if (u->g > u->rhs) { // 过一致 (找到更短路)
                u->g = u->rhs;
                for (auto& s : succ[u->id]) updateVertex(nodes[s.first], startId);
            } else { // 欠一致 (代价变大/断路)
                u->g = INF;
                updateVertex(u, startId);
                for (auto& s : succ[u->id]) updateVertex(nodes[s.first], startId);
            }
        }
    }
};
```
