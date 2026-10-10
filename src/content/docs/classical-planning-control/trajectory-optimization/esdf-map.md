---
title: "ESDF 地图"
sidebar:
  order: 5
---

# 1. 增量式ESDF构建 (基于 Voxblox)

## 1.1 数学定义与构造概念

在 Voxblox 框架中，机器人的空间感知首先被融合进截断符号距离场（TSDF）中，用于平滑传感器噪声并提取物体表面。随后，算法利用 TSDF 提供的表面距离信息，向未知的自由空间传播，构建欧氏符号距离场（ESDF）。

给定 TSDF  voxel 集合，设 $v_T$ 为 TSDF 体素，$v_E$ 为对应的 ESDF 体素。
算法提取 TSDF 中距离表面极近的体素构成**固定带 (Fixed Band)**，即满足 $|v_T.d| < \gamma$ 的体素作为波前（Wavefront）的起点。
波前传播基于广度优先搜索（BFS），并拆分为两个独立的队列：

1. **Raise 队列（升高/失效）**：当障碍物消失或原距离变大时，使得受影响的体素距离设为无穷大。
2. **Lower 队列（降低/更新）**：当发现新障碍物或更优路径时，基于邻居体素更新当前体素的距离。

## 1.2 算法性质与证明

1. **投影误差累积**
   TSDF 测量的是沿传感器射线的投影距离，必然大于或等于真实的欧氏距离。
   证明：设真实距离为 $d$，射线与表面的入射角为 $\theta$。投影误差为 $r_p(\theta) = d \sin(\theta) - d$。Voxblox 证明，通过多次独立观测同一体素，该投影误差会迅速收敛。对3次以上的观测，95%置信度下的误差上限可降至 $0.5\delta$（$\delta$ 为截断距离）。
2. **准欧氏近似 (Quasi-Euclidean Approximation)**
   为加快传播速度，ESDF 沿网格连线（26邻域）计算距离，而非严格欧氏距离。
   证明：射线与网格连线最大夹角出现在 $\phi = 22.5^\circ$ 处。积分计算可知，准欧氏距离引入的最大理论误差为实际距离的 $8.24%$，预期误差为 $5.48%$。工程上通常将机器人碰撞包围盒膨胀 $8.5%$ 来完全抵消此误差。

## 1.3 算法伪代码

```text
Algorithm Voxblox_ESDF_Propagate(map_TSDF, map_ESDF):
    // 1. 初始化波前
    For each updated_voxel v_T in map_TSDF:
        If is_fixed_band(v_T): 
            If v_E.d > v_T.d: // 发现更近的表面
                v_E.d = v_T.d
                lower_queue.push(v_E)
            Else:             // 原表面消失
                v_E.d = v_T.d
                raise_queue.push(v_E)
                lower_queue.push(v_E)
                
    // 2. 处理失效的体素
    While raise_queue not empty:
        curr = raise_queue.pop()
        curr.d = MAX_DIST
        For each neighbor of curr:
            If neighbor's closest point was curr:
                raise_queue.push(neighbor)
            Else:
                lower_queue.push(neighbor)
                
    // 3. 处理距离更新的体素
    While lower_queue not empty:
        curr = lower_queue.pop()
        For each neighbor of curr:
            dist_to_neighbor = calculate_distance(curr, neighbor)
            If curr.d + dist_to_neighbor < neighbor.d:
                neighbor.d = curr.d + dist_to_neighbor
                neighbor.parent = curr
                lower_queue.push(neighbor)
```

## 1.4 C++代码实现

```cpp
#include <queue>
#include <vector>
#include <cmath>

struct Voxel {
    double distance = 1e6;
    bool is_fixed = false;
    Voxel* parent = nullptr;
    std::vector<Voxel*> neighbors;
};

class VoxbloxESDF {
private:
    std::queue<Voxel*> raise_queue;
    std::queue<Voxel*> lower_queue;

public:
    void processRaiseQueue() {
        while (!raise_queue.empty()) {
            Voxel* curr = raise_queue.front();
            raise_queue.pop();
            curr->distance = 1e6; 

            for (Voxel* neighbor : curr->neighbors) {
                if (neighbor->parent == curr) {
                    raise_queue.push(neighbor);
                } else {
                    lower_queue.push(neighbor);
                }
            }
        }
    }

    void processLowerQueue() {
        while (!lower_queue.empty()) {
            Voxel* curr = lower_queue.front();
            lower_queue.pop();

            for (Voxel* neighbor : curr->neighbors) {
                // 使用准欧氏近似计算距离 (假设相邻网格间距为 1 或 sqrt(2) 等)
                double dist_step = 1.0; 
                if (curr->distance + dist_step < neighbor->distance) {
                    neighbor->distance = curr->distance + dist_step;
                    neighbor->parent = curr;
                    lower_queue.push(neighbor);
                }
            }
        }
    }
};
```

# 2. 极速增量ESDF构建 (FIESTA)

## 2.1 数学定义与构造概念

Voxblox 依赖 TSDF，且在处理动态障碍物消失时计算开销巨大。FIESTA 直接基于**占据栅格地图 (Occupancy Grid Map)** 构建 ESDF，并引入了全新的核心数据结构。

1. **分离更新队列**：拆分为插入队列 (Insert Queue) 和删除队列 (Delete Queue)。
2. **双向链表 (Doubly Linked Lists, DLL)**：每个体素维护一个 DLL，记录所有“将当前体素作为最近障碍物”的其他体素。当某障碍物被删除时，只需沿着 DLL 精准失效依赖它的体素，避免了全局重算。
3. **最近障碍物坐标 (coc, Closest Obstacle Coordinate)**：体素严格存储欧氏空间下的障碍物坐标，基于坐标差计算严格欧氏距离，消除了准欧氏近似误差。

## 2.2 算法性质与证明

1. **精准的局部失效 (O(1) 追溯)**
   当占据状态发生改变时，通过索引数据结构（哈希表或数组），算法能在常数时间 $\Theta(1)$ 内定位到修改的体素，并利用 DLL 以 $O(k)$ 的时间复杂度使得 $k$ 个受影响的子节点失效。
2. **基于 BFS 的次优性界限 (Sub-optimality)**
   任何基于 BFS 向外扩张距离的 ESDF 算法都无法保证全局严格最优的欧氏距离。
   证明：考虑平面上覆盖的圆形区域。算法仅基于离散的24/26连通邻域更新。用有限个半径较小的圆去覆盖一个大圆，必然存在“未覆盖的缝隙”。当真实最近障碍物刚好落在此缝隙中时，BFS 无法传递正确的源头。但 FIESTA 证明这种最坏情况极为罕见，实际 RMS 误差通常在零点几个体素以内，满足运动规划需求。

## 2.3 算法伪代码

```text
Algorithm FIESTA_ESDF_Update(insertQueue, deleteQueue):
    // 1. 处理删除队列 (障碍物消失)
    While deleteQueue not empty:
        curr = deleteQueue.pop()
        For each voxel in curr.DLL:  // 追溯依赖该障碍物的所有体素
            delete_from_DLL(voxel.coc, voxel)
            voxel.coc = Infinity
            voxel.distance = Infinity
            
            // 尝试从邻居处获取新的最近障碍物
            For each nbr in voxel.neighbors:
                If nbr.coc exists and DIST(nbr.coc, voxel) < voxel.distance:
                    voxel.distance = DIST(nbr.coc, voxel)
                    voxel.coc = nbr.coc
            
            insert_into_DLL(voxel.coc, voxel)
            updateQueue.push(voxel)

    // 2. 处理插入队列 (新障碍物产生)
    While insertQueue not empty:
        curr = insertQueue.pop()
        curr.coc = curr.position
        curr.distance = 0
        insert_into_DLL(curr.coc, curr)
        updateQueue.push(curr)

    // 3. BFS 扩张更新
    While updateQueue not empty:
        curr = updateQueue.pop()
        For each nbr in curr.neighbors:
            // 严格欧氏距离判定
            If DIST(curr.coc, nbr.position) < nbr.distance:
                nbr.distance = DIST(curr.coc, nbr.position)
                delete_from_DLL(nbr.coc, nbr)
                nbr.coc = curr.coc
                insert_into_DLL(nbr.coc, nbr)
                updateQueue.push(nbr)
```

## 2.4 C++代码实现

这里展示包含核心 DLL 逻辑的 Voxel 数据结构及 BFS 拓展逻辑。

```cpp
#include <queue>
#include <vector>
#include <cmath>
#include <list>

struct Point {
    int x, y, z;
    double distTo(const Point& other) const {
        return std::sqrt(pow(x - other.x, 2) + pow(y - other.y, 2) + pow(z - other.z, 2));
    }
};

struct VIS { // Voxel Information Structure
    Point pos;
    double distance = 1e6;
    Point coc; // Closest Obstacle Coordinate
    
    // 双向链表实现，存储所有 coc 指向当前 VIS 的体素
    std::list<VIS*> dll; 
    std::list<VIS*>::iterator dll_it; // 自身在父节点 DLL 中的迭代器，用于 O(1) 删除
    
    std::vector<VIS*> neighbors;
};

class FiestaESDF {
private:
    std::queue<VIS*> updateQueue;

    void insertIntoDLL(VIS* parent, VIS* child) {
        if (parent == nullptr) return;
        parent->dll.push_front(child);
        child->dll_it = parent->dll.begin();
    }

    void deleteFromDLL(VIS* parent, VIS* child) {
        if (parent == nullptr) return;
        parent->dll.erase(child->dll_it);
    }

public:
    void processUpdateQueue() {
        while (!updateQueue.empty()) {
            VIS* curr = updateQueue.front();
            updateQueue.pop();

            for (VIS* nbr : curr->neighbors) {
                // 使用严格欧氏距离计算
                double new_dist = curr->coc.distTo(nbr->pos);
                if (new_dist < nbr->distance) {
                    nbr->distance = new_dist;
                    
                    // 假设可以从全局索引获取原 coc 对应的 VIS 指针 origin_coc_ptr
                    VIS* origin_coc_ptr = /* Get by nbr->coc */;
                    deleteFromDLL(origin_coc_ptr, nbr);
                    
                    nbr->coc = curr->coc;
                    VIS* new_coc_ptr = /* Get by curr->coc */;
                    insertIntoDLL(new_coc_ptr, nbr);
                    
                    updateQueue.push(nbr);
                }
            }
        }
    }
};
```
