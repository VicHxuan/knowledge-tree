# 🌳 Knowledge Tree

[![Deploy](https://github.com/VicHxuan/knowledge-tree/actions/workflows/deploy.yml/badge.svg)](https://github.com/VicHxuan/knowledge-tree/actions/workflows/deploy.yml)

> 个人技术知识树：数学基础、传统规划控制、学习式规划、工程工具、仿真评估与真机飞行的结构化笔记与开源教学。
> Markdown 原件驱动，确定性构建，部署在 GitHub Pages。

**在线地址**：<https://vichxuan.github.io/knowledge-tree/>

## 两层内容模型

**知识笔记的唯一事实源是 `tech_stack/`**，网站页面由同步脚本按内容映射表确定性生成——正文只改一处，网站自动跟进：

```
tech_stack/                        # ← 在这里写笔记（tracked，事实源）
├── 01_math-physic-base/           #    数学基础
├── 02_classical-planning-control/ #    传统规划控制
├── 03_learning-based-planning/    #    学习式规划
├── 04_engineering-tools/          #    工程工具链
├── 05_simulation-evaluation/      #    仿真与评估
└── 06_real-flight/                #    真机飞行

        │  scripts/sync-docs.mjs（按 src/data/content-map.json 生成标题、
        │  sidebar、草稿标记，转换相对链接）
        ▼

src/content/docs/                  # → 生成产物（untracked）+ 少量手写页
├── index.mdx                      #    首页（Splash）
├── <六大领域>/                     #    生成的章节页 + 手写的板块首页
├── learning-paths/                #    学习路径（手写，含进度追踪）
├── knowledge-map/                 #    知识图谱（React Flow 交互页）
├── bookmarks/                     #    书签与最近阅读（本地数据）
└── projects/                      #    研究项目页（手写）
```

领域注册表在 `src/data/domains.mjs`（侧栏分组、首页卡片、图谱域色共用的单一事实源）；`astro.config.mjs` 的 sidebar 用 `autogenerate` 按目录结构生成分组，`src/routeData.ts` 从各分类首页读取中文名称和顺序，并使上下篇导航保持一致。

## 本地开发

```bash
npm ci
npm run dev -- --background --host 127.0.0.1  # 后台预览，自动同步原件
npm run build         # 生产构建（prebuild 钩子自动先同步文档）
npm run docs:check    # 只读校验：同步、六模块、链接、重定向和关系
npm run privacy:check # 本地隐私规则扫描
npm run site:check    # 构建后检查真实 HTML 链接、资源和锚点
npm test             # 维护脚本与学习进度回归
npm run test:browser # 构建后验收折叠、深链接和学习路径
```

推送到 `main` 分支后，GitHub Actions 会先跑校验再构建发布。

## 阅读与内容架构

技术文章统一六模块，默认正文给出学习主线；推导、长算例和代码使用可展开细节。网页支持深链接自动展开、批量开合和完整打印。[PPO](tech_stack/03_learning-based-planning/02_rl-algorithms/02_model-free-rl/03_ppo.md)是完整示范，可运行教学实现位于 [examples/ppo_core.py](examples/ppo_core.py)。

分类页统一为板块定位、阅读主线、主题目录、下一步，目录和发布状态自动派生。学习路径保留教学顺序，以稳定步骤 ID 保存本地进度；项目页连接知识与实践证据。文章注册表负责展示元数据，关系文件只记录知识之间的边。

## 贡献指南

见 [CONTRIBUTING.md](CONTRIBUTING.md)。最简单的方式：在任一页面底部点击「编辑此页」——它会直接指向 `tech_stack/` 里的原始笔记，GitHub 会自动引导你创建 Pull Request。

本地维护可使用 `knowledge-tree-editor` 编辑文档，使用 `knowledge-tree-maintainer` 同步 GitHub 与网站。标题采用简洁的学术名称，不附重复中英文括注。

## 许可

- **网站代码**：[MIT License](LICENSE)
- **文档内容**：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，转载请注明来源并链接本仓库
