# 🌳 Knowledge Tree

[![Deploy](https://github.com/ShangjinghangXUan/knowledge-tree/actions/workflows/deploy.yml/badge.svg)](https://github.com/ShangjinghangXUan/knowledge-tree/actions/workflows/deploy.yml)

> 个人技术知识树：数学、编程、机器学习、强化学习与机器人学的结构化笔记与开源教学。
> Markdown 源文件驱动，由 [Astro](https://astro.build) + [Starlight](https://starlight.astro.build) 构建，部署在 GitHub Pages。

**在线地址**：<https://shangjinghangxuan.github.io/knowledge-tree/>

## 内容结构

```
src/content/docs/
├── index.mdx               # 知识树首页（Splash）
├── mathematics/            # 数学基础（规划中）
├── programming/            # 编程与工程：Git、多服务器 GPU 训练
├── machine-learning/       # 机器学习
│   └── reinforcement-learning/   # 强化学习：PPO、工程避坑、参数、TensorBoard
├── robotics/               # 机器人学
│   ├── planning/           # 轨迹表示、参数化与优化
│   └── control/            # PX4 姿态与角速度控制
├── projects/               # 科研与开源项目实践
└── learning-paths/         # 学习路径
```

文件夹即知识分类，Markdown 文件即教学页面，无需维护独立的目录数据库。

## 本地开发

```bash
npm install
npm run dev      # 开发服务器 http://localhost:4321/knowledge-tree/
npm run build    # 生产构建
npm run preview  # 预览生产构建
```

推送到 `main` 分支后，GitHub Actions 会自动构建并发布网站。

## 如何贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。最简单的方式：在任一页面右上角点击「编辑此页」，GitHub 会自动引导你创建 Pull Request。

## 许可

- **网站代码**：[MIT License](LICENSE)
- **文档内容**：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，转载请注明来源并链接本仓库
