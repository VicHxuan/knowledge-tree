# 贡献指南

感谢你为知识树贡献内容！

## 两层内容模型（先读这个）

- **章节正文**：写在 `tech_stack/` 下的原始笔记里（事实源）。`src/content/docs/` 中对应的页面由 `scripts/sync-docs.mjs` **自动生成**（untracked），直接改生成产物会在下次同步时被覆盖。
- **手写页面**：板块首页（各目录的 `index.md`）、`learning-paths/`、`knowledge-map/`、`projects/` 直接维护在 `src/content/docs/` 下。

不确定某篇正文该改哪里？点页面上角的「编辑此页」，链接会指向事实源。

## 新增或修改章节

1. 原始笔记放在 `tech_stack/` 对应领域目录下，文件名使用小写英文与连字符（kebab-case）；
2. 在 `src/data/content-map.json` 注册一条映射（source / output / slug / title / label / order / status）；
3. 运行 `npm run docs:sync` 生成页面；状态为 `draft` 的条目会带 `draft: true`，只在开发服务器可见、不进生产构建；
4. 正文从二级标题 `##` 开始组织，**不要**写一级标题——页面大标题由映射表的 `title` 渲染（与标题相同的 H1 会被自动吸收）；
5. 数学公式使用 `$...$`（行内）与 `$$...$$`（块级），由 KaTeX 渲染；
6. 原始笔记中的站内相对链接会被同步脚本自动转换；站内跳转写完整路径并带 `/knowledge-tree` 前缀，例如：

   ```md
   [PPO](/knowledge-tree/learning-based-planning/rl-algorithms/ppo/)
   ```

## 新增手写页面

在 `src/content/docs/` 对应目录放置 `.md` / `.mdx` 文件，frontmatter 示例：

```md
---
title: PPO 强化学习算法
description: 一句话简介，用于搜索与页面摘要
sidebar:
  order: 5
---
```

新增分类目录时放置 `index.md` 作为分组首页，并检查 `astro.config.mjs` 中 `sidebar` 的分组配置。

## 撰写规范

- 正文内容保持原始笔记风格，修改他人内容时尽量做最小改动。
- 引用其他知识点时优先链接站内页面，而不是复制内容。
- 涉及内部项目的内容，注意不要包含真实服务器地址、密码或密钥。
- 知识点之间的依赖关系（前置 / 相关）维护在 `src/data/knowledge-relations.json`，会在页面右侧目录栏与知识图谱中展示。

## 本地验证

```bash
npm run docs:check    # 同步无漂移 + 链接与重定向合法
npm run privacy:check # 确认可发布文件不含个人隐私数据
npm run build         # 构建验证：frontmatter、图标的问题都会在这里暴露
```

## 提交变更

- 小改动：直接在页面右上角点「编辑此页」发起 PR。
- 仓库内工作：Fork → 新建分支 → 提交 → PR 到 `main`。
- 推送到 `main` 后 GitHub Actions 自动校验、构建并发布网站。
