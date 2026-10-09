# 贡献指南

感谢你为知识树贡献内容！

## 新增或修改文档

1. 文档位于 `src/content/docs/`，**文件夹即知识分类**，每个 `.md` / `.mdx` 文件生成一个页面。
2. 文件名使用小写英文与连字符（kebab-case），正文与标题可以使用中文。
3. 每篇文档需要 frontmatter：

   ```md
   ---
   title: PPO 强化学习算法
   description: 一句话简介，用于搜索与页面摘要
   sidebar:
     order: 5
   ---
   ```

4. 正文从二级标题 `##` 开始组织，**不要**写一级标题——页面大标题由 frontmatter 的 `title` 渲染。
5. 数学公式使用 `$...$`（行内）与 `$$...$$`（块级），由 KaTeX 渲染。
6. 站内跳转写完整路径并带 `/knowledge-tree` 前缀，例如：

   ```md
   [PPO](/knowledge-tree/machine-learning/reinforcement-learning/ppo/)
   ```

7. 新增分类目录时，在目录下放置 `index.md` 作为分组首页，并检查 `astro.config.mjs` 中 `sidebar` 的分组配置。

## 撰写规范

- 正文内容保持原始笔记风格，修改他人内容时尽量做最小改动。
- 引用其他知识点时优先链接站内页面，而不是复制内容。
- 涉及内部项目的内容，注意不要包含真实服务器地址、密码或密钥。

## 本地验证

```bash
npm run dev     # 本地实时预览 http://localhost:4321/knowledge-tree/
npm run build   # 构建验证：frontmatter、链接、图标的问题都会在这里暴露
```

## 提交变更

- 小改动：直接在页面右上角点「编辑此页」发起 PR。
- 仓库内工作：Fork → 新建分支 → 提交 → PR 到 `main`。
- 推送到 `main` 后 GitHub Actions 自动构建并发布网站。
