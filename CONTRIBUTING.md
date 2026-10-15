# 贡献指南

感谢你为知识树贡献内容！

## 两层内容模型（先读这个）

- **章节正文**：写在 `tech_stack/` 下的原始笔记里（事实源）。`src/content/docs/` 中对应的页面由 `scripts/sync-docs.mjs` **自动生成**（untracked），直接改生成产物会在下次同步时被覆盖。
- **手写页面**：板块首页（各目录的 `index.md`）、`learning-paths/`、`knowledge-map/`、`projects/` 直接维护在 `src/content/docs/` 下。

不确定某篇正文该改哪里？点页面上角的「编辑此页」，链接会指向事实源。

## 新增或修改章节

1. 原始笔记放在 `tech_stack/` 对应领域目录下，文件名使用小写英文与连字符（kebab-case）；
2. 运行 `npm run docs:add -- tech_stack/<域>/<板块>/<文件>.md` 注册文章：自动校验路径、从首个 H1 推导标题、生成 slug 并同时写入 `content-map.json` 与 `knowledge-relations.json`（可用 `--title` / `--label` / `--status` / `--order` / `--description` 覆盖默认值）。也可以手工在 `src/data/content-map.json` 注册映射（source / output / slug / title / label / order / status）；
3. 运行 `npm run docs:sync` 生成页面；状态为 `draft` 的条目会带 `draft: true`，只在开发服务器可见、不进生产构建；
4. 正文从二级标题 `##` 开始组织，**不要**写一级标题——页面大标题由映射表的 `title` 渲染（与标题相同的 H1 会被自动吸收）；已发布文章须符合**六模块结构**（见下文写作规范），`docs:check` 会逐篇校验；
5. 数学公式使用 `$...$`（行内）与 `$$...$$`（块级），由 KaTeX 渲染；
6. 架构与流程图使用 ` ```mermaid ` 代码块书写（flowchart、sequenceDiagram、graph 等），构建时保留源码并在页面加载后按当前主题客户端渲染——只在不影响文章正文的场合使用，复杂图示仍优先手绘图片；
7. 原始笔记中的站内相对链接会被同步脚本自动转换；站内跳转写完整路径并带 `/knowledge-tree` 前缀，例如：

   ```md
   [PPO](/knowledge-tree/learning-based-planning/rl-algorithms/ppo/)
   ```

移动 / 重命名已有文章：先 `git mv` 原件到新路径，再运行
`npm run docs:rename -- --from <旧slug> --to tech_stack/<新路径>.md`。脚本会同步改写映射条目与关系边端点，旧文曾发布过时还会自动在 `redirects.json` 登记旧 slug 的兜底跳转。

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

## 技术文章写作规范（Technical Article Writing Standards）

### 规范入口

所有底层技术教程（数学理论、算法推导、论文方法、工程工具）统一遵循
[Technical Article Template](./tech_stack/TECHNICAL_ARTICLE_TEMPLATE.md)。
该文件既是写作规范，也是可复制的文章骨架；它属于知识库规范文件，不进入网站的教程发布映射。

### 文章结构要求

底层技术文章统一采用**六模块写作体系**，模块顺序固定：

1. **概述（Overview）**：背景、问题动机、应用场景与文章定位。
2. **基础与定义（Foundations）**：先修概念、问题建模、输入输出定义、变量与符号、坐标系与基本假设。
3. **核心原理（Core Principles）**：方法思想、算法架构、数学推导、关键公式与执行流程——通常是算法与理论文章的主体。
4. **实现与应用（Implementation & Applications）**：伪代码与实现、命令与配置、算法参数、实验示例与应用案例。
5. **分析与总结（Analysis & Summary）**：方法特点、性能与适用条件、局限、工程常见问题与核心结论。
6. **关联与参考（Related Topics & References）**：前置知识、相关技术、进阶阅读、研究项目与参考资料（论文、官方文档、开源仓库）。

排版约定：原件使用一个 `# 文章标题`；六个主模块统一为 `##`，模块内使用 `###`，复杂推导再使用 `####`。三级及以下标题随技术内容自由扩展。原文没有材料的模块写 `> 待补充。`，不凭空生成技术事实。

### 文档内容管理与同步关系

项目采用双层内容架构：

- `tech_stack/`：技术文章原件，唯一维护入口；
- `src/content/docs/`：网站发布内容（生成副本 + 少量手写页面）；
- `astro.config.mjs`：网站路由、侧栏与布局配置。

同步关系（依据 `scripts/` 的实际实现）：

- `scripts/sync-docs.mjs` 按 `src/data/content-map.json` 的映射，把 `tech_stack/` 原件确定性生成为 Starlight 页面（注入 frontmatter、吸收与页面标题一致的 H1、转换站内链接、`draft` 状态只在开发态可见），并统计各域文章数、最近更新与每篇的最近维护日期（取原件的 git author-date）；`predev` / `prebuild` 钩子在每次 `dev` / `build` 前自动执行，`--check` 模式用于检测漂移；
- `scripts/validate-docs.mjs` 校验站内链接与重定向（含重定向环检测）、已发布文章的六模块标题结构，以及 content-map / knowledge-relations / redirects 之间的数据一致性（slug 唯一、边端点存在、前置依赖成无环 DAG 等）；
- `scripts/docs-add.mjs` / `scripts/docs-rename.mjs` 是新增 / 移动文章的登记器（见上文用法），注册后仍需人工补充 `knowledge-relations.json` 的前置 / 相关边；
- `scripts/privacy-check.mjs` 在提交前扫描可发布文件中的个人隐私数据；
- 六大领域的目录名、名称与配色统一登记在 `src/data/domains.mjs`（侧栏分组、同步、校验、知识图谱共用）；新增领域时先在这里注册，再建目录。

因此：**技术正文一律修改 `tech_stack/` 原件**，网站副本由脚本重新生成；手改生成页面会在下次同步时被覆盖。

### 新增文章检查清单

新增技术文章时：

1. 使用统一模板（六模块结构，从规范文件复制骨架）；
2. 遵循现有目录与文件命名规范（数字前缀 = 领域内章节顺序）；
3. 正确设置标题层级（一个 H1 + 六个 `##` 模块，模块内 `###`/`####`）；
4. 在「基础与定义」给出必要的概念、符号与变量定义；
5. 保持公式、符号及代码准确（LaTeX 书写公式，代码块标明语言）；
6. 在「关联与参考」添加相关知识和参考资料（先修、进阶、论文、文档、仓库均给明确链接）；
7. 检查 Markdown 在 VS Code 与网站渲染均正常（本地 `npm run dev` 预览）。

### 已有文章维护原则

调整已有 Markdown 结构时：

- 尽可能保留全部原始技术内容；以章节重排、标题分级和段落归类为主要修改方式；
- 完整保留公式、变量、推导、代码、命令、图片、表格、实验结果和引用；
- 技术纠错与知识补充通过独立的内容修订任务处理，不与结构重排混在一起；
- 对内容不足的模块明确标记 `> 待补充。`，并在任务报告中登记；
- 保证文档在 VS Code Markdown 和 Astro + Starlight 中正常渲染。

### Pull Request 检查规则

提交技术文章时，确认以下各项：

- [ ] 文章结构符合模板（六模块齐全，缺失模块有 `> 待补充。` 标记）；
- [ ] 标题层级正确（一个 H1，模块 `##`，子章节逐级递进）；
- [ ] 数学公式和代码完整（LaTeX 正确、代码块语言标注、内容可复现）；
- [ ] 内部链接与图片有效（`npm run docs:check` 通过）；
- [ ] 技术来源清晰（推导、结论、参数有出处或复现条件）；
- [ ] 网站构建通过（`npm run build` 无错误）；
- [ ] 公开文件符合项目隐私规则（`npm run privacy:check` 通过）。

### 风格与安全补充

- 正文内容保持原始笔记风格，修改他人内容时尽量做最小改动。
- 引用其他知识点时优先链接站内页面，而不是复制内容。
- 涉及内部项目的内容，注意不要包含真实服务器地址、密码或密钥。
- 知识点之间的依赖关系（前置 / 相关）维护在 `src/data/knowledge-relations.json`，会在页面右侧目录栏与知识图谱中展示。

## 本地验证

```bash
npm run docs:check    # 同步无漂移 + 链接/重定向合法 + 六模块结构 + 数据一致性
npm run privacy:check # 确认可发布文件不含个人隐私数据
npm run build         # 构建验证：frontmatter、图标的问题都会在这里暴露
```

新增 / 移动文章的登记命令：

```bash
npm run docs:add -- tech_stack/<域>/<板块>/<文件>.md [--title …] [--status published|draft]
npm run docs:rename -- --from <旧slug> --to tech_stack/<新路径>.md
```

## 提交变更

- 小改动：直接在页面右上角点「编辑此页」发起 PR。
- 仓库内工作：Fork → 新建分支 → 提交 → PR 到 `main`。
- 推送到 `main` 后 GitHub Actions 自动校验、构建并发布网站。
