# Knowledge Tree 维护入口

先读 `README.md`、`CONTRIBUTING.md` 与 `tech_stack/TECHNICAL_ARTICLE_TEMPLATE.md`。`SITE-AND-WORKFLOW.md` 为本地实现说明；它不入库，也不替代正式规范。

## 编辑边界

- 技术正文只改 `tech_stack/` 原件；生成文章由 `scripts/sync-docs.mjs` 管理。
- 全部正文（含 draft）一个 H1、六个固定 H2；主线可独立阅读，长证明/算例用 `details.kt-derivation`。
- 各级标题与折叠标题采用简洁的学术名词短语，避免问句、操作句、过细步骤和抽象口号。不添加重复中英译名或无信息括注；必要定义放在正文首次出现处，保留数学、单位、版本和必要消歧。
- 分类导读是手写 `index.mdx`，目录由 `TopicIndex.astro` 从注册表生成；项目和学习路径遵守各自页面结构。
- 先查看 git status，保留用户已有修改。移动原件默认保持 slug；只有明确换网址时生成重定向。
- 新增或迁移使用 docs:add / docs:rename；展示元数据只在 content-map 维护，relations 只维护 edges。
- 本地编辑使用 `knowledge-tree-editor`；同步到 GitHub 和网站使用 `knowledge-tree-maintainer`。仓库规范是跨机器可用的依据。

## 验证

内容修改：`npm run docs:sync`、`npm run docs:check`、`npm run privacy:check`、`npm run build`、`npm run site:check`。

维护脚本或进度状态变更：增加有行为意义的回归，运行 `npm test`。折叠、目录、搜索、打印等交互变更运行 `npm run test:browser`。PPO 示例按 CONTRIBUTING 运行独立 Python 测试。验证结束清理一次性输出，保留正式示例和回归测试。

## Development

Start the dev server in background mode:

```bash
npm run dev -- --background --host 127.0.0.1
```

Manage it with `npm run astro -- dev stop`, `npm run astro -- dev status`, and `npm run astro -- dev logs`. Production browser tests use `astro preview` after a build.

## Astro documentation

Consult the relevant official guide before changing its area:

- Routing: https://docs.astro.build/en/guides/routing/
- Components: https://docs.astro.build/en/basics/astro-components/
- Framework integration: https://docs.astro.build/en/guides/framework-components/
- Content: https://docs.astro.build/en/guides/content-collections/
- Styling: https://docs.astro.build/en/guides/styling/
- Internationalization: https://docs.astro.build/en/guides/internationalization/
