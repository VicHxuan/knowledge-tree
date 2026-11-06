# 贡献与维护指南

## 内容在哪里维护

| 内容 | 维护入口 |
|---|---|
| 技术正文，包括草稿 | `tech_stack/` 原件；遵循[六模块模板](tech_stack/TECHNICAL_ARTICLE_TEMPLATE.md) |
| 文章标题、路由、发布状态、排序 | `src/data/content-map.json` |
| 六大领域 | `src/data/domains.mjs` |
| 前置和相关关系 | `src/data/knowledge-relations.json` 的 edges |
| 旧网址兼容 | `src/data/redirects.json` |
| 分类导读、学习路径、项目页 | `src/content/docs/` 中的手写 MD/MDX |
| 网站交互与样式 | `src/components/`、`src/styles/` |

生成文章带所有权标记，由 sync 重建；不要手工编辑。分类页是 `index.mdx`，其 `TopicIndex` 从注册表派生文章目录及发布状态。节点展示信息从文章注册表派生，统计和下载文件由 sync 生成。

## 技术文章的统一结构

全部技术文章（含草稿）采用：**概述 → 基础与定义 → 核心原理 → 实现与应用 → 分析与总结 → 关联与参考**。原件一个 H1，网站自动吸收为页面标题；模块为 H2，主线小节为 H3。

标题采用简洁的学术名词短语，如“网络结构”“优势估计”“策略优化”。具体步骤、结论和限制条件写在正文，不堆进小标题。章节及折叠标题不附重复英文译名或“重要”“详细说明”等无信息括注；正文保留必要的首次术语定义、数学表达式、单位、版本和消歧说明。

按模板组织“结论先行、细节展开”。必要公式、假设和边界保持可见；证明、算例和长代码放在 `<details class="kt-derivation" id="稳定英文ID">` 中。主线与推导写作示范见 [PPO 原件](tech_stack/03_learning-based-planning/02_rl-algorithms/02_model-free-rl/04_ppo.md)。

修改技术内容时查原始论文、官方文档或实现。保留来源、符号和复现条件；缺失材料写 `> 待补充。`，发布状态与内容完善程度分开处理。

## 手写页面的统一结构

- 分类页：板块定位、阅读主线、主题目录、下一步。目录用 `TopicIndex`，不要再手抄文章标题和草稿状态。
- 学习路径：学习目标、开始之前、学习步骤、完成标准、延伸方向。文章步骤用 slug 引用，步骤顺序和教学说明人工维护；稳定步骤 ID 与历史迁移表不能按新排序重写。
- 项目页：项目定位、系统与知识关系、实践入口、进展与证据。记录实际已有的结果，计划与完成状态明确区分。

## 修改已有文章

1. 修改原件，必要时更新映射的 title/description。
2. 更新相关关系；原件链接可以写相对 `.md`（含 query/fragment），同步后转成 `/knowledge-tree/<slug>/`。
3. 同步、检查、构建，再检查受影响网页。结构变化时检查旧锚点。

批量修改已有标题前，可从当前生成页保存锚点基线。仅标题改名、层级与顺序不变时，使用以下命令维护 `src/data/heading-aliases.json`：

```bash
node scripts/heading-aliases.mjs capture /tmp/knowledge-tree-headings.json
# 编辑原件标题后：
npm run docs:sync
node scripts/heading-aliases.mjs update /tmp/knowledge-tree-headings.json
node scripts/heading-aliases.mjs update /tmp/knowledge-tree-headings.json --write
```

先审阅输出再写入；章节增删、重排或改层级时须人工核对映射。六个主模块使用稳定锚点 `kt-section-1` 至 `kt-section-6`；旧标题别名只在构建时生成，不显示在目录中。改名已发生但尚未同步时，原有生成页仍可作为基线；生成页已更新时应从对应旧版本恢复基线。完成验证后删除临时基线。

```bash
npm run docs:sync
npm run docs:check
npm run privacy:check
npm run build
npm run site:check
```

`docs:check` 只比较，不写入或删除。首次 checkout 没有生成文章，应先 sync。`site:check` 检查实际 HTML 的链接、资源、锚点、技术文章六模块及公式解析错误，也能覆盖 MDX 组件生成的链接。生产构建排除 draft；分类页保留其无链接的撰写中状态。

## 新增文章

从模板复制骨架，文件名采用 `NN_kebab-case.md`。数字前缀用于原件排序，网站排序取映射的 order。

```bash
npm run docs:add -- tech_stack/<领域>/<子目录>/<文件>.md --status draft
```

默认 slug 从注册领域与各级相对目录推导，并去掉数字前缀；可用 `--slug` 指定稳定路由。可用参数还有 `--title`、`--label`、`--order`、`--description`。添加后按教学需要补充关系边：`prerequisite` 从先修指向后续，`related` 为无方向关联。

## 移动或改名

先移动原件，再更新登记。默认保留公开 URL：

```bash
npm run docs:rename -- --from <原slug> --to tech_stack/<新路径>.md
```

确需换 URL 时显式传 `--new-slug`；脚本会更新关系端点、迁移标题锚点别名，并将旧网址和已有历史路由直接指向最终目标。同步后检查其他原件的相对链接。删除已发布文章时为旧 URL 提供合理去向，不让它静默失效。

## 验证与预览

```bash
npm test                     # 维护脚本、数据与进度迁移回归
npm run build
npm run site:check
npm run test:browser          # 自动启用生产预览，测试折叠和学习路径
```

浏览器测试优先使用 `CHROME_PATH`，本机有 `/usr/bin/google-chrome` 时自动使用；否则先运行 `npx playwright install chromium`。测试输出在 `test-results/`，修复失败后删除不再需要的截图与 trace。

PPO 教学代码另需 Python + PyTorch：

```bash
python -m unittest discover -s tests -p 'test_ppo_core.py'
python examples/ppo_core.py
```

算法示例验证 GAE、边界与单轮梯度，不用长时间训练替代这些检查。代码的运行依赖独立于网站的 npm 依赖。

本地开发用后台模式：

```bash
npm run dev -- --background --host 127.0.0.1
npm run astro -- dev status
npm run astro -- dev stop
```

生产预览为 `npm run preview -- --host 127.0.0.1`，需先构建；搜索在生产构建中验收。

## 提交与发布

本机提供两个 skill：`knowledge-tree-editor` 负责本地文档编辑与预览；`knowledge-tree-maintainer` 负责将已修改的内容同步到 GitHub 和网站，并核实对应部署结果。明确要求“同步到 GitHub 并更新网站”时，按该范围完成检查、提交、推送与线上验收。

提交前检查具体 diff，定向暂存原件、配置和手写页面。`SITE-AND-WORKFLOW.md` 是本地实现说明，不入库；`tech_stack/personal/`、`tech_stack/references/` 和隐私模式库保持本地。

本地有隐私规则时运行扫描；无规则的 CI 会明确显示跳过。推送 main 会自动触发校验、构建和 GitHub Pages 发布。使用当前仓库的 SSH remote，按用户的提交/发布指令执行。
