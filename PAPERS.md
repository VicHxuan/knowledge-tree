# 文献维护

知识树维护技术正文和主题关系，Zotero 维护书目信息、全文、批注与阅读笔记。本地 `paper` 是下载收件箱和全文访问入口。

## 数据入口

| 内容 | 维护位置 |
| --- | --- |
| 知识主题与文章 | `tech_stack/`、`src/data/content-map.json` |
| 文献标题、作者、日期、出版信息 | Zotero 原条目 |
| 公开论文目录 | `src/data/papers.json`，从 Zotero 选定条目生成 |
| 论文与文章的关系 | `src/data/paper-relations.json` |
| Zotero 条目与网站文献 ID 的映射 | `paper/maintenance/site-map.json`，仅本地 |
| PDF、批注、阅读笔记 | Zotero 托管附件与笔记 |
| 新下载全文 | `paper/inbox/` |
| 已入库全文入口 | `paper/library/`，指向 Zotero 原件的符号链接 |
| 待补全文、待核对条目 | `paper/catalog/` |

`paper` 默认是本仓库的同级目录；Python 命令支持 `--paper` 指定其他位置。迁移到另一台机器时，恢复 Zotero 数据并重建本地全文入口。

## 分类

Zotero 的“知识树”根分类按六个知识领域组织，主题粒度对应文章分组。同一条目可以进入多个分类。跨领域论文按其方法用途归类，未纳入知识主线的收藏进入“扩展参考”。旧分类保留在“旧分类归档”中。

已有文献条目、附件记录和本地可读全文分别统计。阅读状态由用户维护，全文状态由工具读取文件检查结果生成。

## 常规刷新

启动 Zotero，启用设置中的本机应用通信，然后执行：

```bash
npm run papers:refresh
npm run docs:sync
npm run docs:check
```

刷新只读取 Zotero，更新本地目录和公开书目。公开数据采用字段白名单，包含标题、作者、年份、来源及已选资源链接。笔记、附件路径、阅读状态和授权密钥保持本地。

构建使用已导出的 JSON，CI 和网站浏览器无需连接 Zotero。

## 导入全文

1. 把 PDF 放入 `paper/inbox/`。
2. 核对目标论文的标题、作者和 DOI 或 arXiv 标识；识别参考文献及同页相邻论文中的标识，不能直接采用 PDF 中第一个 DOI。
3. 匹配现有文献。哈希只用于判定文件是否完全一致；预印本和发表版本使用文献标识与标题核对。
4. 准备本地导入清单，核对后通过官方本地 API 写入。
5. 验证导入附件的 SHA-256，再整理收件箱中的源文件。

```bash
npm run papers:authorize
npm run papers:apply -- --manifest /path/to/reviewed-import.json
npm run papers:refresh
```

Zotero 会通过原生对话框授权写入。密钥保存在 `paper/maintenance/.zotero-auth.json`，文件权限为 0600。工具只向本机 23119 端口发送凭据，检查 Zotero 实例标识和对象版本。

导入清单示例：

```json
{
  "collections": ["知识树/03_学习式规划/强化学习/无模型方法"],
  "imports": [{
    "id": "stable-import-id",
    "data": {
      "itemType": "preprint",
      "title": "Verified paper title",
      "creators": [{"creatorType": "author", "firstName": "Given", "lastName": "Family"}],
      "date": "2026",
      "url": "https://example.org/verified-paper"
    },
    "collections": ["知识树/03_学习式规划/强化学习/无模型方法"],
    "file": "inbox/paper.pdf"
  }]
}
```

`id` 是该导入任务的稳定标识。工具保存父条目和附件键后继续上传，失败重试会复用已创建对象。已有附件只有通过哈希比对后才复用。多条文献匹配时停止该导入，保留候选供核对。导入工具不会自动删除源文件或合并条目。

批量重构前备份 Zotero 数据、附件和分类映射。通过 Zotero API 或客户端修改文库，不能直接写 SQLite。对象版本冲突时重新核对当前条目，不覆盖其他修改。

## 官网版本补充

新增全文优先使用出版社、会议论文集和发布机构官网。按当前收集要求，下载时排除 arXiv 版本。作者书稿、投稿稿、正式发表版和补充材料分别记录；补充材料不能替代主论文。已有附件和批注继续保留。

下载默认在后台进行。需要浏览器时使用独立任务浏览器，结束后仅关闭本次创建的窗口并清理临时登录状态。遇到机构访问限制或验证码时记录原因，停止该站的自动下载。

每份 PDF 核对标题、作者、出版标识和完整性。图像型 PDF 通过渲染封面核对。核验后的本地清单使用以下格式：

```json
{
  "editions": [{
    "itemKey": "ZOTERO_KEY",
    "file": "inbox/official/paper.pdf",
    "sourceUrl": "https://publisher.example/paper",
    "version": "published",
    "sha256": "reviewed-file-sha256",
    "reviewed": true,
    "evidence": "标题、作者与会议出版标识一致"
  }]
}
```

`version` 支持 `published`、`technical-report`、`supplement`、`author-book`。已有附件可以提供 `existingAttachment` 和 `origin: "existing"`，表示复核原文件。下载链接使用稳定的公开地址，排除登录会话和签名下载地址。

```bash
npm run papers:official -- --manifest /path/to/reviewed-editions.json
npm run papers:refresh
```

工具校验文件格式、版本标记与 SHA-256，保存附件键后上传，失败重试复用同一附件。正式回执位于 `paper/maintenance/official-imports.json`。导入成功后可将清单中的文件路径更新为 Zotero 原件路径，再清理已核验的收件箱副本。

`catalog/official-fulltext.csv` 记录全部文献的版本核验结果；`catalog/missing-official-fulltext.csv` 是正式版本补充队列。它与 `missing-fulltext.csv` 的区别是：已有预印本仍可能缺少正式版，旧附件缺失也可能已有另一份可读全文。每次刷新检查实际文件和哈希，不以历史下载记录代替当前状态。

## 网站关联

`paper/maintenance/site-map.json` 保存显式选入网站的条目及其公开资源：

```json
{
  "version": 1,
  "papers": [{
    "id": "trpo",
    "itemKey": "ZOTERO_KEY",
    "topics": ["learning-based-planning"],
    "sources": ["https://proceedings.mlr.press/v37/schulman15.html"],
    "openAccessUrl": "https://proceedings.mlr.press/v37/schulman15.pdf"
  }]
}
```

文献 ID 首次确定后保持稳定。条目合并时先更新本地条目映射；既有网站 ID 和文章关系保持稳定。多个旧网站 ID 合并前，应为已发布的固定链接设计兼容处理。

`src/data/paper-relations.json` 的每条关系包含 `paper`、`article`、`note`。`article` 使用现有文章 slug，`note` 说明该论文与文章的关系。不要由“已下载”推断“已阅读”或“已整合”。

已有文章的原文链接保持可用，构建会为已登记的引用添加“文献”入口，文章末尾提供关联论文筛选链接。论文目录展示公开书目和相关文章；草稿文章只显示名称与撰写中状态。公开链接使用 DOI、arXiv、出版社或作者官方页面。

论文目录属于辅助导航，不要求为每篇论文建立知识文章。已确认正式出版来源的条目优先链接出版社或会议页面；旧 arXiv 链接可保留为引用匹配别名。`openAccessUrl` 仅指向已核验且公开可访问的官网全文，机构订阅下载不标为开放获取。

## 验证

```bash
npm run papers:check
npm run docs:check
npm run privacy:check
npm test
python3 -m unittest discover -s tests -p 'test_zotero*.py'
npm run build
npm run site:check
npm run test:browser -- tests/browser/papers.spec.mjs
```

本地检查确保条目和附件键未意外变化、已有笔记与批注保留、导入哈希一致、重复运行不新增条目。网页检查覆盖分类筛选、引用跳转、草稿链接、移动端和无脚本阅读。同步到 GitHub 与发布网站仍按用户的发布指令执行。
