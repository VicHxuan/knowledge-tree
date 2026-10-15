#!/usr/bin/env node
/**
 * docs-rename.mjs — 移动 / 重命名文章的登记器（P2 工作流减负）。
 *
 * 用法（先用 git mv 移动 tech_stack 原件，再运行本脚本）：
 *   node scripts/docs-rename.mjs --from <旧 slug> --to tech_stack/<新路径>.md \
 *     [--new-title "新标题"] [--new-label "新侧栏标签"]
 *
 * 做什么：
 *   1. 校验：旧 slug 已注册、新文件已存在、新 slug 未被占用；
 *   2. 从新路径推导新 slug（域目录 → 域 id；文件名去 NN_ 前缀、小写）；
 *   3. 更新 content-map 条目（source / output / slug / 可选 title / label）；
 *   4. 同步 knowledge-relations：节点 id + 所有引用旧 slug 的边端点；
 *   5. 旧文曾发布过（status=published）则在 redirects.json 登记「旧 slug → 新 slug」。
 *
 * 不做什么：不动正文、不动 git（请用 git mv 保留历史）、不自动跑校验。
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, resolve, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DOMAINS } from '../src/data/domains.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MAP_PATH = join(ROOT, 'src/data/content-map.json');
const REL_PATH = join(ROOT, 'src/data/knowledge-relations.json');
const REDIR_PATH = join(ROOT, 'src/data/redirects.json');

const args = process.argv.slice(2);
const flag = (name) => {
	const i = args.indexOf(`--${name}`);
	return i >= 0 && args[i + 1] && !args[i + 1].startsWith('--') ? args[i + 1] : null;
};
const fromSlug = flag('from');
const toPath = flag('to');
if (!fromSlug || !toPath || !toPath.startsWith('tech_stack/')) {
	console.error('用法: node scripts/docs-rename.mjs --from <旧 slug> --to tech_stack/<新路径>.md [--new-title …] [--new-label …]');
	process.exit(1);
}

const map = JSON.parse(readFileSync(MAP_PATH, 'utf8'));
const entry = map.entries.find((e) => e.slug === fromSlug);
if (!entry) {
	console.error(`✗ content-map 中不存在旧 slug: ${fromSlug}`);
	process.exit(1);
}
const abs = join(ROOT, toPath);
if (!existsSync(abs)) {
	console.error(`✗ 新文件不存在（请先 git mv 原件到新路径）: ${toPath}`);
	process.exit(1);
}

// ---------- 从新路径推导新 slug ----------
const rel = toPath.slice('tech_stack/'.length);
const parts = rel.split('/'); // [域目录, …板块目录, 文件.md]
const domain = DOMAINS.find((d) => d.tsDir === parts[0]);
if (!domain) {
	console.error(`✗ 域目录 "${parts[0]}" 未在 src/data/domains.mjs 注册`);
	process.exit(1);
}
const stem = basename(parts[parts.length - 1]).replace(/\.md$/, '').replace(/^\d+_/, '');
const newSlug = [domain.id, ...parts.slice(1, -1), stem].join('/').toLowerCase();
if (newSlug === fromSlug) {
	console.error(`✗ 新旧 slug 相同（${newSlug}）——这不需要 rename`);
	process.exit(1);
}
const conflict = map.entries.find((e) => e.slug === newSlug && e.slug !== fromSlug);
if (conflict) {
	console.error(`✗ 新 slug 已被占用: ${newSlug}（${conflict.source}）`);
	process.exit(1);
}

// ---------- 更新 content-map ----------
const newTitle = flag('new-title') ?? entry.title;
const newLabel = flag('new-label') ?? entry.label;
Object.assign(entry, {
	source: toPath,
	output: `src/content/docs/${newSlug}.md`,
	slug: newSlug,
	title: newTitle,
	label: newLabel,
});
writeFileSync(MAP_PATH, JSON.stringify(map, null, 2) + '\n');

// ---------- 更新 knowledge-relations ----------
const relations = JSON.parse(readFileSync(REL_PATH, 'utf8'));
const node = relations.nodes.find((n) => n.id === fromSlug);
if (node) {
	node.id = newSlug;
	node.title = newTitle;
	node.label = newLabel;
	if (node.domain !== domain.id) {
		node.domain = domain.id;
		node.domainLabel = domain.label;
	}
}
let edgeHits = 0;
for (const e of relations.edges) {
	if (e.from === fromSlug) { e.from = newSlug; edgeHits++; }
	if (e.to === fromSlug) { e.to = newSlug; edgeHits++; }
}
writeFileSync(REL_PATH, JSON.stringify(relations, null, 2) + '\n');

// ---------- redirects（旧链接兜底） ----------
let redirectAdded = false;
if (entry.status === 'published') {
	const redirects = JSON.parse(readFileSync(REDIR_PATH, 'utf8'));
	if (redirects[fromSlug]) {
		console.warn(`⚠ redirects.json 已存在 "${fromSlug}" 条目（${fromSlug} → ${redirects[fromSlug]}），请人工确认是否需要改为指向 ${newSlug}`);
	} else {
		redirects[fromSlug] = `/${newSlug}`;
		writeFileSync(REDIR_PATH, JSON.stringify(redirects, null, 2) + '\n');
		redirectAdded = true;
	}
}

console.log('✓ 重命名登记完成：');
console.log(`    slug:   ${fromSlug}`);
console.log(`        →   ${newSlug}`);
console.log(`    source: ${toPath}`);
console.log(`    relations: 更新 ${node ? '节点 + ' : ''}${edgeHits} 处边引用`);
console.log(redirectAdded ? '    redirect: 已登记旧 slug 兜底' : '    redirect: 未登记（草稿或已有条目）');
console.log('');
console.log('后续步骤：');
console.log('  1. 检查其他文章内指向旧文件的相对 .md 链接（sync 会按新路径自动改写，链接文字里的旧提法需人工核对）；');
console.log('  2. 跑四连校验：npm run docs:sync && npm run docs:check && npm run privacy:check && npm run build');
