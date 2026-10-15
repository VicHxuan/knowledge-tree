#!/usr/bin/env node
/**
 * docs-add.mjs — 新增文章的注册向导（P2 工作流减负）。
 *
 * 用法：
 *   node scripts/docs-add.mjs tech_stack/<域>/<板块>/<文件>.md \
 *     [--title "文章标题"] [--label "侧栏短标签"] [--status published|draft] \
 *     [--order N] [--description "一句话描述"]
 *
 * 做什么：
 *   1. 校验路径：域目录必须在 src/data/domains.mjs 注册、文件必须已存在、
 *      未被 content-map 注册过；
 *   2. 从文件首个 H1 / 文件名 / 参数推导 title，从 title 推导 label；
 *   3. 生成 slug（域/板块/kebab 文件名）、output、order（板块内自增）；
 *   4. 把条目按域/板块顺序插入 src/data/content-map.json（保持稳定排序）；
 *   5. 在 knowledge-relations.json 追加节点（挂在同板块节点之后，无边——
 *      前置/相关关系需要人工判断，脚本只保证结构合法）。
 *
 * 不做什么：不写正文、不加 redirects（重命名场景用 docs-rename）、
 * 不自动跑 sync/validate——完成后请按 CONTRIBUTING 跑四连校验。
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { join, resolve, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DOMAINS } from '../src/data/domains.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MAP_PATH = join(ROOT, 'src/data/content-map.json');
const REL_PATH = join(ROOT, 'src/data/knowledge-relations.json');

const args = process.argv.slice(2);
const flag = (name, fallback) => {
	const i = args.indexOf(`--${name}`);
	return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const source = args.find((a) => !a.startsWith('--'));
if (!source || !source.startsWith('tech_stack/')) {
	console.error('用法: node scripts/docs-add.mjs tech_stack/<域>/<板块>/<文件>.md [--title …] [--label …] [--status published|draft] [--order N] [--description …]');
	process.exit(1);
}

const rel = source.slice('tech_stack/'.length); // 域内相对路径
const abs = join(ROOT, source);
const domainDir = rel.split('/')[0];
const domain = DOMAINS.find((d) => d.tsDir === domainDir);
if (!domain) {
	console.error(`✗ 域目录 "${domainDir}" 未在 src/data/domains.mjs 注册。已注册: ${DOMAINS.map((d) => d.tsDir).join(', ')}`);
	process.exit(1);
}
if (!existsSync(abs)) {
	console.error(`✗ 文件不存在（请先用模板创建正文）: ${source}`);
	process.exit(1);
}

// ---------- 从文件推导默认值 ----------
const raw = readFileSync(abs, 'utf8');
const h1 = raw.match(/^#\s+(.+)$/m)?.[1]?.trim();
const stem = basename(rel).replace(/\.md$/, '');
const stemNoNum = stem.replace(/^\d+_/, '');

const title = flag('title') ?? h1 ?? stemNoNum;
const status = flag('status', 'draft');
if (!['published', 'draft'].includes(status)) {
	console.error(`✗ 非法 status: ${status}（应为 published 或 draft）`);
	process.exit(1);
}
const label = flag('label') ?? (title.length <= 12 ? title : title.slice(0, 11) + '…');
const description = flag('description') ?? `${title}——${domain.label}板块文章。`;

// ---------- 计算板块与 slug ----------
const parts = rel.split('/'); // [域目录, 板块目录?, 文件.md]
const sectionDir = parts.length >= 3 ? parts[parts.length - 2] : null;
const fileName = parts[parts.length - 1];
const slug = [
	domain.id,
	...(sectionDir ? [sectionDir] : []),
	fileName.replace(/\.md$/, ''),
].join('/');

// ---------- content-map ----------
const map = JSON.parse(readFileSync(MAP_PATH, 'utf8'));
if (map.entries.some((e) => e.source === source)) {
	console.error(`✗ content-map 已注册该 source: ${source}`);
	process.exit(1);
}
if (map.entries.some((e) => e.slug === slug)) {
	console.error(`✗ slug 冲突: ${slug}（换文件名或先处理旧条目）`);
	process.exit(1);
}

// order：同板块内最大 order + 10（板块 = slug 去掉最后一段）
const sectionPrefix = slug.split('/').slice(0, -1).join('/');
const sameSection = map.entries.filter((e) => e.slug.startsWith(`${sectionPrefix}/`));
const order = Number(flag('order')) || Math.max(0, ...sameSection.map((e) => e.order ?? 0)) + 10;

const entry = {
	source,
	output: `src/content/docs/${slug}.md`,
	slug,
	title,
	label,
	order,
	status,
	description,
};

// 稳定插入：放在同板块最后一条之后（无则放同域最后一条之后，再无则追加到末尾），
// 不做全局排序，避免整文件重排制造无谓 diff
map.entries.splice(insertionIndex(map.entries.map((e) => e.slug), slug), 0, entry);
writeFileSync(MAP_PATH, JSON.stringify(map, null, 2) + '\n');

// ---------- knowledge-relations ----------
const relations = JSON.parse(readFileSync(REL_PATH, 'utf8'));
if (!relations.nodes.some((n) => n.id === slug)) {
	relations.nodes.splice(
		insertionIndex(relations.nodes.map((n) => n.id), slug),
		0,
		{
			id: slug,
			label,
			title,
			domain: domain.id,
			domainLabel: domain.label,
			status,
			order,
		},
	);
	writeFileSync(REL_PATH, JSON.stringify(relations, null, 2) + '\n');
}

/** 就近插入位置：同板块末尾 → 同域末尾 → 文件末尾 */
function insertionIndex(existingSlugs, newSlug) {
	const section = newSlug.split('/').slice(0, -1).join('/');
	const domainId = newSlug.split('/')[0];
	let idx = -1;
	for (let i = 0; i < existingSlugs.length; i++) {
		const s = existingSlugs[i];
		if (s.startsWith(`${section}/`) || (!section && s.startsWith(`${domainId}/`))) idx = i + 1;
	}
	if (idx >= 0) return idx;
	for (let i = 0; i < existingSlugs.length; i++) {
		if (existingSlugs[i].startsWith(`${domainId}/`)) idx = i + 1;
	}
	return idx >= 0 ? idx : existingSlugs.length;
}

console.log('✓ 已注册新文章：');
console.log(`    source:  ${entry.source}`);
console.log(`    slug:    ${entry.slug}`);
console.log(`    title:   ${entry.title}`);
console.log(`    label:   ${entry.label}`);
console.log(`    order:   ${entry.order}`);
console.log(`    status:  ${entry.status}`);
console.log('');
console.log('后续步骤：');
console.log('  1. 在 knowledge-relations.json 里补充该文的 prerequisite / related 边（图谱与右栏关联区依赖它）；');
console.log('  2. 若从旧文移动/重命名而来，在 redirects.json 加「旧 slug → 新 slug」；');
console.log('  3. 跑四连校验：npm run docs:sync && npm run docs:check && npm run privacy:check && npm run build');
