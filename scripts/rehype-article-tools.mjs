/**
 * rehype-article-tools.mjs — 在已发布文章正文顶部注入「文章信息条」。
 *
 * 注入内容（只作用于 content-map 内 status=published 的生成页，手写页/草稿页跳过）：
 *   <aside class="kt-article-bar">
 *     <nav class="kt-breadcrumb">领域 › 板块</nav>
 *     <div class="kt-article-actions">
 *       <span class="kt-article-date">最近维护 YYYY-MM-DD</span>
 *       <kt-actions slug title></kt-actions>   ← Footer 脚本定义的自定义元素（书签/反馈）
 *     </div>
 *   </aside>
 *
 * 数据来源：
 *   - 面包屑：slug 目录层级 + src/data/domains.mjs（域标签）+ 板块 index.md 的 frontmatter title
 *   - 最近维护日期：src/data/content-stats.json 的 lastModified 映射
 *     （由 scripts/sync-docs.mjs 用 git author-date 预先算好——构建期不调 git，
 *      生成页 untracked，日期取的是 tech_stack/ 原件的提交历史）
 *
 * 页面定位：Astro 在 vfile 上提供 file.data.astro.frontmatter，但 slug 以
 * file.path（src/content/docs/ 下的绝对路径）反推更稳，不依赖 frontmatter 形态。
 */
import { readFileSync } from 'node:fs';
import { dirname, isAbsolute, join, posix, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DOMAINS, DOMAIN_BY_ID } from '../src/data/domains.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DOCS = join(ROOT, 'src/content/docs');
const SITE_BASE = '/knowledge-tree';

const MAP = JSON.parse(readFileSync(join(ROOT, 'src/data/content-map.json'), 'utf8'));
const BY_SLUG = new Map(MAP.entries.map((e) => [e.slug, e]));

// stats.json 由 predev/prebuild 钩子（sync-docs）先行生成；缺失或过期时静默降级为不显示日期
let LAST_MODIFIED = {};
try {
	LAST_MODIFIED = JSON.parse(readFileSync(join(ROOT, 'src/data/content-stats.json'), 'utf8')).lastModified ?? {};
} catch {
	/* 构建入口未先跑 sync：文章条照常注入，只是不带日期 */
}

/** 板块 index.md 的 frontmatter title 缓存（构建期多次调用） */
const SECTION_TITLE = new Map();
function sectionTitle(domainId, sectionDir) {
	const key = `${domainId}/${sectionDir}`;
	if (SECTION_TITLE.has(key)) return SECTION_TITLE.get(key);
	let title = null;
	try {
		const fm = readFileSync(join(DOCS, domainId, sectionDir, 'index.md'), 'utf8');
		const m = fm.match(/^title:\s*(?:"([^"]*)"|'([^']*)'|(.+?))\s*(?:#.*)?$/m);
		if (m) title = (m[1] ?? m[2] ?? m[3] ?? '').trim() || null;
	} catch {
		/* 板块没有 index.md：面包屑退化为目录名 */
	}
	SECTION_TITLE.set(key, title);
	return title;
}

const link = (href, label) => ({
	type: 'element',
	tagName: 'a',
	properties: { className: ['kt-crumb'], href },
	children: [{ type: 'text', value: label }],
});
const sep = () => ({
	type: 'element',
	tagName: 'span',
	properties: { className: ['kt-crumb-sep'], 'aria-hidden': true },
	children: [{ type: 'text', value: '›' }],
});

/** 领域 › 板块（文章自身的标题由页面 H1 承担，不进面包屑） */
function breadcrumb(slug) {
	const parts = slug.split('/');
	const crumbs = [];
	const domain = DOMAIN_BY_ID[parts[0]];
	if (domain) {
		crumbs.push(link(`${SITE_BASE}/${domain.id}/`, domain.label));
		for (let i = 1; i < parts.length - 1; i++) {
			const prefix = parts.slice(0, i + 1).join('/');
			crumbs.push(sep(), link(`${SITE_BASE}/${prefix}/`, sectionTitle(parts[0], parts[i]) ?? parts[i]));
		}
	} else {
		crumbs.push(link(`${SITE_BASE}/${parts[0]}/`, parts[0]));
	}
	return crumbs;
}

function articleBar(entry) {
	const date = LAST_MODIFIED[entry.slug];
	const children = [
		{
			type: 'element',
			tagName: 'nav',
			properties: { className: ['kt-breadcrumb'], 'aria-label': '面包屑' },
			children: breadcrumb(entry.slug),
		},
		{
			type: 'element',
			tagName: 'div',
			properties: { className: ['kt-article-actions'] },
			children: [
				...(date
					? [{
							type: 'element',
							tagName: 'span',
							properties: { className: ['kt-article-date'], title: '依据 tech_stack 原件最近一次提交' },
							children: [{ type: 'text', value: `最近维护 ${date.slice(0, 10)}` }],
						}]
					: []),
				{
					type: 'element',
					tagName: 'kt-actions',
					properties: { slug: entry.slug, title: entry.title },
					children: [],
				},
			],
		},
	];
	return {
		type: 'element',
		tagName: 'aside',
		properties: { className: ['kt-article-bar'], 'data-slug': entry.slug },
		children,
	};
}

export default function rehypeArticleTools() {
	return (tree, file) => {
		const raw = file?.path || file?.history?.[file.history.length - 1];
		if (!raw) return;
		const abs = isAbsolute(raw) ? raw : resolve(ROOT, raw);
		const rel = relative(DOCS, abs);
		if (rel.startsWith('..')) return; // docs 集合之外
		const slug = posix.normalize(rel.split('\\').join('/')).replace(/\.(md|mdx)$/, '');
		const entry = BY_SLUG.get(slug);
		if (!entry || entry.status !== 'published') return;
		tree.children?.unshift(articleBar(entry));
	};
}
