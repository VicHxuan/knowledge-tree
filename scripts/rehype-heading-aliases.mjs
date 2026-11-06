/** Preserve incoming heading fragments without adding obsolete titles to the ToC. */
import { readFileSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { rehypeHeadingIds } from '@astrojs/markdown-remark';
import { parseFragment } from 'parse5';
import { assignArticleModuleIds } from './lib/heading-aliases.mjs';
import { readArticleModules } from './lib/article-structure.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DATA = new URL('../src/data/heading-aliases.json', import.meta.url);

export default function rehypeHeadingAliases(options = {}) {
	const aliases = options.aliases ?? JSON.parse(readFileSync(DATA, 'utf8'));
	if (aliases.version !== 1 || !aliases.pages || typeof aliases.pages !== 'object') {
		throw new Error('heading-aliases: expected version 1 and a pages map');
	}
	const docsRoot = options.docsRoot ?? resolve(ROOT, 'src/content/docs');
	const articleSlugs = new Set(options.articleSlugs ?? JSON.parse(readFileSync(new URL('../src/data/content-map.json', import.meta.url), 'utf8')).entries.map((entry) => entry.slug));
	const moduleTitles = options.moduleTitles ?? readArticleModules();
	const collectHeadings = rehypeHeadingIds();
	return (tree, file) => {
		const path = file?.path || file?.history?.at(-1);
		if (!path) return;
		const local = relative(docsRoot, isAbsolute(path) ? path : resolve(ROOT, path));
		if (local.startsWith('..') || isAbsolute(local)) return;
		const slug = local.replaceAll('\\', '/').replace(/\.(md|mdx)$/, '');
		// These six IDs stay stable when module wording changes, and do not consume
		// a title-derived ID that may already belong to an older subsection link.
		if (articleSlugs.has(slug)) assignArticleModuleIds(tree, moduleTitles);
		const pageAliases = aliases.pages[slug];
		if (!pageAliases || !Object.keys(pageAliases).length) return;

		// Astro normally assigns IDs after user plugins. Its official helper gives us
		// the same IDs now; the final built-in pass preserves those explicit IDs.
		collectHeadings(tree, file);
		const ids = new Set();
		const headings = new Map();
		const rawIds = (node) => {
			for (const attr of node.attrs ?? []) if (attr.name === 'id') ids.add(attr.value);
			for (const child of node.childNodes ?? []) rawIds(child);
			if (node.content) rawIds(node.content);
		};
		const visit = (node) => {
			if (typeof node.properties?.id === 'string') {
				ids.add(node.properties.id);
				if (/^h[1-6]$/.test(node.tagName)) headings.set(node.properties.id, node);
			}
			// Native details/summary and hand-written anchors are still raw HTML at
			// this stage. Include their IDs before deciding whether an alias is free.
			if (node.type === 'raw') rawIds(parseFragment(node.value));
			for (const child of node.children ?? []) visit(child);
		};
		visit(tree);
		for (const [oldId, currentId] of Object.entries(pageAliases)) {
			if (!oldId || typeof currentId !== 'string' || !currentId) {
				throw new Error(`heading-aliases: invalid alias on ${slug}`);
			}
			if (ids.has(oldId)) continue;
			const heading = headings.get(currentId);
			if (!heading) throw new Error(`heading-aliases: ${slug} #${oldId} targets missing heading #${currentId}`);
			heading.children.unshift({
				type: 'element', tagName: 'span',
				properties: {
					id: oldId,
					className: ['kt-heading-alias'],
					'aria-hidden': 'true',
					'data-pagefind-ignore': 'all',
					style: 'scroll-margin-top: calc(var(--sl-nav-height, 4rem) + 1.5rem)',
				},
				children: [],
			});
			ids.add(oldId);
		}
	};
}
