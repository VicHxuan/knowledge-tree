import test from 'node:test';
import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createMarkdownProcessor } from '@astrojs/markdown-remark';
import { parseFragment } from 'parse5';
import rehypeHeadingAliases from '../scripts/rehype-heading-aliases.mjs';
import { deriveAliases } from '../scripts/heading-aliases.mjs';

const docsRoot = resolve('/virtual/docs');
const snapshot = (headings) => ({ version: 1, pages: { article: headings.map(([slug, depth = 2]) => ({ slug, depth, text: slug })) } });
const collect = (node, predicate, found = []) => {
	if (predicate(node)) found.push(node);
	for (const child of node.childNodes ?? []) collect(child, predicate, found);
	return found;
};
const attr = (node, name) => node.attrs?.find((attribute) => attribute.name === name)?.value;

test('aliases preserve native fragments and details ancestry without changing ToC metadata', async () => {
	const aliases = { version: 1, pages: { article: { '1-概述overview': 'kt-section-1', '旧推导': '新推导' } } };
	const renderer = await createMarkdownProcessor({ syntaxHighlight: false, rehypePlugins: [[rehypeHeadingAliases, { aliases, docsRoot, articleSlugs: ['article'] }]] });
	const result = await renderer.render('## 1. 概述\n\n<details class="kt-derivation"><summary>细节</summary>\n\n### 新推导\n\n内容\n\n</details>', { fileURL: pathToFileURL(resolve(docsRoot, 'article.md')) });
	assert.deepEqual(result.metadata.headings.map(({ slug, text }) => [slug, text]), [['kt-section-1', '1. 概述'], ['新推导', '新推导']]);
	const tree = parseFragment(result.code);
	assert.equal(collect(tree, (node) => attr(node, 'id') === '1-概述overview').length, 1);
	const details = collect(tree, (node) => node.tagName === 'details')[0];
	assert.equal(collect(details, (node) => attr(node, 'id') === '旧推导').length, 1);
	assert.equal(attr(collect(details, (node) => attr(node, 'id') === '旧推导')[0], 'aria-hidden'), 'true');
});

test('canonical module IDs preserve the meaning of an older subsection fragment with the same new title', async () => {
	const markdown = '## 1. 概述\n\n### 1. 架构定义';
	const options = { docsRoot, articleSlugs: ['article'] };
	const before = snapshot([['1-概述overview'], ['1-概述', 3]]);
	const currentRenderer = await createMarkdownProcessor({ syntaxHighlight: false, rehypePlugins: [[rehypeHeadingAliases, { ...options, aliases: { version: 1, pages: {} } }]] });
	const current = await currentRenderer.render(markdown, { fileURL: pathToFileURL(resolve(docsRoot, 'article.md')) });
	const aliases = deriveAliases(before, { version: 1, pages: { article: current.metadata.headings } });
	assert.deepEqual(aliases.pages.article, { '1-概述overview': 'kt-section-1', '1-概述': '1-架构定义' });
	const renderer = await createMarkdownProcessor({ syntaxHighlight: false, rehypePlugins: [[rehypeHeadingAliases, { ...options, aliases }]] });
	const result = await renderer.render(markdown, { fileURL: pathToFileURL(resolve(docsRoot, 'article.md')) });
	const tree = parseFragment(result.code);
	const h2 = collect(tree, (node) => node.tagName === 'h2')[0];
	const h3 = collect(tree, (node) => node.tagName === 'h3')[0];
	assert.equal(attr(h2, 'id'), 'kt-section-1');
	assert.equal(collect(h3, (node) => attr(node, 'id') === '1-概述').length, 1);
	assert.equal(collect(h2, (node) => attr(node, 'id') === '1-概述').length, 0);
	assert.equal(result.metadata.headings[0].text, '1. 概述');
	// Hand-written pages keep their ordinary title-derived heading IDs.
	const handWritten = await renderer.render(markdown, { fileURL: pathToFileURL(resolve(docsRoot, 'hand-written.md')) });
	assert.equal(handWritten.metadata.headings[0].slug, '1-概述');
});

test('same old IDs are isolated by article; raw HTML IDs and heading IDs are never duplicated', async () => {
	const aliases = { version: 1, pages: { article: { old: 'new', keep: 'new', new: 'new' }, second: { old: 'another' } } };
	const renderer = await createMarkdownProcessor({ syntaxHighlight: false, rehypePlugins: [[rehypeHeadingAliases, { aliases, docsRoot }]] });
	const first = await renderer.render('<span id="keep"></span>\n\n## new', { fileURL: pathToFileURL(resolve(docsRoot, 'article.md')) });
	const second = await renderer.render('## another', { fileURL: pathToFileURL(resolve(docsRoot, 'second.md')) });
	for (const result of [first, second]) {
		const ids = collect(parseFragment(result.code), (node) => attr(node, 'id') !== undefined).map((node) => attr(node, 'id'));
		assert.equal(new Set(ids).size, ids.length);
		assert.ok(ids.includes('old'));
	}
});

test('missing alias targets fail rather than publishing broken compatibility links', async () => {
	const aliases = { version: 1, pages: { article: { old: 'missing' } } };
	const transform = rehypeHeadingAliases({ aliases, docsRoot });
	assert.throws(() => transform({ type: 'root', children: [] }, { path: resolve(docsRoot, 'article.md'), history: [], data: {} }), /targets missing heading/);
});

test('title-only migration maps by stable order and retargets older generations', () => {
	const before = snapshot([['intro'], ['detail', 3], ['ending']]);
	const after = snapshot([['概述'], ['细节', 3], ['ending']]);
	const previous = { version: 1, pages: { article: { historical: 'intro' } } };
	assert.deepEqual(deriveAliases(before, after, previous).pages.article, { historical: '概述', intro: '概述', detail: '细节' });
	assert.deepEqual(previous.pages.article, { historical: 'intro' });
});

test('automatic migration rejects changed heading counts, levels, and observable reordering', () => {
	const before = snapshot([['one'], ['two', 3]]);
	assert.throws(() => deriveAliases(before, snapshot([['one']])), /count changed/);
	assert.throws(() => deriveAliases(before, snapshot([['one'], ['two', 2]])), /level changed/);
	assert.throws(() => deriveAliases(snapshot([['one'], ['two']]), snapshot([['two'], ['one']])), /order changed/);
});
