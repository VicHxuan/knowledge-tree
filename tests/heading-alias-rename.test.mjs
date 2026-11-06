import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { renameHeadingAliasPage } from '../scripts/lib/heading-aliases.mjs';

const slug = 'learning-based-planning/rl-algorithms/ppo';
const target = `${slug}-guide`;
const source = 'tech_stack/03_learning-based-planning/02_rl-algorithms/04_ppo.md';
const moved = source.replace('04_ppo.md', '08_ppo.md');

test('heading alias route migration is immutable and leaves unchanged routes intact', () => {
	const data = { version: 1, pages: { [slug]: { overview: '概述' }, other: { old: 'new' } } };
	assert.deepEqual(renameHeadingAliasPage(data, slug), data);
	assert.deepEqual(renameHeadingAliasPage(data, slug, target).pages, { [target]: { overview: '概述' }, other: { old: 'new' } });
	assert.ok(Object.hasOwn(data.pages, slug));
	assert.ok(!Object.hasOwn(data.pages, target));
	assert.deepEqual(renameHeadingAliasPage(data, 'unmapped', 'new-page'), data);
	assert.throws(() => renameHeadingAliasPage(data, slug, 'other'), /已有标题别名冲突/);
});

function fixture(t, withConflict = false) {
	const root = mkdtempSync(join(tmpdir(), 'knowledge-tree-alias-rename-'));
	t.after(() => rmSync(root, { recursive: true, force: true }));
	const put = (name, text) => {
		const path = join(root, name);
		mkdirSync(join(path, '..'), { recursive: true });
		writeFileSync(path, text);
	};
	for (const name of ['scripts/docs-rename.mjs', 'scripts/lib/docs-model.mjs', 'scripts/lib/heading-aliases.mjs']) {
		put(name, readFileSync(new URL(`../${name}`, import.meta.url), 'utf8'));
	}
	put('src/data/domains.mjs', "export const DOMAINS = [{ tsDir: '03_learning-based-planning', id: 'learning-based-planning' }];\n");
	put(moved, '# PPO\n');
	put('src/data/content-map.json', JSON.stringify({ version: 2, entries: [{ source, slug, title: 'PPO', status: 'published' }] }));
	put('src/data/knowledge-relations.json', JSON.stringify({ version: 3, edges: [{ from: slug, to: 'related', type: 'related' }] }));
	put('src/data/redirects.json', JSON.stringify({ '/historical-ppo': `/${slug}` }));
	put('src/data/heading-aliases.json', JSON.stringify({ version: 1, pages: { [slug]: { overview: '概述' }, ...(withConflict ? { [target]: { earlier: '旧页面标题' } } : {}) } }));
	const run = (...args) => spawnSync(process.execPath, [join(root, 'scripts/docs-rename.mjs'), '--from', slug, '--to', moved, ...args], { cwd: root, encoding: 'utf8' });
	const json = (name) => JSON.parse(readFileSync(join(root, `src/data/${name}.json`), 'utf8'));
	return { root, run, json };
}

const snapshot = (root) => {
	const files = {};
	const walk = (dir) => {
		for (const entry of readdirSync(dir, { withFileTypes: true })) {
			const path = join(dir, entry.name);
			if (entry.isDirectory()) walk(path);
			else files[path] = readFileSync(path, 'utf8');
		}
	};
	walk(root);
	return files;
};

test('docs:rename keeps aliases for source-only moves and carries them across explicit slug changes', (t) => {
	const { run, json } = fixture(t);
	const originalAliases = json('heading-aliases');
	const stable = run();
	assert.equal(stable.status, 0, stable.stderr);
	assert.deepEqual(json('heading-aliases'), originalAliases);
	assert.equal(json('content-map').entries[0].slug, slug);
	assert.equal(json('content-map').entries[0].source, moved);
	const renamed = run('--new-slug', target);
	assert.equal(renamed.status, 0, renamed.stderr);
	assert.deepEqual(json('heading-aliases').pages, { [target]: { overview: '概述' } });
	assert.equal(json('content-map').entries[0].slug, target);
	assert.equal(json('knowledge-relations').edges[0].from, target);
	assert.equal(json('redirects')[`/${slug}`], `/${target}`);
	assert.equal(json('redirects')['/historical-ppo'], `/${target}`);
});

test('docs:rename rejects a heading-alias target conflict before writing any file', (t) => {
	const { root, run } = fixture(t, true);
	const before = snapshot(root);
	const result = run('--new-slug', target);
	assert.equal(result.status, 1);
	assert.match(result.stderr, /已有标题别名冲突/);
	assert.deepEqual(snapshot(root), before);
});
