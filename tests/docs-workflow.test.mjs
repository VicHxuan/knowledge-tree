import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, readdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { createMarkdownProcessor } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { deriveSlug, rewriteSourceTarget, renameRegistration, flattenRedirects } from '../scripts/lib/docs-model.mjs';
import { planGeneratedFiles, applyFilePlan, checkFilePlan, GENERATED_MARKER } from '../scripts/lib/generated-files.mjs';
import { validateSite } from '../scripts/validate-site.mjs';
import { readArticleModules } from '../scripts/lib/article-structure.mjs';

const domains = [{ tsDir: '03_learning-based-planning', id: 'learning-based-planning' }];
const source = 'tech_stack/03_learning-based-planning/02_rl-algorithms/02_model-free-rl/04_ppo.md';
const slug = 'learning-based-planning/rl-algorithms/model-free-rl/ppo';

function fixture(t) {
  const dir = mkdtempSync(join(tmpdir(), 'knowledge-tree-test-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return dir;
}
function put(root, name, text) {
  const path = join(root, name);
  mkdirSync(join(path, '..'), { recursive: true });
  writeFileSync(path, text);
  return path;
}
function snapshot(dir) {
  const result = {};
  const visit = (path) => { for (const e of readdirSync(path, { withFileTypes: true })) { const p = join(path, e.name); if (e.isDirectory()) visit(p); else result[p] = readFileSync(p, 'utf8'); } };
  visit(dir);
  return result;
}

test('nested paths preserve every section and source links preserve query, fragment and external URLs', () => {
  assert.equal(deriveSlug(source, domains), slug);
  const map = new Map([[source.slice('tech_stack/'.length), { slug }]]);
  const dir = '03_learning-based-planning/02_rl-algorithms/02_model-free-rl';
  assert.equal(rewriteSourceTarget('04_ppo.md?view=full#推导', dir, map), `/knowledge-tree/${slug}/?view=full#推导`);
  assert.equal(rewriteSourceTarget('https://example.org/a.md#x', dir, map), 'https://example.org/a.md#x');
  assert.equal(rewriteSourceTarget('#local', dir, map), '#local');
});

test('source moves preserve URLs; explicit rename flattens historical aliases and updates graph edges', () => {
  const map = { entries: [{ source, slug, status: 'published' }] };
  const relations = { version: 3, edges: [{ from: slug, to: 'target', type: 'related' }] };
  const moved = source.replace('04_ppo.md', '08_ppo.md');
  const stable = renameRegistration(map, relations, { '/ancient': `/${slug}` }, { from: slug, source: moved }, domains);
  assert.equal(stable.entry.slug, slug);
  assert.equal(stable.entry.source, moved);
  const renamed = `${slug}-guide`;
  const result = renameRegistration(map, relations, stable.redirects, { from: slug, source: moved, slug: renamed }, domains);
  assert.equal(result.redirects['/ancient'], `/${renamed}`);
  assert.equal(result.redirects[`/${slug}`], `/${renamed}`);
  assert.equal(result.relations.edges[0].from, renamed);
  assert.throws(() => flattenRedirects({ '/a': '/b', '/b': '/a' }), /环/);
});

test('check is read-only and cleanup preserves hand-written files and custom ignore rules', (t) => {
  const root = fixture(t);
  const docs = join(root, 'docs');
  const hand = put(docs, 'legacy/custom.md', '# Hand-written page\n');
  const index = put(docs, 'legacy/index.mdx', '# Hand-written index\n');
  const stale = put(docs, 'legacy/stale.md', `---\n${GENERATED_MARKER}\n---\nOld generated page\n`);
  const ignore = put(docs, 'new/.gitignore', '/private.txt\n');
  const current = join(docs, 'new/ppo.md');
  const text = `---\n${GENERATED_MARKER}\n---\nNew generated page\n`;
  const plan = planGeneratedFiles({ docs, generated: [{ path: current, content: text }], articlePaths: [current] });
  const before = snapshot(root);
  assert.ok(checkFilePlan(plan).length);
  assert.deepEqual(snapshot(root), before);
  applyFilePlan(plan);
  assert.ok(existsSync(hand));
  assert.ok(existsSync(index));
  assert.ok(!existsSync(stale));
  assert.match(readFileSync(ignore, 'utf8'), /\/private\.txt/);
  assert.match(readFileSync(ignore, 'utf8'), /\/ppo\.md/);
  const fresh = planGeneratedFiles({ docs, generated: [{ path: current, content: text }], articlePaths: [current] });
  assert.deepEqual(checkFilePlan(fresh), []);
});

test('sync CLI checks statistics, ignores and public downloads without changing the fixture', (t) => {
  const root = fixture(t);
  for (const name of ['scripts/sync-docs.mjs', 'scripts/lib/docs-model.mjs', 'scripts/lib/generated-files.mjs']) {
    put(root, name, readFileSync(new URL(`../${name}`, import.meta.url), 'utf8'));
  }
  put(root, 'src/data/domains.mjs', `export const DOMAINS = ${JSON.stringify(domains)};`);
  put(root, source, '# PPO\n\n[Proof](04_ppo.md?view=full#proof)\n');
  put(root, 'examples/ppo_core.py', '# public teaching example\n');
  put(root, 'src/data/content-map.json', JSON.stringify({ version: 2, entries: [{ source, slug, output: `src/content/docs/${slug}.md`, title: 'PPO', label: 'PPO', order: 1, status: 'published', description: 'PPO example' }] }));
  put(root, 'src/content/docs/rl-end-to-end/hand.md', '# Legacy hand-written page\n');
  const run = (...args) => spawnSync(process.execPath, [join(root, 'scripts/sync-docs.mjs'), ...args], { cwd: root, encoding: 'utf8' });
  assert.equal(run().status, 0);
  assert.equal(run('--check').status, 0);
  assert.equal(readFileSync(join(root, 'public/examples/ppo_core.py'), 'utf8'), '# public teaching example\n');
  put(root, 'src/data/content-stats.json', '{}\n');
  put(root, 'public/examples/ppo_core.py', '# stale copy\n');
  put(root, `src/content/docs/${slug.split('/').slice(0, -1).join('/')}/.gitignore`, '# custom ignore only\n');
  const before = snapshot(root);
  const check = run('--check');
  assert.equal(check.status, 1);
  assert.match(check.stderr, /content-stats\.json/);
  assert.match(check.stderr, /public\/examples\/ppo_core\.py/);
  assert.match(check.stderr, /\.gitignore/);
  assert.deepEqual(snapshot(root), before);
  assert.ok(existsSync(join(root, 'src/content/docs/rl-end-to-end/hand.md')));
});

test('built HTML checker covers MDX-generated links, local assets, query strings and closed detail anchors', (t) => {
  const dist = fixture(t);
  put(dist, 'index.html', '<a href="/knowledge-tree/article/?a=1#%E6%8E%A8%E5%AF%BC">Go</a><img src="/knowledge-tree/media/a.svg"><a href="https://elsewhere.test/missing">External</a>');
  put(dist, 'article/index.html', '<details><summary>More</summary><h3 id="推导">Proof</h3></details>');
  put(dist, 'media/a.svg', '<svg></svg>');
  assert.deepEqual(validateSite(dist).errors, []);
  put(dist, 'index.html', '<a href="/knowledge-tree/article/#missing">Bad fragment</a><img src="/knowledge-tree/missing.png">');
  const errors = validateSite(dist).errors;
  assert.equal(errors.length, 2);
  assert.ok(errors.some((e) => e.includes('missing fragment')));
  assert.ok(errors.some((e) => e.includes('broken src')));
  assert.ok(validateSite(dist, [{ slug: 'article', status: 'draft' }]).errors.some((e) => e.includes('draft emitted')));
});

test('built article structure rejects extra Setext and HTML headings that source ATX checks miss', async (t) => {
  const dist = fixture(t);
  const processor = await createMarkdownProcessor();
  const markdown = readArticleModules().map((heading) => `## ${heading}\n\nModule text.`).join('\n\n');
  const entries = [{ slug: 'article', status: 'published' }];
  const emit = async (body) => {
    const { code } = await processor.render(body);
    put(dist, 'article/index.html', `<aside><h2>Navigation outside the article</h2></aside><div class="sl-markdown-content">${code}</div>`);
  };
  await emit(markdown);
  assert.deepEqual(validateSite(dist, entries).errors, []);
  for (const extra of ['\\frac{w_v}{\\Delta t}g_{k-1}\n-', '<h2>Unexpected HTML heading</h2>']) {
    await emit(`${markdown}\n\n${extra}\n`);
    const errors = validateSite(dist, entries).errors;
    assert.equal(errors.length, 1);
    assert.match(errors[0], /rendered H2 module sequence mismatch.*found 7/);
  }
});

test('built HTML checker reports the page and KaTeX parse error from failed formulas', async (t) => {
  const dist = fixture(t);
  const processor = await createMarkdownProcessor({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] });
  const emit = async (markdown) => {
    const { code } = await processor.render(markdown);
    put(dist, 'article/index.html', `<div class="sl-markdown-content">${code}</div>`);
  };
  await emit(String.raw`$D_{\mathrm{KL}}\!left(\pi\right)$`);
  const errors = validateSite(dist).errors;
  assert.equal(errors.length, 1);
  assert.match(errors[0], /^article\/index\.html: KaTeX error: .*KaTeX parse error:/);
  assert.match(errors[0], /\\right/);
  await emit(String.raw`$D_{\mathrm{KL}}\!\left(\pi\right)$`);
  assert.deepEqual(validateSite(dist).errors, []);
});
