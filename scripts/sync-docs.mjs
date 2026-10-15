#!/usr/bin/env node
/**
 * sync-docs.mjs — Knowledge Tree V2 deterministic publishing pipeline.
 *
 * Source of truth: tech_stack/ originals + src/data/content-map.json.
 * Generates Starlight pages under src/content/docs/<domain>/...
 *  - frontmatter: title / description / editUrl (-> tech_stack original) / sidebar label+order / draft
 *  - first body H1 matching the curated title is absorbed into the page title;
 *    otherwise all body headings are demoted one level (fence-aware)
 *  - internal links to other originals are rewritten to /knowledge-tree/ slugs;
 *    curated external overrides (references/papers -> arXiv) applied
 *  - generated dirs get a .gitignore (*.md, !index.md): chapters are build-time
 *    artifacts, only hand-written index.md pages are tracked
 *  - stale pages from previous layouts are removed
 *
 * Flags: --check  verify freshness without writing (exit 1 on drift)
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, rmSync, unlinkSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { posix } from 'node:path';
import { DOMAINS } from '../src/data/domains.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const TS = join(ROOT, 'tech_stack');
const DOCS = join(ROOT, 'src/content/docs');
const MAP_PATH = join(ROOT, 'src/data/content-map.json');
const STATS_PATH = join(ROOT, 'src/data/content-stats.json');
const REPO = 'https://github.com/VicHxuan/knowledge-tree';
const SITE_BASE = '/knowledge-tree';
// 六大知识域目录由 src/data/domains.mjs 统一登记（单一事实源）
const TS_DOMAINS = DOMAINS.map((d) => d.tsDir);
const SITE_DOMAIN_DIRS = DOMAINS.map((d) => d.id);
const LEGACY_SITE_DIRS = ['rl-end-to-end', 'machine-learning', 'robotics', 'programming'];

// curated external link overrides: tech_stack-relative resolved path -> URL
const EXTERN = {
  'references/papers/PPO.pdf': 'https://arxiv.org/abs/1707.06347',
  'references/papers/SAC_v1.pdf': 'https://arxiv.org/abs/1801.01290',
};

const CHECK = process.argv.includes('--check');
const md5 = (s) => createHash('md5').update(s).digest('hex');

function walk(dir, out = []) {
  for (const name of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, name.name);
    if (name.isDirectory()) walk(p, out);
    else if (name.name.endsWith('.md')) out.push(p);
  }
  return out;
}

function tsRel(p) { // tech_stack-relative posix path
  return posix.normalize(relative(TS, p).split('\\').join('/'));
}

// ---------- content map ----------

function loadMap() {
  const map = JSON.parse(readFileSync(MAP_PATH, 'utf8'));
  if (map.version !== 2) throw new Error(`content-map version ${map.version} != 2`);
  const entries = map.entries;
  const bySource = new Map();
  const bySlug = new Map();
  for (const e of entries) {
    if (!e.source.startsWith('tech_stack/') || !e.output.startsWith('src/content/docs/')) {
      throw new Error(`bad map entry paths: ${e.source}`);
    }
    if (bySlug.has(e.slug)) throw new Error(`duplicate slug: ${e.slug}`);
    bySource.set(e.source.slice('tech_stack/'.length), e);
    bySlug.set(e.slug, e);
  }
  // coverage: every original in the six domains is mapped, and vice versa
  const onDisk = new Set();
  for (const d of TS_DOMAINS) {
    const dir = join(TS, d);
    if (!existsSync(dir)) throw new Error(`missing tech_stack domain dir: ${d}`);
    for (const f of walk(dir)) onDisk.add(tsRel(f));
  }
  const unmapped = [...onDisk].filter((p) => !bySource.has(p)).sort();
  const missing = [...bySource.keys()].filter((p) => !existsSync(join(TS, p))).sort();
  if (unmapped.length || missing.length) {
    for (const p of unmapped) console.error(`tech_stack file not in content-map: ${p}`);
    for (const p of missing) console.error(`content-map source missing on disk: ${p}`);
    throw new Error('content map coverage check failed');
  }
  return { entries, bySource, bySlug };
}

// ---------- markdown processing ----------

const norm = (s) => s
  .toLowerCase()
  .replace(/[\d\s\-_/\\.,;:!?()[\]{}"'`~@#$%^&*+=|<>、，。：；！？（）【】《》—…·「」『』]+/g, '');

function yamlScalar(s) {
  return JSON.stringify(s);
}

function frontmatter(e) {
  let fm = '---\n';
  fm += `title: ${yamlScalar(e.title)}\n`;
  fm += `description: ${yamlScalar(e.description)}\n`;
  fm += `editUrl: ${yamlScalar(`${REPO}/edit/main/${e.source}`)}\n`;
  fm += 'sidebar:\n';
  fm += `  label: ${yamlScalar(e.label)}\n`;
  fm += `  order: ${e.order}\n`;
  if (e.status === 'draft') fm += 'draft: true\n';
  fm += '---\n';
  return fm;
}

/** fence-aware per-line view: returns lines + inFence flags */
function markFences(lines) {
  const inFence = new Array(lines.length).fill(false);
  let open = null; // {ch, len}
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^\s{0,3}(`{3,}|~{3,})/);
    if (open) {
      if (m && m[1][0] === open.ch && m[1].length >= open.len && !/\S/.test(lines[i].slice(lines[i].indexOf(m[1]) + m[1].length))) {
        open = null; // closing fence: line itself is fence context
      }
      inFence[i] = true;
    } else if (m) {
      open = { ch: m[1][0], len: m[1].length };
      inFence[i] = true;
    }
  }
  return inFence;
}

function processBody(raw, e, bySource) {
  const lines = raw.split(/\r?\n/);
  const inFence = markFences(lines);
  const srcDir = posix.dirname(e.source.slice('tech_stack/'.length));

  // find first body H1 outside fences
  let h1Idx = -1;
  for (let i = 0; i < lines.length; i++) {
    if (!inFence[i] && /^#\s+\S/.test(lines[i])) { h1Idx = i; break; }
  }
  const absorb = h1Idx >= 0 && norm(lines[h1Idx].replace(/^#\s+/, '')) === norm(e.title);

  const out = [];
  for (let i = 0; i < lines.length; i++) {
    if (i === h1Idx && absorb) continue; // absorbed into page title
    let line = lines[i];
    if (!inFence[i]) {
      if (!absorb) {
        line = line.replace(/^#{1,5}(\s)/, (m) => `#${m}`); // demote headings one level
      }
      line = convertLinks(line, srcDir, bySource);
    }
    out.push(line);
  }
  // collapse a doubled blank line left by the removed H1
  for (let i = out.length - 2; i > 0; i--) {
    if (out[i] === '' && out[i - 1] === '' && out[i + 1] === '') out.splice(i, 1);
  }
  return out.join('\n');
}

function convertLinks(line, srcDir, bySource) {
  return line.replace(/\]\(([^)\s]*)\)/g, (whole, target) => {
    if (/^(https?:|mailto:|#)/.test(target)) return whole;
    let p = posix.normalize(posix.join(srcDir, target));
    if (p.startsWith('../')) return whole; // escapes tech_stack -> leave untouched
    if (EXTERN[p]) return `](${EXTERN[p]})`;
    const entry = bySource.get(p);
    if (entry) return `](${SITE_BASE}/${entry.slug}/)`;
    return whole;
  });
}

// ---------- generation ----------

function generate(map) {
  const written = [];
  for (const e of map.entries) {
    const srcPath = join(ROOT, e.source);
    const outPath = join(ROOT, e.output);
    const text = frontmatter(e) + '\n' + processBody(readFileSync(srcPath, 'utf8'), e, map.bySource) + '\n';
    written.push({ entry: e, outPath, text });
  }
  return written;
}

// ---------- housekeeping ----------

const GITIGNORE = '# 章节页面由 scripts/sync-docs.mjs 从 tech_stack/ 原件生成，属构建产物不入库。\n# 手写的 index.md 通过 !index.md 保留入库。\n*.md\n!index.md\n';

function housekeep(written) {
  const expected = new Set(written.map((w) => w.outPath));
  const genDirs = new Set(written.map((w) => dirname(w.outPath)));

  // legacy top-level dirs from previous layouts
  const removedLegacy = [];
  for (const d of LEGACY_SITE_DIRS) {
    const p = join(DOCS, d);
    if (existsSync(p)) { rmSync(p, { recursive: true }); removedLegacy.push(d); }
  }

  // stale generated pages inside the six site domains
  const stale = [];
  for (const d of SITE_DOMAIN_DIRS) {
    const dir = join(DOCS, d);
    if (!existsSync(dir)) continue;
    for (const f of walk(dir)) {
      if (!expected.has(f) && !f.endsWith('index.md')) stale.push(f);
    }
  }
  for (const d of SITE_DOMAIN_DIRS) {
    const dir = join(DOCS, d);
    if (!existsSync(dir)) continue;
    pruneEmpty(dir, expected);
  }

  return { stale, removedLegacy, genDirs };
}

function pruneEmpty(dir, expected) {
  for (const name of readdirSync(dir, { withFileTypes: true })) {
    if (!name.isDirectory()) continue;
    const p = join(dir, name.name);
    pruneEmpty(p, expected);
    const left = readdirSync(p);
    if (left.length === 1 && left[0] === '.gitignore') {
      const ig = join(p, '.gitignore');
      if (readFileSync(ig, 'utf8') === GITIGNORE) rmSync(p, { recursive: true });
    }
  }
}

function writeAll(written, house) {
  for (const w of written) {
    mkdirSync(dirname(w.outPath), { recursive: true });
    writeFileSync(w.outPath, w.text);
  }
  for (const f of house.stale) unlinkSync(f);
  for (const d of house.genDirs) {
    const ig = join(d, '.gitignore');
    if (!existsSync(ig) || readFileSync(ig, 'utf8') !== GITIGNORE) writeFileSync(ig, GITIGNORE);
  }
}

/** 原件最近一次提交的 author-date（YYYY-MM-DD）；无 git 历史时返回 null。 */
function gitAuthorDate(source) {
  try {
    const out = execFileSync('git', ['log', '-1', '--format=%aI', '--', source], {
      cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
    return /^\d{4}-\d{2}-\d{2}/.test(out) ? out.slice(0, 10) : null;
  } catch {
    return null;
  }
}

function stats(written) {
  const domains = {};
  const lastModified = {};
  for (const w of written) {
    const top = w.entry.slug.split('/')[0];
    domains[top] ??= { published: 0, draft: 0, total: 0 };
    domains[top][w.entry.status] += 1;
    domains[top].total += 1;
    const iso = gitAuthorDate(w.entry.source);
    if (iso) lastModified[w.entry.slug] = iso;
  }
  // 首页「最近更新」：已发布且能取到提交日期的前 5 篇，按日期倒序、slug 正序破平
  const recent = written
    .filter((w) => w.entry.status === 'published' && lastModified[w.entry.slug])
    .map((w) => ({
      slug: w.entry.slug,
      title: w.entry.title,
      domain: w.entry.slug.split('/')[0],
      date: lastModified[w.entry.slug],
    }))
    .sort((a, b) => (a.date === b.date ? a.slug.localeCompare(b.slug) : a.date < b.date ? 1 : -1))
    .slice(0, 5);
  const hash = md5(written.map((w) => w.entry.slug + ':' + md5(w.text)).sort().join('\n'));
  return {
    version: 2,
    entries: written.length,
    published: written.filter((w) => w.entry.status === 'published').length,
    draft: written.filter((w) => w.entry.status === 'draft').length,
    domains,
    recent,
    lastModified,
    contentHash: hash,
  };
}

// ---------- main ----------

function main() {
  const map = loadMap();
  const written = generate(map);
  const house = housekeep(written);

  if (CHECK) {
    let drift = 0;
    for (const w of written) {
      if (!existsSync(w.outPath)) { console.error(`missing: ${relative(ROOT, w.outPath)}`); drift++; continue; }
      if (readFileSync(w.outPath, 'utf8') !== w.text) { console.error(`stale:  ${relative(ROOT, w.outPath)}`); drift++; }
    }
    for (const f of house.stale) { console.error(`extra:  ${relative(ROOT, f)}`); drift++; }
    for (const d of house.removedLegacy) { console.error(`legacy dir still present: src/content/docs/${d}`); drift++; }
    if (drift) { console.error(`docs sync check FAILED (${drift} file(s) need sync)`); process.exit(1); }
    console.log(`docs sync check OK: ${written.length} pages in sync`);
    return;
  }

  writeAll(written, house);
  const s = stats(written);
  mkdirSync(dirname(STATS_PATH), { recursive: true });
  writeFileSync(STATS_PATH, JSON.stringify(s, null, 2) + '\n');
  console.log(`synced ${s.entries} pages (${s.published} published, ${s.draft} draft) from ${map.entries.length} map entries`);
  if (house.stale.length) console.log(`removed ${house.stale.length} stale generated page(s)`);
  if (house.removedLegacy.length) console.log(`removed legacy dir(s): ${house.removedLegacy.join(', ')}`);
  console.log(`content hash: ${s.contentHash}`);
}

main();
